"""Paper-guided, independently coded reimplementations (not author code).
Yang et al. PeerJ CS 2023 NPS-WKFCM: iterative NPS replacement,
featurewise Gaussian kernel, cluster-specific feature weights.
Li & Pimentel-Alarcon arXiv 2602.00262 CSC: disjoint masks, NT-Xent,
backbone encoder + sparse subspace clustering (SSC).
Both operate without accessing external class labels.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize

EPS=1e-10

def memberships_from_distances(D,m=2.):
    D=np.maximum(np.asarray(D,dtype=float),EPS)
    A=D**(-1./(m-1.))
    return A/np.maximum(A.sum(axis=1,keepdims=True),EPS)

def make_disjoint_views(mask,rng):
    """Partition observed values into disjoint views, each nonempty per row."""
    mask=np.asarray(mask,dtype=bool)
    ma=np.zeros_like(mask); mb=np.zeros_like(mask)
    for i in range(mask.shape[0]):
        pos=np.flatnonzero(mask[i]); rng.shuffle(pos)
        if len(pos)<2:
            raise ValueError('Need at least two observed features per sample')
        cut=max(1,min(len(pos)-1,len(pos)//2))
        ma[i,pos[:cut]]=True; mb[i,pos[cut:]]=True
    assert np.all(~(ma&mb)) and np.all((ma|mb)<=mask)
    return ma,mb

def nt_xent(z1,z2,temperature=0.25):
    n=z1.shape[0]
    z=F.normalize(torch.cat([z1,z2],dim=0),dim=1)
    sim=z@z.T/temperature
    sim.fill_diagonal_(-1e9)
    targets=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(sim,targets)

class NPSWKFCMReimplementation:
    """Per-feature weighted RBF fuzzy c means with NPS missing completion.
    Implementation of published *update structure*, not verified byte-for-byte
    author algorithm. Kernel = exp(-(x-v)^2 / sigma2). Weight exponent beta=2.
    """
    def __init__(self,n_clusters=2,m=2.,beta=2.,sigma2=2.,max_iter=80,
                 tol=1e-5,random_state=42,weight_floor=1e-3):
        self.c=n_clusters;self.m=m;self.beta=beta;self.sigma2=sigma2
        self.max_iter=max_iter;self.tol=tol;self.random_state=random_state
        self.weight_floor=weight_floor
    def _kernel(self,X,V):
        return np.exp(-np.clip(((X[:,None,:]-V[None,:,:])**2)/self.sigma2,0,100))
    def _distance(self,K,W):
        return (2*(1.-K)*(W[None,:,:]**self.beta)).sum(axis=2)
    def fit(self,X,mask=None):
        X=np.asarray(X,float)
        M=np.isfinite(X) if mask is None else np.asarray(mask,dtype=bool)
        if X.ndim!=2 or X.shape!=M.shape or np.any(M.sum(axis=0)==0): raise ValueError('invalid input')
        n,d=X.shape;c=self.c
        # Mean completion using observed entries only, NPS then iteratively overwrites.
        mu=np.sum(np.where(M,X,0.),axis=0)/M.sum(axis=0)
        completed=np.where(M,X,mu[None,:]).copy()
        rng=np.random.default_rng(self.random_state)
        U=memberships_from_distances(rng.uniform(0.1,1,(n,c)),self.m)
        V=np.vstack([completed[rng.choice(n)] for _ in range(c)])
        W=np.full((c,d),1./d)
        for it in range(self.max_iter):
            Um=U**self.m
            K=self._kernel(completed,V)
            # Prototype update: kernel-affinity weighted mean per coordinate.
            weights=Um[:,:,None]*K
            V=(weights*completed[:,None,:]).sum(axis=0)/np.maximum(weights.sum(axis=0),EPS)
            # Each cluster's feature weights normalized across features (Eq 3.22).
            K=self._kernel(completed,V)
            scatter=(Um[:,:,None]*(1.-K)).sum(axis=0)+EPS
            inv=scatter**(-1./(self.beta-1.))
            inv=np.maximum(inv,self.weight_floor*inv.mean(axis=1,keepdims=True))
            W=inv/np.maximum(inv.sum(axis=1,keepdims=True),EPS)
            D=self._distance(K,W)
            U_new=memberships_from_distances(D,self.m)
            # Nearest-prototype completion. Only masked entries may change.
            nearest=np.argmin(D,axis=1)
            completed=np.where(M,X,V[nearest])
            if np.max(np.abs(U_new-U))<self.tol:
                U=U_new;break
            U=U_new
        # Final E/M consistency update using final completed matrix.
        K=self._kernel(completed,V)
        D=self._distance(K,W)
        U=memberships_from_distances(D,self.m)
        self.labels_=U.argmax(axis=1);self.membership_=U
        self.centers_=V;self.feature_weights_=W
        self.X_completed_=completed;self.iterations_=it+1
        self.objective_=float(np.sum(U**self.m*D))
        return self

class ContrastiveBackbone(nn.Module):
    def __init__(self,d,latent=10):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(2*d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,latent))
        self.projector=nn.Sequential(nn.Linear(latent,32),nn.ReLU(),nn.Linear(32,latent))
    def forward(self,x):
        z=self.encoder(x)
        return z,self.projector(z)

class CSCReimplementation:
    """CSC-style training followed by SSC. No author code/benchmark equivalence claimed.
    MLP width/epochs and SSC solver adapted for n<=184 tabular samples.
    """
    def __init__(self,n_clusters=2,latent_dim=10,epochs=100,lr=1e-3,
                 temperature=.25,ssc_alpha=.005,random_state=42):
        self.c=n_clusters;self.latent_dim=latent_dim;self.epochs=epochs;self.lr=lr
        self.temperature=temperature;self.ssc_alpha=ssc_alpha;self.random_state=random_state
    def fit(self,X,mask=None):
        X=np.asarray(X,float);M=np.isfinite(X) if mask is None else np.asarray(mask,dtype=bool)
        X0=np.where(M,X,0.).astype(np.float32)
        torch.manual_seed(self.random_state);rng=np.random.default_rng(self.random_state)
        self.net_=ContrastiveBackbone(X.shape[1],self.latent_dim)
        opt=torch.optim.Adam(self.net_.parameters(),lr=self.lr)
        M0=M.astype(np.float32)
        for _ in range(self.epochs):
            ma,mb=make_disjoint_views(M,rng)
            xa=np.concatenate([X0*ma,ma.astype(np.float32)],axis=1)
            xb=np.concatenate([X0*mb,mb.astype(np.float32)],axis=1)
            _,za=self.net_(torch.from_numpy(xa))
            _,zb=self.net_(torch.from_numpy(xb))
            loss=nt_xent(za,zb,self.temperature)
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            Z=self.net_.encoder(torch.from_numpy(np.concatenate([X0,M0],axis=1))).numpy()
        self.embedding_=Z
        return self._fit_ssc(Z)
    def _fit_ssc(self,Z):
        # Columnwise self-expressiveness, zero diagonal, symmetric affinity.
        n=len(Z)
        Z=normalize(Z-Z.mean(axis=0),axis=1)
        A=np.zeros((n,n))
        for i in range(n):
            Xdict=Z.copy();Xdict[i,:]=0.
            model=Lasso(alpha=self.ssc_alpha,fit_intercept=False,max_iter=1500,tol=1e-4)
            model.fit(Xdict.T,Z[i])
            w=model.coef_;w[i]=0.;A[i]=np.abs(w)
        A=(A+A.T)/2.
        np.fill_diagonal(A,0)
        if A.max()<1e-8:
            # Explicit, documented fallback for empty SSC graph.
            from sklearn.cluster import KMeans
            pred=KMeans(self.c,n_init=10,random_state=self.random_state).fit_predict(Z)
            self.ssc_fallback_=True
        else:
            # Avoid fully disconnected singleton vertices; no labels accessed.
            isolated=np.where(A.sum(axis=1)<1e-10)[0]
            if len(isolated):
                cosine=Z@Z.T
                for i in isolated:
                    cosine[i]=-np.inf
                    j=int(np.argmax(cosine[i]));A[i,j]=A[j,i]=1e-3
            pred=SpectralClustering(self.c,affinity='precomputed',assign_labels='kmeans',n_init=10,random_state=self.random_state).fit_predict(A)
            self.ssc_fallback_=False
        self.affinity_=A;self.labels_=pred
        return self
