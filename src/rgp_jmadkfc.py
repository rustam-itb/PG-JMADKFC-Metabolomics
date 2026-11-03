"""RGP-JMADKFC, prototype-separated and reconstruction-guarded adaptive-kernel FCM.

Research candidate (not a proven improvement). No external labels are accessed in fit.
Based on the original reviewer worker; retains its data/model/loss definitions.
"""
import numpy as np
import torch
from JMADKFC_diagnostic_worker import ReviewerJMADKFC


class RGPJMADKFC(ReviewerJMADKFC):
    """Two-cluster stability-aware candidate; hyperparameters fixed without label tuning.

    Modes: candidate, no_guard, no_projection, no_variance, no_mask,
           shuffled_mask, original_full. All modes share the same architecture.

    Prototype constraint: ||v1-v0|| >= min_sep_fraction * initial separation.
    Encoder gradient conflict projection and cap are applied prior to Adam;
    this is only a first-order direction safeguard, not a monotonicity guarantee.
    """
    MODES = ('candidate','no_guard','no_projection','no_variance','no_mask',
             'shuffled_mask','original_full')

    def __init__(self, *, mode='candidate', min_sep_fraction=0.25,
                 mask_scale=0.25, grad_cap=0.5, variance_floor=0.5,
                 variance_weight=0.1, **kwargs):
        if mode not in self.MODES: raise ValueError('Invalid mode')
        if not (0. < min_sep_fraction <= 1.): raise ValueError('min_sep_fraction out of range')
        if not (0. <= mask_scale <= 2.): raise ValueError('mask_scale out of range')
        if not (0. < grad_cap <= 5.): raise ValueError('grad_cap out of range')
        if not (0. <= variance_floor <= 1.): raise ValueError('variance_floor out of range')
        if variance_weight < 0: raise ValueError('variance_weight must be nonnegative')
        self.mode=mode
        self.min_sep_fraction=float(min_sep_fraction)
        self.mask_scale=float(mask_scale)
        self.grad_cap=float(grad_cap)
        self.variance_floor=float(variance_floor)
        self.variance_weight=float(variance_weight)
        self.std_ref_=None
        self.trace_=[]
        self.projection_count_=0
        self.initial_mask_=None
        super().__init__(variant='Full', **kwargs)
        if self.n_clusters != 2:
            raise ValueError('Projection of prototypes is implemented for c=2 only')

    def _prepare_inputs(self, X_missing, mask=None):
        Xzero, M, Xt=super()._prepare_inputs(X_missing,mask)
        self.initial_mask_=M.copy()
        if self.mode=='original_full':
            return Xzero,M,Xt
        if self.mode=='no_mask':
            masked_part=np.zeros_like(M)
        elif self.mode=='shuffled_mask':
            rng=np.random.default_rng(self.random_state+735141)
            shuffled=np.column_stack([rng.permutation(M[:,j]) for j in range(M.shape[1])]).astype(np.float32)
            masked_part=shuffled-shuffled.mean(axis=0,keepdims=True)
        else:
            masked_part=M-M.mean(axis=0,keepdims=True)
        Xt=np.concatenate([Xzero,self.mask_scale*masked_part],axis=1).astype(np.float32)
        return Xzero,M,Xt

    def _sample_bandwidths_np(self, Z):
        # First invocation follows FCM initialization, before joint encoder updates.
        if self.std_ref_ is None:
            self.std_ref_=np.std(np.asarray(Z),axis=0,ddof=0).astype(np.float32)
        return super()._sample_bandwidths_np(Z)

    def _stabilize_prototypes(self,Z,U,V,epoch):
        if self.mode in ('no_projection','original_full'):
            return V
        delta=max(self.min_sep_fraction*self._initial_proto_separation,1e-6)
        vec=V[1]-V[0]
        dist=float(np.linalg.norm(vec))
        if dist+1e-12>=delta:
            return V
        if dist<=1e-10:
            # Use initial direction if available; otherwise principal component.
            orient=getattr(self,'_last_valid_direction',None)
            if orient is None:
                center=Z-Z.mean(axis=0)
                orient=np.linalg.svd(center,full_matrices=False)[2][0]
        else:
            orient=vec/dist
        orient=orient/(np.linalg.norm(orient)+1e-12)
        center=(V[0]+V[1])*0.5
        self.projection_count_+=1
        return np.vstack([center-0.5*delta*orient,center+0.5*delta*orient])

    @staticmethod
    def _flat(grads,params):
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for g,p in zip(grads,params)])

    def _optimizer_step_diagnostic(self, optimizer, rec_loss, akf_loss, reg_loss,
                                   lam, Z, U, V, sigma_i, epoch):
        if self.mode=='original_full':
            (rec_loss+lam*akf_loss+self.beta_reg*reg_loss).backward()
            optimizer.step()
            self.trace_.append(dict(epoch=epoch+1,rec=float(rec_loss.detach()),
                                    cluster=float(akf_loss.detach()),
                                    prototype_separation=float(np.linalg.norm(V[1]-V[0]))))
            return
        enc=list(self.model_.encoder.parameters())
        dec=list(self.model_.decoder.parameters())
        std=torch.sqrt(torch.var(Z,dim=0,unbiased=False)+1e-8)
        ref=torch.as_tensor(self.std_ref_,device=Z.device,dtype=Z.dtype)
        floor=self.variance_floor*ref
        var_loss=torch.mean(torch.relu(floor-std)**2)
        include_var = self.mode != 'no_variance'
        combined_cl=akf_loss+(self.variance_weight*var_loss if include_var else 0.)
        grad_rec=torch.autograd.grad(rec_loss,enc,retain_graph=True,allow_unused=True)
        grad_cl=torch.autograd.grad(combined_cl,enc,retain_graph=True,allow_unused=True)
        vr=self._flat(grad_rec,enc)
        vc=self._flat(grad_cl,enc)
        nr=torch.linalg.vector_norm(vr)
        nc=torch.linalg.vector_norm(vc)
        dot=torch.dot(vr,vc)
        cosine=(dot/(nr*nc+1e-12)).detach().item()
        guarded=self.mode!='no_guard'
        if guarded:
            # One-sided projection of clustering-related gradient onto halfspace
            # orthogonal to conflicting reconstruction-gradient component.
            vc=vc-(torch.minimum(dot,torch.zeros_like(dot))/(nr*nr+1e-12))*vr
            cap=torch.clamp(self.grad_cap*nr/(torch.linalg.vector_norm(vc)+1e-12),max=1.0)
            vc=vc*cap
        genc=vr+lam*vc
        greg_enc=torch.autograd.grad(reg_loss,enc,retain_graph=True,allow_unused=True)
        gdec=torch.autograd.grad(rec_loss+self.beta_reg*reg_loss,dec,allow_unused=True)
        greg=self._flat(greg_enc,enc)
        genc=genc+self.beta_reg*greg
        offset=0
        for p in enc:
            m=p.numel()
            p.grad=genc[offset:offset+m].reshape_as(p).detach().clone()
            offset+=m
        for p,g in zip(dec,gdec):
            p.grad=(torch.zeros_like(p) if g is None else g).detach().clone()
        optimizer.step()
        if epoch in (0,24,49,99,149,199,self.joint_epochs-1):
            um=np.clip(U,1e-12,1.0)
            entropy=float(np.mean(-np.sum(um*np.log(um),axis=1))/np.log(2))
            self.trace_.append(dict(epoch=epoch+1,rec=float(rec_loss.detach()),
               cluster=float(akf_loss.detach()),variance_loss=float(var_loss.detach()),
               proto_separation=float(np.linalg.norm(V[1]-V[0])),
               membership_entropy=entropy,gradient_cosine=float(cosine),
               guarded=guarded))

    def fit(self,X_missing,mask=None):
        super().fit(X_missing,mask)
        final_sep=float(np.linalg.norm(self.centers_[0]-self.centers_[1]))
        bound=self.min_sep_fraction*self._initial_proto_separation
        if self.mode not in ('no_projection','original_full') and final_sep+1e-6<max(bound,1e-6):
            raise AssertionError('Prototype separation constraint violated')
        if not np.isfinite(self.membership_).all():
            raise FloatingPointError('Nonfinite fuzzy memberships')
        assert np.allclose(self.membership_.sum(axis=1),1,atol=1e-6)
        um=np.clip(self.membership_,1e-12,1)
        self.entropy_=float(np.mean(-np.sum(um*np.log(um),axis=1))/np.log(2))
        self.separation_=final_sep
        self.collapse_=bool(self.entropy_>0.99)
        return self
