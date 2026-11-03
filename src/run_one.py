"""Single immutable, label-blind fit with final post-fit phenotype scoring.
Usage: python src/run_one.py --dataset MTBLS161 --mechanism MAR --rate 0.2 --seed 1061 --method PG-JMADKFC
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import json, argparse, time, random, hashlib, platform
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from JMADKFC_diagnostic_worker import (
    generate_missingness, observed_normalize, set_torch_seed, FuzzyCMeans,
    ReviewerJMADKFC, original_baseline_by_name, MASK_BASELINES,
    external_clustering_metrics)
from rgp_jmadkfc import RGPJMADKFC
from nps_factorial import FactorialNPSJMADKFC
from pg_jmadkfc import PGJMADKFC
from freeze_protocol import CFG, check
ROOT=Path(__file__).resolve().parents[1]

def make_model_kwargs(d, seed, pretrain, joint):
    return dict(n_clusters=2,alpha=2.,latent_dim=min(10,max(2,d//4)),
                hidden_dims=(128,64),lambda_cluster=1.,beta_reg=1e-6,
                q_neighbors=5,pretrain_epochs=pretrain,joint_epochs=joint,
                learning_rate=1e-3,tol=0.,prototype_inner_iter=1,
                random_state=int(seed),dropout=0.,device='cpu',verbose=False)

def make_setting(name,mechanism,rate,seed):
    p=ROOT/'data'/(name+'_prepared.npz')
    with np.load(p,allow_pickle=False) as d:
        X_log=np.asarray(d['X_log'],float)
        y=np.asarray(d['y'],np.int64)
        source_hash=str(d['source_sha256'])
    _,M=generate_missingness(X_log,mechanism,rate,random_state=int(seed),gamma=3.0)
    M=np.asarray(M,dtype=bool)
    setting=observed_normalize(X_log,M)
    assert np.array_equal(M, np.isfinite(setting['X_obs']))
    return y, M, setting, source_hash

def run_fit(dataset,mechanism,rate,seed,method,pretrain=100,joint=200,smoke=False):
    if dataset not in CFG['data'] or method not in CFG['methods']:
        raise ValueError('Unexpected dataset/method')
    if mechanism not in CFG['mechanisms'] or float(rate) not in CFG['rates'] and not smoke:
        raise ValueError('Setting not in precommitted protocol')
    if int(seed) not in CFG['seeds'] and not smoke:
        raise ValueError('Seed not in precommitted protocol')
    if (pretrain,joint)!=(CFG['pretrain_epochs'],CFG['joint_epochs']) and not smoke:
        raise ValueError('Epoch mismatch: only smoke tests may use reduced epochs')
    y,M,S,source_sha=make_setting(dataset,mechanism,rate,seed)
    X=S['X_obs']; d=X.shape[1]
    torch.set_num_threads(1)
    random.seed(int(seed));np.random.seed(int(seed));set_torch_seed(int(seed))
    kw=make_model_kwargs(d,seed,pretrain,joint)
    started=time.perf_counter()
    if method=='PG-JMADKFC':
        model=PGJMADKFC(mask_scale=.25,grad_cap=.5,min_sep_fraction=.25,
                        variance_floor=.5,variance_weight=.1,
                        burnin_epochs=50,completion_eta=.25,completion_clip=5.,**kw)
        pred=model.fit_predict(X,M)
    elif method=='Full-JMADKFC':
        model=ReviewerJMADKFC(variant='Full',**kw)
        pred=model.fit_predict(X,M)
    elif method=='RGP-JMADKFC':
        model=RGPJMADKFC(mode='candidate',mask_scale=.25,grad_cap=.5,
            min_sep_fraction=.25,variance_floor=.5,variance_weight=.1,**kw)
        pred=model.fit_predict(X,M)
    elif method=='RGP-NPS':
        model=FactorialNPSJMADKFC(nps_completion=True,feature_weighting=False,
            mask_scale=.25,grad_cap=.5,min_sep_fraction=.25,
            variance_floor=.5,variance_weight=.1,**kw)
        pred=model.fit_predict(X,M)
    elif method=='PCA-FCM':
        # No sample labels used in PCA or FCM; missing entries zero after observed-only z-scoring.
        Xzero=np.where(M,X,0.)
        z=PCA(n_components=min(10,len(y)-1,d),svd_solver='full').fit_transform(Xzero)
        model=FuzzyCMeans(2,alpha=2.,max_iter=300,tol=1e-5,random_state=int(seed))
        pred=model.fit_predict(z)
    else:
        model=original_baseline_by_name(method,2,int(seed),d,pretrain,joint)
        pred=(model.fit_predict(X,M) if method in MASK_BASELINES
              else model.fit_predict(X))
    elapsed=time.perf_counter()-started
    pred=np.asarray(pred).astype(int)
    if pred.shape!=y.shape:raise ValueError('Invalid cluster prediction dimensions')
    # These reference labels and hidden positions are used ONLY after model fitting.
    metric=external_clustering_metrics(y,pred)
    U=getattr(model,'membership_',None)
    entropy=occupancy=None
    if U is not None:
        U=np.asarray(U,float)
        if U.shape==(len(y),2):
            assert np.allclose(U.sum(axis=1),1,atol=1e-5)
            entropy=float(np.mean(-np.sum(U*np.log(np.clip(U,1e-12,1)),axis=1))/np.log(2))
            occ=U.mean(axis=0)
            occupancy=float(-np.sum(occ*np.log(np.clip(occ,1e-12,1)))/np.log(2))
    rec=getattr(model,'X_reconstructed_',None)
    rec_mse=float(np.mean((np.asarray(rec)[M]-X[M])**2)) if rec is not None else None
    comp=getattr(model,'completion_',None)
    cmse=(float(np.mean((np.asarray(comp)[~M]-S['X_truth_scaled'][~M])**2))
          if comp is not None and (~M).any() else None)
    result=dict(dataset=dataset,mechanism=mechanism,rate=float(rate),seed=int(seed),
        method=method,method_family=('proposed' if method=='PG-JMADKFC' else 'baseline'),
        n_samples=int(len(y)),n_features=int(d),label_counts=[int((y==j).sum()) for j in (0,1)],
        dataset_source_sha256=source_sha,
        mask_sha256=hashlib.sha256(M.astype(np.uint8).tobytes()).hexdigest(),
        missing_fraction_actual=float((~M).mean()),
        code_version=CFG['version'],
        pretrain_epochs=int(pretrain),joint_epochs=int(joint),
        ARI=float(metric['ARI']),NMI=float(metric['NMI']),ACC=float(metric['ACC']),
        runtime_seconds=float(elapsed),fuzzy_entropy=entropy,occupancy_entropy=occupancy,
        prototype_separation=(float(model.separation_) if hasattr(model,'separation_') else None),
        projection_count=(int(model.projection_count_) if hasattr(model,'projection_count_') else None),
        dynamic_updates=(int(model.dynamic_updates_) if hasattr(model,'dynamic_updates_') else None),
        observed_reconstruction_mse=rec_mse,posthoc_missing_completion_mse=cmse,
        software={'python':platform.python_version(),'torch':str(torch.__version__),
                  'numpy':np.__version__},status='success')
    if method=='PG-JMADKFC':
        result['completion_trace']=model.completion_log_
    for k in ('ARI','NMI','ACC'):
        if not np.isfinite(result[k]):raise FloatingPointError(f'Nonfinite {k}')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--mechanism',required=True)
    ap.add_argument('--rate',type=float,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--method',required=True)
    ap.add_argument('--pretrain',type=int,default=100)
    ap.add_argument('--joint',type=int,default=200)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--output',required=True)
    v=ap.parse_args()
    lock=check()
    r=run_fit(v.dataset,v.mechanism,v.rate,v.seed,v.method,v.pretrain,v.joint,v.smoke)
    r['protocol_fingerprint']=lock['fingerprint']
    p=Path(v.output);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(p)
    print('SUCCESS',v.dataset,v.mechanism,v.rate,v.seed,v.method,'ARI',round(r['ARI'],5),flush=True)
