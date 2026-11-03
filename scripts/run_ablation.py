#!/usr/bin/env python3
"""Pre-specified PG-JMADKFC component ablations on frozen three-dataset benchmark.
No class labels are passed into training; reference labels are read only for post-fit metrics.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import sys,json,time,hashlib,random,argparse,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'src'))
import torch
from pg_jmadkfc import PGJMADKFC
from JMADKFC_diagnostic_worker import generate_missingness, observed_normalize, set_torch_seed, external_clustering_metrics
from run_one import make_model_kwargs
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ablation_runs'
VARIANTS=['PG-control','without-completion','without-mask-channel','shuffled-mask-channel','without-gradient-guard','without-separation','without-variance-safeguard']

class PGGlobalBandwidth(PGJMADKFC):
    """Replace local qNN bandwidths with within-epoch global median."""
    def _sample_bandwidths_np(self,Z):
        local=super()._sample_bandwidths_np(Z)
        return np.full_like(local,max(float(np.median(local)),self.eps))
    def _prototype_bandwidths_np(self,U,sigma_i):
        return np.full(self.n_clusters,float(sigma_i[0]),dtype=float)

class PGZeroTargetReconstruction(PGJMADKFC):
    """Ablation: penalize predictions at missing entries against fixed zero placeholders."""
    def _masked_reconstruction_loss(self,X_zero_tensor,mask_tensor,X_hat_tensor):
        return torch.mean((X_zero_tensor-X_hat_tensor)**2)

SEEDS=[1061,2081,3109,4127,5147,6163,7187,8209,9221,10243]

def make_inputs(dataset,mechanism,rate,seed,gamma=3.):
    with np.load(BASE/'data'/f'{dataset}_prepared.npz',allow_pickle=False) as src:
        Xlog=src['X_log'].astype(float);y=src['y'].astype(int)
        features=src['features'].astype(str).tolist()
        datahash=str(src['source_sha256'])
    gen_kwargs={} if mechanism=='MCAR' else {'gamma':float(gamma)}
    _,M=generate_missingness(Xlog,mechanism,rate,random_state=seed,**gen_kwargs)
    M=np.asarray(M,dtype=bool)
    S=observed_normalize(Xlog,M)
    return y,features,datahash,M,S

def fit_one(job):
    dataset,mechanism,rate,seed,variant,gamma=job
    name=f'{dataset}__{mechanism}__{int(round(rate*100))}__s{seed}__g{gamma:g}__{variant}'
    target=OUT/(name+'.json')
    if target.exists():
        try:
            d=json.loads(target.read_text())
            if d.get('status')=='success':return {'cached':True,'success':True,'name':name}
        except Exception:pass
    started=time.perf_counter()
    try:
        torch.set_num_threads(1)
        y,features,datahash,M,S=make_inputs(dataset,mechanism,rate,seed,gamma)
        random.seed(seed);np.random.seed(seed);set_torch_seed(seed)
        opts=make_model_kwargs(len(features),seed,100,200)
        kw=dict(mask_scale=.25,grad_cap=.5,min_sep_fraction=.25,variance_floor=.5,variance_weight=.1,
                burnin_epochs=50,completion_eta=.25,completion_clip=5.)
        mode='candidate'
        model_cls=PGJMADKFC
        if variant=='mask-scale-0.10':kw['mask_scale']=0.10
        elif variant=='mask-scale-0.50':kw['mask_scale']=0.50
        elif variant=='global-bandwidth-kernel':model_cls=PGGlobalBandwidth
        elif variant=='zero-target-reconstruction':model_cls=PGZeroTargetReconstruction
        elif variant=='without-clustering-gradient':opts['lambda_cluster']=0.
        elif variant=='without-completion':kw['burnin_epochs']=199
        elif variant=='without-mask-channel':mode='no_mask'
        elif variant=='shuffled-mask-channel':mode='shuffled_mask'
        elif variant=='without-gradient-guard':mode='no_guard'
        elif variant=='without-separation':mode='no_projection'
        elif variant=='without-variance-safeguard':mode='no_variance'
        elif variant not in ('PG-control','mask-scale-0.10','mask-scale-0.50','global-bandwidth-kernel','zero-target-reconstruction','without-clustering-gradient'):
            raise ValueError('unrecognized variant')
        model=model_cls(**kw,**opts)
        model.mode=mode
        pred=model.fit_predict(S['X_obs'],M)
        scores=external_clustering_metrics(y,pred)
        C=np.asarray(model.completion_)
        targetarr=S['X_truth_scaled']
        hidden=M==0
        comp_error=C[hidden]-targetarr[hidden]
        U=np.asarray(model.membership_,dtype=float)
        ent=float(np.mean(-np.sum(U*np.log(np.clip(U,1e-12,1)),axis=1))/np.log(2))
        result=dict(status='success',dataset=dataset,mechanism=mechanism,rate=rate,seed=seed,
                    variant=variant,gamma=gamma,mask_sha256=hashlib.sha256(M.astype(np.uint8).tobytes()).hexdigest(),
                    source_sha256=datahash,n_samples=len(y),n_features=len(features),
                    ARI=float(scores['ARI']),NMI=float(scores['NMI']),ACC=float(scores['ACC']),
                    fuzzy_entropy=ent,prototype_separation=float(model.separation_),
                    projection_count=int(model.projection_count_),dynamic_updates=int(model.dynamic_updates_),
                    missing_completion_rmse=float(np.sqrt(np.mean(comp_error**2))),
                    missing_completion_mae=float(np.mean(np.abs(comp_error))),
                    observed_reconstruction_mse=float(np.mean((model.X_reconstructed_[M]-S['X_obs'][M])**2)),
                    runtime_fit_seconds=float(time.perf_counter()-started))
        if variant=='PG-control':
            # All interpretation evidence is post-training and is NOT fed to optimization.
            P=model._decode_numpy(np.asarray(model.centers_,dtype=float))
            obs=S['X_obs']
            Ua=np.maximum(U,0)**model.alpha
            numerator=Ua.T@np.where(M,obs,0.)
            denominator=Ua.T@M.astype(float)
            profile=numerator/np.maximum(denominator,1e-10)
            mu=np.sum(np.where(M,obs,0.),axis=0)/np.maximum(M.sum(axis=0),1)
            sd=np.sqrt(np.sum(np.where(M,(obs-mu[None,:])**2,0.),axis=0)/np.maximum(M.sum(axis=0),1))
            contrast=(profile-mu[None,:])/np.maximum(sd[None,:],1e-10)
            corrs=[]
            for k in range(2):
                valid=np.isfinite(profile[k]) & np.isfinite(P[k])
                c=float(np.corrcoef(profile[k,valid],P[k,valid])[0,1]) if valid.sum()>2 else None
                corrs.append(c)
            result['proto_profile_pearson']=corrs
            result['metabolites']=[{'cluster':k,'top10':[{'feature':str(features[j]),'signed_contrast':float(contrast[k,j]),
                  'abs_importance':float(abs(contrast[k,j]))} for j in np.argsort(-np.abs(contrast[k]))[:10]]} for k in range(2)]
            result['feature_count_with_zero_observations']=int(np.sum(M.sum(axis=0)==0))
        target.write_text(json.dumps(result,allow_nan=False,indent=2),encoding='utf8')
        return {'cached':False,'success':True,'name':name,'seconds':result['runtime_fit_seconds']}
    except Exception as ex:
        err={'status':'failed','name':name,'error':repr(ex),'traceback':traceback.format_exc()}
        (OUT/(name+'.error.json')).write_text(json.dumps(err,indent=2))
        return {'cached':False,'success':False,'name':name,'error':repr(ex)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3)
    p.add_argument('--seeds',type=int,default=10);p.add_argument('--rates',nargs='+',type=float,default=[.1,.2,.3])
    p.add_argument('--datasets',nargs='+',default=['MTBLS161','MTBLS404','human_cachexia'])
    p.add_argument('--mechanisms',nargs='+',default=['MAR','MNAR'])
    p.add_argument('--variants',nargs='+',default=VARIANTS)
    p.add_argument('--gamma',type=float,default=3.)
    args=p.parse_args(); OUT.mkdir(exist_ok=True,parents=True)
    jobs=[(d,m,r,s,v,args.gamma) for d in args.datasets for m in args.mechanisms for r in args.rates for s in SEEDS[:args.seeds] for v in args.variants]
    print('START',len(jobs),'jobs',args.variants,flush=True)
    st=time.time(); ok=bad=cached=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs=[pool.submit(fit_one,j) for j in jobs]
        for f in as_completed(futs):
            o=f.result()
            ok+=int(o['success']);bad+=int(not o['success']);cached+=int(o.get('cached',False))
            if not o['success']:print('FAIL',o['name'],o.get('error'),flush=True)
            if (ok+bad)%30==0 or bad:print('PROGRESS',ok+bad,'/',len(jobs),'success',ok,'failed',bad,'cached',cached,'wall_s',round(time.time()-st,1),flush=True)
    print('FINISHED',len(jobs),'success',ok,'failed',bad,'elapsed',round(time.time()-st,2),flush=True)
if __name__=='__main__':main()
