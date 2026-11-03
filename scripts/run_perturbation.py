#!/usr/bin/env python3
"""Post-fit metabolite sensitivity audit of PG-JMADKFC.
One observed feature at a time is permuted among its observed samples; mask, fitted
model, missing entries and all remaining features are unchanged. This is an
inference-time perturbation diagnostic, NOT biological causal validation.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import json,sys,time,random,hashlib,traceback, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_ablation import make_inputs,SEEDS
from run_one import make_model_kwargs
from pg_jmadkfc import PGJMADKFC
from JMADKFC_diagnostic_worker import set_torch_seed
import torch
OUT=ROOT/'perturbation_runs'

def predict_soft(model,Xt,reference_U):
    z=model._encode_numpy(torch.as_tensor(Xt,dtype=torch.float32,device=model.device))
    si=model._sample_bandwidths_np(z)
    sk=model._prototype_bandwidths_np(reference_U,si)
    D,_=model._adaptive_kernel_np(z,model.centers_,si,sk)
    return model._update_membership_np(D)

def one(job):
    dataset,mech,seed=job
    label=f'{dataset}__{mech}__20__s{seed}'
    p=OUT/(label+'.json')
    if p.exists():
        try:
            if json.loads(p.read_text()).get('status')=='success':return True,label
        except Exception:pass
    try:
        torch.set_num_threads(1)
        _,features,sha,M,S=make_inputs(dataset,mech,.2,seed)
        random.seed(seed);np.random.seed(seed);set_torch_seed(seed)
        model=PGJMADKFC(mask_scale=.25,grad_cap=.5,min_sep_fraction=.25,variance_floor=.5,variance_weight=.1,
                        burnin_epochs=50,completion_eta=.25,completion_clip=5.,
                        **make_model_kwargs(len(features),seed,100,200))
        model.fit(S['X_obs'],M)
        Xm=model.completion_.copy()
        Mt=.25*(M.astype(float)-M.mean(axis=0))
        Xt=np.hstack([Xm,Mt]).astype('float32')
        Uref=model.membership_.copy()
        U0=predict_soft(model,Xt,Uref)
        P=model._decode_numpy(model.centers_)
        wa=Uref**model.alpha
        profile=(wa.T@np.where(M,S['X_obs'],0.))/(wa.T@M.astype(float)+1e-10)
        mu=np.sum(np.where(M,S['X_obs'],0.),axis=0)/M.sum(axis=0)
        sd=np.sqrt(np.sum(np.where(M,(S['X_obs']-mu[None,:])**2,0.),axis=0)/M.sum(axis=0))
        delta=(profile-mu[None,:])/np.maximum(sd[None,:],1e-8)
        importance=np.max(np.abs(delta),axis=0)
        rng=np.random.default_rng(seed+991)
        rec=[]
        for j,feat in enumerate(features):
            xp=Xt.copy()
            obs=np.flatnonzero(M[:,j])
            if len(obs)>1:
                # Randomized intervention only in the observed values of metabolite j.
                xp[obs,j]=xp[rng.permutation(obs),j]
            U1=predict_soft(model,xp,Uref)
            sensitivity=float(np.mean(np.abs(U1-U0)))
            flip=float(np.mean(np.argmax(U1,axis=1)!=np.argmax(U0,axis=1)))
            rec.append({'feature':str(feat),'signed_cluster0':float(delta[0,j]),'signed_cluster1':float(delta[1,j]),
                        'importance_abs_max':float(importance[j]),'membership_sensitivity':sensitivity,
                        'cluster_flip_rate':flip})
        o={'status':'success','dataset':dataset,'mechanism':mech,'rate':.2,'seed':seed,'source_sha256':sha,
           'mask_sha256':hashlib.sha256(M.astype(np.uint8).tobytes()).hexdigest(),
           'proto_profile_pearson':[float(np.corrcoef(P[k],profile[k])[0,1]) for k in range(2)],
           'per_feature':rec}
        p.write_text(json.dumps(o,indent=2,allow_nan=False));return True,label
    except Exception as e:
        (OUT/(label+'.error.json')).write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2));return False,f'{label}:{e}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--seeds',type=int,default=5)
    a=p.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    jobs=[(d,m,s) for d in ['MTBLS161','MTBLS404','human_cachexia'] for m in ['MAR','MNAR'] for s in SEEDS[:a.seeds]]
    print('START perturbation',len(jobs),'models',flush=True)
    st=time.time();ok=bad=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fut=[pool.submit(one,j) for j in jobs]
        for f in as_completed(fut):
            yes,msg=f.result();ok+=int(yes);bad+=int(not yes)
            if not yes:print('ERROR',msg,flush=True)
            if (ok+bad)%5==0:print('PROGRESS',ok+bad,len(jobs),'elapsed',round(time.time()-st,1),flush=True)
    print('FINISHED',ok,bad,flush=True)
if __name__=='__main__':main()
