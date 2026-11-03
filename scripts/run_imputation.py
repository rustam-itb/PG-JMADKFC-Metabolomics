#!/usr/bin/env python3
"""Independent masked-entry recovery comparison. All transforms fit observed entries only.
Predictions are compared to held-out positions solely after completion.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import argparse,sys,json,time,traceback,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from sklearn.impute import KNNImputer
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_ablation import make_inputs,SEEDS
from JMADKFC_diagnostic_worker import mean_impute,median_impute,mice_impute
OUT=ROOT/'imputation_runs';METHODS=['Mean','Median','KNN','MICE','IterativeSVD']

def iter_svd_impute(X,rank=5,n_iter=15):
    X=np.asarray(X,float);M=np.isfinite(X)
    mu=np.nanmean(X,axis=0)
    A=np.where(M,X,mu[None,:])
    for _ in range(n_iter):
        # Low-rank matrix completion, fixed rank, no label usage.
        U,S,V=np.linalg.svd(A,full_matrices=False)
        k=min(rank,len(S));L=(U[:,:k]*S[:k])@V[:k]
        A[~M]=L[~M]
        A[M]=X[M]
    return A

def run(job):
    dataset,mech,rate,seed=job
    label=f'{dataset}__{mech}__{int(round(rate*100))}__s{seed}'
    path=OUT/(label+'.json')
    if path.exists():
        try:
            ob=json.loads(path.read_text());
            if ob.get('status')=='success':return True,'cached'
        except Exception:pass
    try:
        _,_,sha,M,S=make_inputs(dataset,mech,rate,seed)
        X=S['X_obs'];truth=S['X_truth_scaled'];rows=[]
        for method in METHODS:
            start=time.perf_counter()
            if method=='Mean':P=mean_impute(X)
            elif method=='Median':P=median_impute(X)
            elif method=='KNN':P=KNNImputer(n_neighbors=5,weights='uniform').fit_transform(X)
            elif method=='MICE':
                with threadpool_limits(limits=1):
                    P=mice_impute(X,random_state=seed,max_iter=10)
            elif method=='IterativeSVD':P=iter_svd_impute(X)
            err=P[~M]-truth[~M]
            rows.append(dict(method=method,RMSE=float(np.sqrt(np.mean(err**2))),MAE=float(np.mean(np.abs(err))),
                             runtime_seconds=float(time.perf_counter()-start)))
        o=dict(status='success',dataset=dataset,mechanism=mech,rate=rate,seed=seed,
            source_sha256=sha,mask_sha256=hashlib.sha256(M.astype(np.uint8).tobytes()).hexdigest(),
            n_missing=int((~M).sum()),rows=rows)
        path.write_text(json.dumps(o,allow_nan=False,indent=2));return True,'fresh'
    except Exception as ex:
        (OUT/(label+'.error.json')).write_text(json.dumps({'error':repr(ex),'traceback':traceback.format_exc()},indent=2));return False,f'{label}: {ex}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--seeds',type=int,default=10)
    a=p.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    jobs=[(d,m,r,s) for d in ['MTBLS161','MTBLS404','human_cachexia'] for m in ['MAR','MNAR'] for r in [.1,.2,.3] for s in SEEDS[:a.seeds]]
    print('START imputation',len(jobs),'cases',len(jobs)*len(METHODS),'fits',flush=True)
    started=time.time();ok=0;failed=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fut=[pool.submit(run,j) for j in jobs]
        for f in as_completed(fut):
            success,msg=f.result();ok+=int(success);failed+=int(not success)
            if not success:print('FAILED',msg,flush=True)
            if (ok+failed)%15==0: print('PROGRESS',ok+failed,'/',len(jobs),'failed',failed,'elapsed',round(time.time()-started,1),flush=True)
    print('FINISHED',ok,failed,'elapsed',round(time.time()-started,1),flush=True)
if __name__=='__main__':main()
