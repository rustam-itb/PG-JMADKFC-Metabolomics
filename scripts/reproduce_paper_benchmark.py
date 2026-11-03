#!/usr/bin/env python3
"""Run the paper's three-dataset / 12-method benchmark.

Uses the unchanged source implementation; intentionally does NOT call the
legacy six-dataset lock in src/freeze_protocol.py. Labels are only used for
post-training evaluation in run_one.run_fit.
"""
import argparse,json,sys,hashlib,time
from pathlib import Path
from itertools import product
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_one import run_fit,CFG

DATASETS=['MTBLS161','MTBLS404','human_cachexia']
METHODS=['PG-JMADKFC','DFCM','VKFCM-K-LP-PDS','FCM-KNN','FCM-Median','DEC','KFCM','FCM-MICE','FCM-PDS','FCM-OCS','FCM-Mean','VKFCM-K-LP-OCS']
SEEDS=[1061,2081,3109,4127,5147,6163,7187,8209,9221,10243]
# Extend only input validation; no changes are made to original algorithm definitions.
CFG['methods']=list(dict.fromkeys([*CFG['methods'],*METHODS]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=DATASETS,default='MTBLS161')
    ap.add_argument('--mechanism',choices=['MAR','MNAR'],default='MAR')
    ap.add_argument('--rate',type=float,choices=[.1,.2,.3],default=.1)
    ap.add_argument('--seed',type=int,choices=SEEDS,default=1061)
    ap.add_argument('--method',choices=METHODS,default='PG-JMADKFC')
    ap.add_argument('--all',action='store_true',help='Run 2160 jobs, rather than the selected single fit')
    ap.add_argument('--max-jobs',type=int,default=2160)
    ap.add_argument('--smoke',action='store_true',help='Reduce training to 1+2 epochs; never for scientific results')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results'/'reproduced')
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    if args.max_jobs<1:ap.error('--max-jobs must be at least 1')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for ds in (DATASETS if args.all else [args.dataset]):
        if not (ROOT/'data'/f'{ds}_prepared.npz').is_file():
            ap.error(f'Missing data/{ds}_prepared.npz; see data/README.md')
    jobs=(product(DATASETS,['MAR','MNAR'],[.1,.2,.3],SEEDS,METHODS) if args.all
          else [(args.dataset,args.mechanism,args.rate,args.seed,args.method)])
    done=skipped=0
    for ds,mech,rate,seed,method in jobs:
        if done+skipped>=args.max_jobs:break
        fn=f'{ds}__{mech}__{round(100*rate):02d}__s{seed}__{method}.json'
        output=args.output_dir/fn
        if output.exists() and not args.overwrite:
            if json.loads(output.read_text()).get('status')=='success':
                skipped+=1;continue
        pre,joint=(1,2) if args.smoke else (100,200)
        record=run_fit(ds,mech,rate,seed,method,pretrain=pre,joint=joint,smoke=args.smoke)
        record['execution_kind']='smoke_test_not_for_manuscript' if args.smoke else 'full_paper_protocol'
        output.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
        done+=1
        print(f'[{done}] {ds} {mech} {rate} seed={seed} {method} ARI={record["ARI"]:.6f}',flush=True)
    print(f'Finished new={done}, skipped={skipped}, destination={args.output_dir}')

if __name__=='__main__':main()
