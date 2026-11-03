"""Run frozen 2,880 fits, one isolated subprocess per method/setting/seed.
Idempotent/resumable with per-fit result JSON and per-fit failure logs.
"""
import argparse,json,sys,subprocess,time,hashlib,traceback
from pathlib import Path
from itertools import product
from freeze_protocol import CFG,check
ROOT=Path(__file__).resolve().parents[1]

def identifier(dataset,mechanism,rate,seed,method):
    return f'{dataset}__{mechanism}__{int(round(100*rate)):02d}__s{seed}__{method}'

def run_all(datasets=None,mechanisms=None,rates=None,seeds=None,methods=None,
            max_fits=None,smoke=False,pretrain=None,joint=None,timeout=1800):
    lock=check();fingerprint=lock['fingerprint']
    datasets=CFG['data'] if datasets is None else datasets
    mechanisms=CFG['mechanisms'] if mechanisms is None else mechanisms
    rates=CFG['rates'] if rates is None else rates
    seeds=CFG['seeds'] if seeds is None else seeds
    methods=CFG['methods'] if methods is None else methods
    pretrain=CFG['pretrain_epochs'] if pretrain is None else pretrain
    joint=CFG['joint_epochs'] if joint is None else joint
    out=ROOT/('smoke_results' if smoke else 'results');out.mkdir(parents=True,exist_ok=True)
    total=0;done=0;failed=0;skipped=0;attempted=0
    for dataset,mechanism,rate,seed,method in product(datasets,mechanisms,rates,seeds,methods):
        total+=1
        run_id=identifier(dataset,mechanism,rate,seed,method)
        result=out/(run_id+'.json');error=out/(run_id+'.error.txt')
        if result.exists():
            record=json.loads(result.read_text())
            if record.get('protocol_fingerprint')!=fingerprint or record.get('status')!='success':
                raise RuntimeError(f'Incompatible or corrupted checkpoint: {result}')
            skipped+=1;continue
        if max_fits is not None and attempted>=max_fits:break
        attempted+=1
        cmd=[sys.executable,str(ROOT/'src'/'run_one.py'),'--dataset',dataset,'--mechanism',mechanism,
             '--rate',str(rate),'--seed',str(seed),'--method',method,'--pretrain',str(pretrain),
             '--joint',str(joint),'--output',str(result)]
        if smoke:cmd.append('--smoke')
        start=time.perf_counter()
        try:
            cp=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              text=True,timeout=timeout,cwd=str(ROOT))
            if cp.returncode!=0:
                failed+=1
                error.write_text(' '.join(cmd)+'\n'+cp.stdout+'\n'+cp.stderr)
                print('FAILED',run_id,cp.stderr[-400:],flush=True)
            else:
                if not result.exists():raise RuntimeError('Child exited without atomic result')
                if error.exists():error.unlink()
                done+=1
                print(f'{total:4d} DONE {run_id} {time.perf_counter()-start:.1f}s',flush=True)
        except Exception:
            failed+=1
            error.write_text(traceback.format_exc())
            print('FAILED exception',run_id,flush=True)
    print(f'Finished traversed={total}, attempted_new={attempted}, success_new={done}, existing={skipped}, failed={failed}. Records: {out}')
    return {'traversed':total,'attempted':attempted,'new':done,'skipped':skipped,'failed':failed}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-fits',type=int,default=None)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--pretrain',type=int,default=None)
    ap.add_argument('--joint',type=int,default=None)
    ap.add_argument('--timeout',type=int,default=1800)
    a=ap.parse_args()
    run_all(max_fits=a.max_fits,smoke=a.smoke,pretrain=a.pretrain,
            joint=a.joint,timeout=a.timeout)
