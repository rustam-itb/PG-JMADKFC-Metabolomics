"""Create or verify immutable code/config/data SHA-256 lock."""
import json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TRACK=['pg_jmadkfc.py','JMADKFC_diagnostic_worker.py','rgp_jmadkfc.py',
       'nps_factorial.py','source_reimplementations.py','run_one.py',
       'prepare_datasets.py','freeze_protocol.py','run_all.py','summarize.py']
SEEDS=[1061,2081,3109,4127,5147,6163,7187,8209,9221,10243]
METHODS=['PG-JMADKFC','Full-JMADKFC','RGP-JMADKFC','RGP-NPS',
         'DFCM','VKFCM-K-LP-PDS','FCM-KNN','PCA-FCM']
DATASETS=['MTBLS161','MTBLS404','MTBLS92','ST000369','ST000496','human_cachexia']
CFG={'version':'PGJMADKFC-1.0.0','data':DATASETS,'methods':METHODS,
     'mechanisms':['MAR','MNAR'],'rates':[.1,.2,.3],'seeds':SEEDS,
     'pretrain_epochs':100,'joint_epochs':200,'lambda_cluster':1.0,
     'burnin_epochs':50,'completion_eta':.25,'completion_clip':5.,
     'mask_scale':.25,'grad_cap':.5,'min_sep_fraction':.25,
     'gamma_missingness':3.,'classes':2,'missing_rate_semantics':'exact masked fraction',
     'note':'Exploratory benchmarking on historically accessed datasets, not pristine holdout.'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inventory():
    return {'code':{x:sha(ROOT/'src'/x) for x in TRACK},
            'prepared':{x:sha(ROOT/'data'/(x+'_prepared.npz')) for x in DATASETS}}
def check(freeze_if_missing=False):
    p=ROOT/'FROZEN_PROTOCOL.json';now={'config':CFG,'sha256':inventory()}
    now['fingerprint']=hashlib.sha256(json.dumps(now,sort_keys=True).encode()).hexdigest()
    if p.is_file():
        locked=json.loads(p.read_text())
        if locked!=now:
            raise RuntimeError('FROZEN PROTOCOL MISMATCH. Restore original files or explicitly version new experiment.')
    elif freeze_if_missing:
        p.write_text(json.dumps(now,indent=2))
    else:
        raise FileNotFoundError('Run freeze_protocol.py to lock first')
    return now
if __name__=='__main__':
    v=check(freeze_if_missing=True)
    print('LOCKED',v['fingerprint'])
