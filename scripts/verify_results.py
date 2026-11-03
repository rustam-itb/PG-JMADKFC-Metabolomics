#!/usr/bin/env python3
"""Verify the reported computational outputs without accessing participant-level data."""
from pathlib import Path
import csv,json,hashlib,sys
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/'results'/'benchmark'/'all_2160_fits.csv'
MAIN_METHODS={'PG-JMADKFC','DFCM','VKFCM-K-LP-PDS','FCM-KNN','FCM-Median','DEC','KFCM','FCM-MICE','FCM-PDS','FCM-OCS','FCM-Mean','VKFCM-K-LP-OCS'}
errors=[]
def require(ok,msg):
    if not ok:errors.append(msg)
def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

records=read(MAIN)
require(len(records)==2160,f'Expected 2160 benchmark fits, found {len(records)}')
methods={r['method'] for r in records}
require(methods==MAIN_METHODS,f'Unexpected method set: {methods ^ MAIN_METHODS}')
require(len({(r['dataset'],r['mechanism'],r['rate'],r['seed'],r['method']) for r in records})==2160,'Duplicated benchmark fit identity')
by_condition=defaultdict(list)
for r in records:by_condition[(r['dataset'],r['mechanism'],r['rate'],r['seed'])].append(r)
require(len(by_condition)==180,'Expected 180 matched conditions')
require(all(len(x)==12 for x in by_condition.values()),'Missing methods for a condition')
require(all(len({v.get('mask_sha256') for v in x})==1 for x in by_condition.values()),'Mask checksum mismatch')
pg=defaultdict(list)
for r in records:
    if r['method']=='PG-JMADKFC':pg[r['dataset']].append(float(r['ARI']))
for ds,expected in [('MTBLS161',.1511),('MTBLS404',.0231),('human_cachexia',.1595)]:
    actual=sum(pg[ds])/len(pg[ds]) if pg[ds] else float('nan')
    require(len(pg[ds])==60 and abs(actual-expected)<.0001,f'PG main mean ARI mismatch: {ds} {actual}')

# Fit-level ablation files, including 540 non-primary missingness runs and 2160 primary variant fits.
ab_files=list((ROOT/'ablation_runs').glob('*.json'))
ab_records=[json.loads(p.read_text()) for p in ab_files if not p.name.endswith('.error.json')]
require(all(r.get('status')=='success' for r in ab_records),'At least one ablation JSON is not successful')
main_ab=[r for r in ab_records if r.get('gamma')==3 and r.get('mechanism') in ['MAR','MNAR']]
require(len(main_ab)==2160,f'Expected 2160 main ablation fits, found {len(main_ab)}')
require(len([r for r in ab_records if r.get('variant')=='PG-control'])==630,'Expected 630 PG-control sensitivity fits')
imp_files=list((ROOT/'imputation_runs').glob('*.json'))
imp_records=[json.loads(p.read_text()) for p in imp_files if not p.name.endswith('.error.json')]
require(len(imp_records)==180,f'Expected 180 imputation scenarios, found {len(imp_records)}')
require(all(len(r.get('rows',[]))==5 for r in imp_records),'Expected 5 imputation comparators per mask')
pert_files=list((ROOT/'perturbation_runs').glob('*.json'))
require(len(pert_files)==30,f'Expected 30 perturbation runs, found {len(pert_files)}')
# Verify hashes of final-analysis CSVs against the archived additional experiment manifest.
manifest_file=ROOT/'final_analysis'/'EXPERIMENT_MANIFEST.json'
if manifest_file.is_file():
    manifest=json.loads(manifest_file.read_text())
    for name,digest in manifest.get('files',{}).items():
        p=ROOT/'final_analysis'/name
        require(p.is_file(),f'Missing manifest file: {name}')
        if p.is_file():require(hashlib.sha256(p.read_bytes()).hexdigest()==digest,f'SHA256 mismatch: {name}')
print('Benchmark fits:',len(records),'methods:',len(methods),'matched conditions:',len(by_condition))
print('PG ARI means:',{k:round(sum(v)/len(v),4) for k,v in pg.items()})
print('Main ablation fits:',len(main_ab),'PG control sensitivity fits:',sum(r.get('variant')=='PG-control' for r in ab_records))
print('Imputation scenario files:',len(imp_records),'perturbation files:',len(pert_files))
if errors:
    print('FAILED:',len(errors),'issues')
    for err in errors:print(' -',err)
    sys.exit(1)
print('PASSED: result-level numerical and integrity checks.')
