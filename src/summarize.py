"""Analyze only completed, fingerprint-matching records, never silently impute missing fits."""
import csv,json,math
from pathlib import Path
from collections import defaultdict
from freeze_protocol import CFG,check
from run_all import identifier
ROOT=Path(__file__).resolve().parents[1]

def save_csv(path,rows):
    if not rows:return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def summarize(smoke=False):
    lock=check()
    folder=ROOT/('smoke_results' if smoke else 'results')
    rows=[]; allkeys=set(); mismatches=[]
    for p in sorted(folder.glob('*.json')):
        r=json.loads(p.read_text());key=(r['dataset'],r['mechanism'],r['rate'],r['seed'],r['method'])
        if r.get('protocol_fingerprint')!=lock['fingerprint'] or r.get('status')!='success':
            mismatches.append(str(p));continue
        if key in allkeys:raise AssertionError(f'Duplicate fit: {key}')
        allkeys.add(key);rows.append(r)
    if mismatches:raise RuntimeError('Stale/incompatible results: '+str(mismatches[:10]))
    bysetting=defaultdict(dict)
    for r in rows:
        k=(r['dataset'],r['mechanism'],r['rate'],r['seed'])
        masks={x['mask_sha256'] for x in bysetting[k].values()}
        if masks and r['mask_sha256'] not in masks:
            raise AssertionError(f'Shared missingness mask violated for {k}')
        bysetting[k][r['method']]=r
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
    save_csv(folder/'all_results.csv',flat)
    agg=defaultdict(list)
    for r in rows:agg[(r['dataset'],r['mechanism'],r['rate'],r['method'])].append(r['ARI'])
    output=[]
    for k,arr in sorted(agg.items()):
        n=len(arr);mu=sum(arr)/n;sd=(sum((x-mu)**2 for x in arr)/(n-1))**.5 if n>1 else None
        output.append(dict(zip(['dataset','mechanism','rate','method'],k),n=n,mean_ARI=mu,sd_ARI=sd))
    save_csv(folder/'summary_by_setting.csv',output)
    paired=[]
    for k, models in sorted(bysetting.items()):
        if 'PG-JMADKFC' not in models:continue
        pg=models['PG-JMADKFC']['ARI']
        for baseline in CFG['methods']:
            if baseline=='PG-JMADKFC' or baseline not in models:continue
            paired.append(dict(dataset=k[0],mechanism=k[1],rate=k[2],seed=k[3],
                               baseline=baseline,delta_ARI=pg-models[baseline]['ARI']))
    save_csv(folder/'paired_PG_minus_baselines.csv',paired)
    expected=len(CFG['data'])*len(CFG['mechanisms'])*len(CFG['rates'])*len(CFG['seeds'])*len(CFG['methods'])
    status={'completed':len(rows),'expected_full':expected,'missing_full':expected-len(rows),
            'n_paired_contrasts':len(paired),'n_scenarios':len(bysetting),
            'by_method':{m:sum(r['method']==m for r in rows) for m in CFG['methods']}}
    (folder/'analysis_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))
    return status
if __name__=='__main__':summarize()
