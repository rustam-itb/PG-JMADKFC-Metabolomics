#!/usr/bin/env python3
"""Prepare the three *exact study-prepared* source tables using standard openpyxl.

No imputation or missingness is performed at this step. Reference labels are
preserved for post-fit external scoring, never for model fitting.
"""
import argparse, csv, hashlib, json, re
from pathlib import Path
import numpy as np
from openpyxl import load_workbook

REGION={'MTBLS161':(60,34),'MTBLS404':(185,124)}  # read source table until original fixed endpoints
DATSETS=['MTBLS161','MTBLS404','human_cachexia']

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_dataset(name, source):
    if name == 'human_cachexia':
        with open(source,newline='',encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        features=[f for f in rows[0] if f not in ('Patient ID','Muscle loss')]
        samples=[(r['Muscle loss'],[r[f] for f in features],i+2) for i,r in enumerate(rows)]
        allowed={'cachexic':0,'control':1}
    else:
        wb=load_workbook(source,data_only=True,read_only=True)
        if 'Data' not in wb.sheetnames:
            raise ValueError(f'{name}: missing Data worksheet')
        sheet=wb['Data']
        maxrow=60 if name=='MTBLS161' else 185
        values=list(sheet.iter_rows(min_row=1,max_row=maxrow,values_only=True))
        wb.close()
        header=list(values[0])
        features=sorted([v for v in header if isinstance(v,str) and re.fullmatch(r'M\d+',v)],key=lambda s:int(s[1:]))
        indices=[header.index(k) for k in features];cl=header.index('Class')
        samples=[(row[cl],[row[j] for j in indices],i+2) for i,row in enumerate(values[1:])]
        allowed={'0':0,'1':1,'0.0':0,'1.0':1}
    xs=[];ys=[];kept=[];bad=unlabeled=0
    for label,values,rowid in samples:
        lab=str(label).strip()
        if lab not in allowed:
            unlabeled+=1;continue
        try: vec=np.array([float(v) for v in values],dtype=np.float64)
        except (ValueError,TypeError):bad+=1;continue
        if not np.isfinite(vec).all() or (vec<0).any():
            bad+=1;continue
        xs.append(vec);ys.append(allowed[lab]);kept.append(rowid)
    X=np.asarray(xs,dtype=np.float64);y=np.asarray(ys,dtype=np.int64)
    if X.ndim!=2 or X.shape[1]!=len(features) or set(y)!={0,1}:
        raise ValueError(f'Unexpected prepared matrix for {name}: {X.shape}')
    return X,y,np.asarray(features),np.asarray(kept),{'source_file':source.name,'source_sha256':sha256(source),'n_used':len(y),'n_features':len(features),'excluded_unlabeled':unlabeled,'excluded_invalid_features':bad}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-dir',type=Path,default=Path('data/sources'))
    ap.add_argument('--output-dir',type=Path,default=Path('data'))
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    audit={}
    for name in DATSETS:
        fn=name+('.csv' if name=='human_cachexia' else '.xlsx')
        source=args.source_dir/fn
        if not source.is_file():raise FileNotFoundError(f'{source} (see data/README.md)')
        X,y,features,kept,meta=read_dataset(name,source)
        target=args.output_dir/f'{name}_prepared.npz'
        np.savez_compressed(target,X_log=np.log1p(X),y=y,features=features,source_sha256=np.array(meta['source_sha256']),source_rows=kept)
        audit[name]={**meta,'prepared_path':str(target),'prepared_sha256':sha256(target)}
        print(f'{name}: {len(y)} x {len(features)} -> {target}')
    (args.output_dir/'preparation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Done. Source SHA-256 values in preparation_audit.json')

if __name__=='__main__':main()
