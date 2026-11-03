"""Generate frozen NPZ datasets from uploaded original sources.

Reads Excel with artifact_tool. Rules are predeclared; study labels never enter
clustering fit. Complete nonnegative sample rows only, binary labeled classes.
Output X_log is log1p(raw intensities) for artificial missingness generation.
"""
from artifact_tool import Blob, SpreadsheetFile
from pathlib import Path
import csv, json, hashlib, re
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
REGION={'MTBLS161':'AH60','MTBLS404':'DT185','MTBLS92':'ET448',
        'ST000369':'GF182','ST000496':'BV101'}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def make_cache(name):
    path = DATA/(name + ('.csv' if name=='human_cachexia' else '.xlsx'))
    if not path.exists():raise FileNotFoundError(path)
    if name == 'human_cachexia':
        with path.open(newline='', encoding='utf-8-sig') as f:
            content=list(csv.DictReader(f))
        features = [k for k in content[0] if k not in ('Patient ID','Muscle loss')]
        samples=[{'Class':r['Muscle loss'], 'values':[r[x] for x in features], 'row':idx+2}
                 for idx,r in enumerate(content)]
        allowed={'cachexic':0,'control':1}
    else:
        wb=SpreadsheetFile.import_xlsx(Blob.load(str(path)))
        rows=wb.worksheets.get_item('Data').get_range('A1:'+REGION[name]).values
        header=rows[0]
        features=sorted([x for x in header if isinstance(x,str) and re.fullmatch(r'M\d+',x)],
                        key=lambda x:int(x[1:]))
        fids=[header.index(k) for k in features]
        cidx=header.index('Class')
        samples=[{'Class':r[cidx], 'values':[r[k] for k in fids], 'row':idx+2}
                 for idx,r in enumerate(rows[1:])]
        allowed={'0':0,'1':1,'0.0':0,'1.0':1}
    X=[]; Y=[]; kept=[]; exclusions={'unlabeled':0,'invalid_feature':0}
    for r in samples:
        label=str(r['Class']).strip()
        if label not in allowed:
            exclusions['unlabeled']+=1;continue
        try:
            vec=np.array([float(v) for v in r['values']],dtype=np.float64)
        except (ValueError,TypeError):
            exclusions['invalid_feature']+=1;continue
        if not np.isfinite(vec).all() or (vec<0).any():
            exclusions['invalid_feature']+=1;continue
        X.append(vec);Y.append(allowed[label]);kept.append(r['row'])
    X=np.asarray(X,dtype=np.float64);Y=np.asarray(Y,dtype=np.int64)
    if X.ndim!=2 or X.shape[1]!=len(features) or set(np.unique(Y))!={0,1}:
        raise ValueError(f'Invalid binary dataset {name}: {X.shape}')
    dest=DATA/(name+'_prepared.npz')
    np.savez_compressed(dest,X_log=np.log1p(X),y=Y,features=np.array(features),
                        source_sha256=np.array(sha(path)),source_rows=np.array(kept))
    return {'source_file':path.name,'source_sha256':sha(path),
            'prepared_sha256':sha(dest),'n_raw':len(samples), 'n_used':len(Y),
            'd':len(features),'classes':{str(k):int((Y==k).sum()) for k in (0,1)},
            'excluded':exclusions}

if __name__=='__main__':
    audit={k:make_cache(k) for k in [*REGION,'human_cachexia']}
    (DATA/'eligibility_audit.json').write_text(json.dumps(audit,indent=2))
    for k,v in audit.items(): print(k, v['n_used'], v['d'], v['excluded'])
