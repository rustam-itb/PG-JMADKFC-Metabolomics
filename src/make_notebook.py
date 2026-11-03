import nbformat as nbf
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
n=nbf.v4.new_notebook()

def md(s):n.cells.append(nbf.v4.new_markdown_cell(s))
def code(s):n.cells.append(nbf.v4.new_code_cell(s))
md('''# PG-JMADKFC | Full-scale metabolomics benchmark

**Frozen candidate 1.0 · 6 datasets · 2 mechanisms · 3 missingness rates · 10 seeds · 8 methods = 2,880 fits**

**Proposed method:** original JMADKFC encoder–decoder + adaptive kernel fuzzy objective + *soft decoded-prototype completion* inspired by NPS + damped completion (`eta=0.25`) after 50 joint epochs + minimum prototype separation + reconstruction-guarded gradients. **PCA is not part of PG-JMADKFC** (PCA-FCM is a baseline). No possibilistic component, feature weighting, or CSC contrastive loss is enabled.

All six datasets were encountered earlier in the research program. Thus this is a **large-scale exploratory / robustness benchmark, not a previously untouched external holdout**. No ARI-based tuning or algorithm changes are permitted after freezing this protocol. Negative results must be retained.

### Usage

1. Extract the entire ZIP. Open this notebook **within the extracted project directory**; do not move the notebook alone.
2. Run cells from top to bottom in `RUN_MODE = 'SMOKE'` (quick technical check).
3. Change to `RUN_MODE = 'FULL'` and set `MAX_NEW_FITS = None` to attempt the entire frozen 2,880-fit schedule, **or** e.g. `MAX_NEW_FITS = 40` to process the next 40 new fits per notebook execution.
4. Restarting the same cell or kernel **resumes from JSON checkpoints**, checks protocol fingerprints, and skips only matching successful records.
5. Never delete `FROZEN_PROTOCOL.json` or edit the code after results exist. A method change requires an explicitly versioned protocol and a new results folder.

Subprocess isolation limits accumulated PyTorch memory. The full schedule is compute-intensive: run in a persistent computing environment and allocate substantial processing time. Notebook execution does **not** create a background task.''')
md('## 1. Locate project and check environment')
code('''from pathlib import Path
import os, sys, json, platform
ROOT = Path.cwd().resolve()
# Also supports a notebook launched from the parent folder.
if not (ROOT/'FROZEN_PROTOCOL.json').exists() and (ROOT/'PG_JMADKFC_Fullscale'/'FROZEN_PROTOCOL.json').exists():
    ROOT=ROOT/'PG_JMADKFC_Fullscale'
assert (ROOT/'FROZEN_PROTOCOL.json').is_file(), 'Open notebook from the extracted PG_JMADKFC_Fullscale directory'
sys.path.insert(0,str(ROOT/'src'))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(k,'1')
import numpy as np, torch, sklearn
print('Root:',ROOT)
print('Python:',platform.python_version(),'Torch:',torch.__version__,'NumPy:',np.__version__,'sklearn:',sklearn.__version__)
print('Platform:',platform.platform())''')
md('## 2. Verify frozen code, datasets, and experimental count')
code('''from freeze_protocol import check, CFG
from itertools import product
LOCK=check()
print('Protocol:',CFG['version'])
print('Fingerprint:',LOCK['fingerprint'])
print('Methods:',CFG['methods'])
print('Datasets:',CFG['data'])
PLAN=len(CFG['data'])*len(CFG['mechanisms'])*len(CFG['rates'])*len(CFG['seeds'])*len(CFG['methods'])
assert PLAN==2880
print('Planned fits:',PLAN)
print('Default deep training:',CFG['pretrain_epochs'],'+',CFG['joint_epochs'])''')
md('## 3. Verify eligibility and preprocessed data')
code('''import pandas as pd
eligibility=json.loads((ROOT/'data'/'eligibility_audit.json').read_text())
display(pd.DataFrame([dict(dataset=k,n=v['n_used'],d=v['d'],raw=v['n_raw'],
                           exclusions=str(v['excluded']),source_SHA=v['source_sha256'][:14])
                      for k,v in eligibility.items()]))
assert all((ROOT/'data'/(k+'_prepared.npz')).exists() for k in CFG['data'])''')
md('## 4. Cheap, label-free input checks and mask reproducibility')
code('''from run_one import make_setting
name='MTBLS161';rate=.20;mechanism='MAR';seed=CFG['seeds'][0]
y,M,S,source_sha=make_setting(name,mechanism,rate,seed)
_,M2,S2,_=make_setting(name,mechanism,rate,seed)
assert np.array_equal(M,M2)
assert np.array_equal(S['X_obs'],S2['X_obs'],equal_nan=True)
assert np.isnan(S['X_obs'][~M]).all()
assert np.isfinite(S['X_obs'][M]).all()
assert abs((~M).mean()-rate)<=1/M.size+1e-12
# Sanitization invariant of PG model: masked unobserved input values do not matter.
from pg_jmadkfc import PGJMADKFC
from run_one import make_model_kwargs
base_kw=make_model_kwargs(S['X_obs'].shape[1],seed,2,54)
ma=PGJMADKFC(**base_kw)
mb=PGJMADKFC(**base_kw)
Xsafe=S['X_obs'].copy();Xjunk=Xsafe.copy();Xjunk[~M]=1e5
x1,m1,t1=ma._prepare_inputs(Xsafe,M)
x2,m2,t2=mb._prepare_inputs(Xjunk,M)
assert np.array_equal(x1,x2) and np.array_equal(m1,m2) and np.array_equal(t1,t2)
print('PASS: shared masks, observed-only normalization, and masked-input sanitization')''')
md('## 5. Execution controls — change only these workflow controls')
code('''# SMOKE runs 8 short fits (2 pretraining, 54 joint epochs) and exercises 3 PG completion updates.
# FULL runs the frozen 100 pretraining + 200 joint epochs and exactly 2,880 scheduled fits.
RUN_MODE = 'SMOKE'   # Change to 'FULL' for large-scale experiments
MAX_NEW_FITS = 8     # Set None for all remaining; 40 or 80 for repeatable batches
FIT_TIMEOUT_SECONDS = 1800  # Change for machine availability only; does not change algorithm
assert RUN_MODE in ('SMOKE','FULL')
print('Mode',RUN_MODE,'Next new fits',MAX_NEW_FITS)''')
md('## 6. Execute/resume experiment (per-fit atomic JSON)')
code('''from run_all import run_all
SMOKE=(RUN_MODE=='SMOKE')
if SMOKE:
    outcome=run_all(max_fits=MAX_NEW_FITS,smoke=True,pretrain=2,joint=54,
                    timeout=FIT_TIMEOUT_SECONDS)
else:
    outcome=run_all(max_fits=MAX_NEW_FITS,smoke=False,
                    timeout=FIT_TIMEOUT_SECONDS)
print('Execution outcome:',outcome)
assert outcome['failed']==0, 'Inspect .error.txt log in the results directory' ''')
md('## 7. Analyze all completed fits; no filling missing experiments')
code('''from summarize import summarize
status=summarize(smoke=SMOKE)
folder=ROOT/('smoke_results' if SMOKE else 'results')
print('Fit records complete:',status['completed'],'/',status['expected_full'])
print('Missing full-schedule records:',status['missing_full'])
if status['completed']:
    df=pd.read_csv(folder/'all_results.csv')
    display(df[['dataset','mechanism','rate','seed','method','ARI','NMI','ACC','fuzzy_entropy']].head(12))''')
md('## 8. Aggregate descriptive results (only when enough paired repeats exist)')
code('''if status['completed']:
    raw=pd.read_csv(folder/'all_results.csv')
    table=raw.groupby('method',as_index=False).agg(n=('ARI','size'),mean_ARI=('ARI','mean'),
                     sd_ARI=('ARI','std'),mean_NMI=('NMI','mean'),mean_ACC=('ACC','mean'))
    display(table.sort_values('mean_ARI',ascending=False))
    paired_path=folder/'paired_PG_minus_baselines.csv'
    if paired_path.exists():
        contrast=pd.read_csv(paired_path)
        if len(contrast):
            display(contrast.groupby('baseline',as_index=False).agg(n=('delta_ARI','size'),
                              mean_paired_delta=('delta_ARI','mean'),
                              sd_delta=('delta_ARI','std')).sort_values('mean_paired_delta',ascending=False))
else:
    print('No completed fits')''')
md('''## 9. Statistical interpretation and model selection guardrails

- **Never choose PG algorithm weights, architecture or damping by inspecting the 2,880 ARI values** and then reuse the same outcomes as independent confirmation.
- `mean±SD` above summarizes setting×seed runs, **not independent populations**. For inference, first aggregate seed-level ARI differences within each dataset (and predeclare how mechanism/rate dimensions are handled), then use dataset-level confidence intervals/tests with limitations from only six historically used datasets.
- Report all rates and mechanisms, all failures, negative paired differences, prototype collapse/entropy, runtimes, and missing completions. Do not claim NPS-WKFCM numerical reproduction or guaranteed global convergence.
- Any refinements after reviewing results must be labeled a **new version**, validated on genuinely new datasets. This is especially important because the same six sources contributed to algorithm development decisions.
- The Full-JMADKFC comparator uses the revised, observed-only baseline code (including the corrected prototype update), not the original leakage-prone submission preprocessing.''')
md('## 10. Package accumulated results for later review')
code('''import zipfile
EXPORT=ROOT/('PG_results_SMOKE.zip' if SMOKE else 'PG_results_FULL_partial_or_complete.zip')
with zipfile.ZipFile(EXPORT,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(folder.glob('*.json')):
        z.write(p,arcname='results/'+p.name)
    for filename in ['all_results.csv','summary_by_setting.csv','paired_PG_minus_baselines.csv','analysis_status.json']:
        p=folder/filename
        if p.is_file():z.write(p,arcname='results/'+filename)
    z.write(ROOT/'FROZEN_PROTOCOL.json',arcname='FROZEN_PROTOCOL.json')
print('Results archive:',EXPORT)
print('No automatic uploads or asynchronous jobs are initiated.')''')
n.metadata= {'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python','version':'3'}}
p=ROOT/'PG_JMADKFC_Fullscale_Experiment.ipynb'
nbf.write(n,p)
print('created',p,'cells',len(n.cells))
