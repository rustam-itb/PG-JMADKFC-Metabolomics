#!/usr/bin/env python3
"""Exploratory phenotype association of named features, labels used after frozen PG fit only.
Tests phenotype differences on the original complete reference without feeding labels to clustering.
This is within-cohort statistical validation, NOT independent biochemical pathway verification.
"""
from pathlib import Path
import numpy as np,pandas as pd,json,glob
from scipy.stats import mannwhitneyu, hypergeom
from statsmodels.stats.multitest import multipletests
BASE=Path(__file__).resolve().parents[1]/'data'
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'phenotype_association';OUT.mkdir(exist_ok=True)
sel=[]
for p in glob.glob(str(ROOT/'ablation_runs'/'*__g3__PG-control.json')):
    r=json.load(open(p))
    if r['mechanism'] not in ('MAR','MNAR'):continue
    key=(r['dataset'],r['mechanism'],r['rate'],r['seed'])
    feats=set()
    for t in r['metabolites']:
        feats.update(x['feature'] for x in t['top10'])
    for f in feats:sel.append(dict(dataset=key[0],mechanism=key[1],rate=key[2],seed=key[3],feature=f))
S=pd.DataFrame(sel)
frame=[];enrich=[]
for ds in ['MTBLS161','MTBLS404','human_cachexia']:
    with np.load(BASE/(ds+'_prepared.npz'),allow_pickle=False) as z:
        X=z['X_log'].astype(float);Y=z['y'].astype(int);F=z['features'].astype(str).tolist()
    results=[]
    for j,f in enumerate(F):
        x=X[:,j];v0=x[Y==0];v1=x[Y==1]
        p=float(mannwhitneyu(v0,v1,alternative='two-sided',method='asymptotic').pvalue)
        mean_diff=float(v1.mean()-v0.mean());pooled=np.sqrt((v0.var(ddof=1)+v1.var(ddof=1))/2)
        cohen=float(mean_diff/pooled) if pooled>1e-12 else 0.
        results.append(dict(dataset=ds,feature=f,p=p,mean_log_difference_class1_minus0=mean_diff,cohen_d=cohen))
    q=multipletests([r['p'] for r in results],method='fdr_bh')[1]
    for r,v in zip(results,q):r['q_BH']=float(v)
    d=pd.DataFrame(results)
    counts=S[S.dataset==ds].groupby('feature').size().rename('PG_top10_selected_in_scenario_seed').reset_index()
    d=d.merge(counts,how='left',on='feature');d['PG_top10_selected_in_scenario_seed']=d['PG_top10_selected_in_scenario_seed'].fillna(0).astype(int)
    d['PG_selection_frequency']=d['PG_top10_selected_in_scenario_seed']/60.
    d['FDR_significant']=d['q_BH']<.05
    d=d.sort_values(['PG_top10_selected_in_scenario_seed','feature'],ascending=[False,True])
    frame.extend(d.to_dict('records'))
    top10=d.head(10)
    n_sign=int(d.FDR_significant.sum());K=int(top10.FDR_significant.sum())
    enrichment_p=float(hypergeom.sf(K-1,len(F),n_sign,10)) if n_sign else 1.
    enrich.append(dict(dataset=ds,n_features=len(F),n_BH_sig=n_sign,top10_selected_features_with_BH_sig=K,
                       hypergeometric_p_descriptive=enrichment_p,top10_named_features='; '.join(top10.feature.tolist())))
pd.DataFrame(frame).to_csv(OUT/'phenotype_association_all_features.csv',index=False)
pd.DataFrame(enrich).to_csv(OUT/'phenotype_association_enrichment.csv',index=False)
print(pd.DataFrame(enrich).to_string(index=False))
print('\nHuman cachexia representative:');print(pd.DataFrame(frame).query("dataset == 'human_cachexia'").head(12).to_string(index=False))
