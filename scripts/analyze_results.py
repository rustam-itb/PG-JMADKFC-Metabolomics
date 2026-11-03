#!/usr/bin/env python3
"""Summarize completed new PG-JMADKFC experiments; all numbers from emitted fit JSON."""
from pathlib import Path
from collections import Counter
import json,hashlib,math,itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,spearmanr
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'final_analysis';OUT.mkdir(exist_ok=True)
EXPECTED_MAIN=180*12 # 7 primary, 3 secondary, 2 mask-scale variants

def files_reader(folder):
    return [json.loads(p.read_text()) for p in sorted((ROOT/folder).glob('*.json')) if not p.name.endswith('.error.json') and json.loads(p.read_text()).get('status')=='success']

def fmt(x,dec=4):return f'{x:.{dec}f}' if np.isfinite(x) else '--'

def frame_write(df,name):
    df.to_csv(OUT/name,index=False,float_format='%.12g')

def tabulate(df,fields=None,maxrows=None):
    if fields:df=df[fields]
    if maxrows is not None:df=df.head(maxrows)
    return df.to_markdown(index=False,floatfmt='.4f')

def add_note(lines,title,content):
    lines.append('## '+title+'\n');lines.extend(content);lines.append('')

ab=files_reader('ablation_runs')
rows=[];profile_rows=[];features_rows=[]
for r in ab:
    row={k:v for k,v in r.items() if isinstance(v,(int,float,str)) and k not in ('source_sha256','mask_sha256')}
    row['source_sha256']=r['source_sha256'];row['mask_sha256']=r['mask_sha256'];rows.append(row)
    if r['variant']=='PG-control':
        for k,v in enumerate(r.get('proto_profile_pearson',[])):
            profile_rows.append(dict(dataset=r['dataset'],mechanism=r['mechanism'],rate=r['rate'],seed=r['seed'],gamma=r['gamma'],cluster=k,pearson=v))
        if r['gamma']==3 and r['mechanism']!='MCAR':
            for cluster_info in r.get('metabolites',[]):
                for rank,x in enumerate(cluster_info['top10'],start=1):
                    features_rows.append(dict(dataset=r['dataset'],mechanism=r['mechanism'],rate=r['rate'],seed=r['seed'],cluster=cluster_info['cluster'],rank=rank,feature=x['feature'],signed_contrast=x['signed_contrast'],abs_importance=x['abs_importance']))
df=pd.DataFrame(rows);df['rate']=df['rate'].round(6)
frame_write(df,'all_additional_pg_fit_metrics.csv')
main=df[(df.gamma==3)&(df.mechanism.isin(['MAR','MNAR']))].copy()
if len(main)!=EXPECTED_MAIN:
    raise RuntimeError(f'Expected {EXPECTED_MAIN} main variant fits, found {len(main)}. Missing/failure?')
key=['dataset','mechanism','rate','seed']
if main.groupby(key+['variant']).size().max()!=1:raise AssertionError('Duplicate frozen fit key')
if main.groupby(key)['mask_sha256'].nunique().max()!=1:raise AssertionError('Ablation mask mismatch')
if main.groupby(key)['source_sha256'].nunique().max()!=1:raise AssertionError('Ablation source mismatch')
reference=main[main.variant=='PG-control'].set_index(key)
variants=main.variant.unique().tolist()
if len(reference)!=180:raise RuntimeError('Invalid PG controls count')
# Check original archive exact regression
orig=pd.read_csv(str(ROOT/'benchmark_reference_2160_fits.csv'))
archive=orig[orig.method=='PG-JMADKFC'].set_index(key)
regression=reference.join(archive[['ARI','NMI','ACC','mask_sha256','posthoc_missing_completion_mse']],rsuffix='_archive')
if len(regression)!=180 or not (regression.mask_sha256==regression.mask_sha256_archive).all():raise AssertionError('Mask hash regression mismatch')
max_err={m:float(np.abs(regression[m]-regression[m+'_archive']).max()) for m in ['ARI','NMI','ACC']}
if max(max_err.values())>1e-10:raise AssertionError('Model replay mismatch')
max_completion_replay=float(np.max(np.abs(regression.missing_completion_rmse**2-regression.posthoc_missing_completion_mse)))
if max_completion_replay>1e-8:raise AssertionError('Completion metric replay mismatch')

# Paired ablation effects. Positive delta = PG control performs better.
paired=main.join(reference[['ARI','NMI','ACC','fuzzy_entropy','missing_completion_rmse','prototype_separation']],on=key,rsuffix='_control')
for metric in ['ARI','NMI','ACC']:
    paired[f'delta_{metric}_control_minus_variant']=paired[f'{metric}_control']-paired[metric]
paired=paired[paired.variant!='PG-control'].copy()
frame_write(paired,'paired_component_ablation_180_conditions.csv')
per_setting=paired.groupby(['variant','dataset','mechanism','rate']).agg(n=('ARI','size'),mean_delta_ARI=('delta_ARI_control_minus_variant','mean'),sd_delta_ARI=('delta_ARI_control_minus_variant','std')).reset_index()
frame_write(per_setting,'ablation_effect_per_18_scenarios.csv')
summary=[]
for variant in variants:
    f=main[main.variant==variant]
    p=paired[paired.variant==variant]
    sc=per_setting[per_setting.variant==variant]
    summary.append(dict(variant=variant,n_fits=len(f),mean_ARI=f.ARI.mean(),sd_ARI=f.ARI.std(ddof=1),
        mean_NMI=f.NMI.mean(),mean_ACC=f.ACC.mean(),
        mean_delta_ARI_control_minus_variant=(p.delta_ARI_control_minus_variant.mean() if len(p) else 0.),
        n_positive_seed_pairs=int((p.delta_ARI_control_minus_variant>0).sum()) if len(p) else None,
        n_positive_scenario_means=int((sc.mean_delta_ARI>0).sum()) if len(sc) else None,
        median_delta_ARI_control_minus_variant=(p.delta_ARI_control_minus_variant.median() if len(p) else 0.)))
summary=pd.DataFrame(summary).sort_values('mean_ARI',ascending=False);frame_write(summary,'ablation_overall_comparison.csv')
dataset_summary=main.groupby(['dataset','variant']).agg(mean_ARI=('ARI','mean'),sd_ARI=('ARI','std'),mean_ACC=('ACC','mean'),mean_completion_rmse=('missing_completion_rmse','mean'),mean_entropy=('fuzzy_entropy','mean')).reset_index()
dataset_summary['paired_delta_mean_ARI']=np.nan
for ix,r in dataset_summary.iterrows():
    pp=paired[(paired.dataset==r.dataset)&(paired.variant==r.variant)]
    if len(pp):dataset_summary.loc[ix,'paired_delta_mean_ARI']=pp.delta_ARI_control_minus_variant.mean()
frame_write(dataset_summary,'ablation_by_dataset.csv')
# Exploratory scenario-mean Wilcoxon: 18 conditions are NOT independent cohorts.
sc_means=main.groupby(['dataset','mechanism','rate','variant']).ARI.mean().unstack('variant')
stat_rows=[]
for variant in [x for x in sc_means.columns if x!='PG-control']:
    delta=sc_means['PG-control']-sc_means[variant]
    _,p=wilcoxon(delta,zero_method='pratt')
    stat_rows.append({'variant':variant,'n_scenario_means':len(delta),'mean_delta_PG_minus_variant':delta.mean(),
                     'scenarios_favor_PG':int((delta>0).sum()),'scenarios_favor_variant':int((delta<0).sum()),'wilcoxon_p_exploratory':p})
stat_df=pd.DataFrame(stat_rows)
stat_df['holm_adjusted_p_exploratory']=multipletests(stat_df.wilcoxon_p_exploratory,method='holm')[1]
frame_write(stat_df,'ablation_wilcoxon_scenario_means_exploratory.csv')

# Missingness sensitivity, separate gamma/mcar. Difference between gamma masks not a paired-mask causal effect.
sens=df[df.variant=='PG-control'].copy()
sens_summary=sens.groupby(['mechanism','gamma','dataset']).agg(n=('ARI','size'),ARI_mean=('ARI','mean'),ARI_sd=('ARI','std'),ACC_mean=('ACC','mean'),completion_RMSE=('missing_completion_rmse','mean')).reset_index().sort_values(['mechanism','gamma','dataset'])
frame_write(sens_summary,'gamma_and_mcar_by_dataset.csv')
sens_agg=sens.groupby(['mechanism','gamma']).agg(n=('ARI','size'),ARI_mean=('ARI','mean'),ARI_sd=('ARI','std'),ACC_mean=('ACC','mean'),completion_RMSE=('missing_completion_rmse','mean')).reset_index()
frame_write(sens_agg,'gamma_and_mcar_overall.csv')

# Reconstruction baselines. Same hidden entries and observed-only scaling.
imp=files_reader('imputation_runs')
if len(imp)!=180:raise RuntimeError(f'Expected 180 imputation setting jobs; got {len(imp)}')
imp_rows=[]
for r in imp:
    for m in r['rows']:
        imp_rows.append({k:r[k] for k in ['dataset','mechanism','rate','seed','mask_sha256']}|m)
for r in reference.reset_index().to_dict('records'):
    imp_rows.append(dict(dataset=r['dataset'],mechanism=r['mechanism'],rate=r['rate'],seed=r['seed'],
        mask_sha256=r['mask_sha256'],method='PG-JMADKFC',RMSE=r['missing_completion_rmse'],MAE=r['missing_completion_mae'],runtime_seconds=r['runtime_fit_seconds']))
imps=pd.DataFrame(imp_rows)
frame_write(imps,'masked_entry_completion_6_methods_1080_fits.csv')
assert len(imps)==1080 and imps.groupby(key).method.nunique().eq(6).all()
imp_sum=imps.groupby(['dataset','method']).agg(n=('RMSE','size'),mean_RMSE=('RMSE','mean'),sd_RMSE=('RMSE','std'),mean_MAE=('MAE','mean'),sd_MAE=('MAE','std')).reset_index()
frame_write(imp_sum,'masked_entry_completion_summary.csv')
imp_overall=imps.groupby('method').agg(n=('RMSE','size'),mean_RMSE=('RMSE','mean'),mean_MAE=('MAE','mean'),median_RMSE=('RMSE','median')).reset_index().sort_values('mean_RMSE')
frame_write(imp_overall,'masked_entry_completion_overall.csv')
imp_pivot=imps.pivot(index=key,columns='method',values='RMSE')
imp_delta=(imp_pivot.drop(columns=['PG-JMADKFC']).subtract(imp_pivot['PG-JMADKFC'],axis=0))
frame_write(imp_delta.reset_index(),'completion_rmse_paired_differences_baseline_minus_pg.csv')

# Observed metabolite profiles, decoded prototype consistency (not biological validation).
pp=pd.DataFrame(profile_rows);frame_write(pp,'prototype_observed_profile_correlations.csv')
pp_summary=pp[(pp.gamma==3)&(pp.mechanism.isin(['MAR','MNAR']))].groupby(['dataset']).agg(n=('pearson','size'),mean_r=('pearson','mean'),median_r=('pearson','median'),min_r=('pearson','min'),max_r=('pearson','max')).reset_index()
frame_write(pp_summary,'prototype_profile_correlation_summary.csv')
fr=pd.DataFrame(features_rows);frame_write(fr,'top10_feature_summaries_across_180_fit.csv')
# per dataset count seeds in which feature is ranked top-10 in either cluster, regardless of label swapping
sel=fr.groupby(['dataset','feature']).agg(n_entries=('rank','size'),n_unique_fits=('seed','nunique'),mean_abs_contrast=('abs_importance','mean')).reset_index()
# Count scenarios in which a feature appears in top-10 at least one seed, separately dataset.
sel2=fr.groupby(['dataset','feature']).apply(lambda x: x[['mechanism','rate','seed']].drop_duplicates().shape[0],include_groups=False).reset_index(name='n_distinct_scenario_seeds')
sel=sel.merge(sel2,on=['dataset','feature']).sort_values(['dataset','n_distinct_scenario_seeds'],ascending=[True,False]);frame_write(sel,'feature_selection_stability.csv')

# Perturbation diagnostics
pert=files_reader('perturbation_runs');pf=[];pr=[]
for r in pert:
    for x in r['per_feature']:
        pf.append({k:r[k] for k in ['dataset','mechanism','rate','seed']}|x)
    a=np.array([x['importance_abs_max'] for x in r['per_feature']]);b=np.array([x['membership_sensitivity'] for x in r['per_feature']]);
    rho,pval=spearmanr(a,b)
    threshold_hi=np.percentile(a,80);threshold_lo=np.percentile(a,20)
    pr.append(dict(dataset=r['dataset'],mechanism=r['mechanism'],rate=r['rate'],seed=r['seed'],
        spearman_importance_vs_perturbation=float(rho),
        mean_high_importance_feature_sensitivity=float(b[a>=threshold_hi].mean()),
        mean_low_importance_feature_sensitivity=float(b[a<=threshold_lo].mean()),
        fraction_cluster_flips_all_features=float(np.mean([x['cluster_flip_rate'] for x in r['per_feature']])),
        mean_prototype_profile_r=float(np.mean(r['proto_profile_pearson']))))
pert_features=pd.DataFrame(pf);frame_write(pert_features,'inference_time_feature_perturbation_all_features.csv')
pert_runs=pd.DataFrame(pr);frame_write(pert_runs,'feature_perturbation_summary_30_fits.csv')
pert_summary=pert_runs.groupby('dataset').agg(n=('spearman_importance_vs_perturbation','size'),mean_spearman_rho=('spearman_importance_vs_perturbation','mean'),
    median_spearman_rho=('spearman_importance_vs_perturbation','median'),
    mean_high_sensitivity=('mean_high_importance_feature_sensitivity','mean'),
    mean_low_sensitivity=('mean_low_importance_feature_sensitivity','mean'),
    mean_flip_fraction=('fraction_cluster_flips_all_features','mean')).reset_index()
frame_write(pert_summary,'feature_perturbation_dataset_summary.csv')

# Supplemental method report, no false publication claims.
lines=['# PG-JMADKFC — Additional Experiments: Executed Numerical Report','',
 'Generated from new individual-run JSON outputs. Every reported score was calculated from the archived masked reference data after training; labels were excluded from model fitting.','',
 '## Reproducibility and integrity','',
 f'- Frozen control fits reproduced: **{len(reference)} / 180**. Maximum absolute difference from original final benchmark: ARI {max_err["ARI"]:.3g}, NMI {max_err["NMI"]:.3g}, ACC {max_err["ACC"]:.3g}. All 180 original mask SHA-256 values matched.','',
 f'- Component ablation runs at gamma=3: **{len(main)}** = 18 scenarios × 10 seeds × {len(variants)} variants; 100+200 epochs each.','',
 f'- Missingness sensitivity runs (including gamma=3 controls): **{len(sens)}**; MCAR is an additional evaluation, not a replacement for MAR/MNAR.','',
 f'- Missing-entry completion comparison: **{len(imps)}** = 180 masks × 6 methods, measured on the *same masked entries*.','',
 f'- Post-fit feature-perturbation models: **{len(pert)}**, at 20% MAR and MNAR, five seeds per dataset–mechanism.','',
 '## Component ablation: larger ARI is better','',
 'Positive paired delta means that the PG control achieved higher ARI than the corresponding ablation. This is an empirical component contribution, not a guarantee of generalization beyond the three datasets.','',
 tabulate(summary,['variant','n_fits','mean_ARI','sd_ARI','mean_delta_ARI_control_minus_variant','n_positive_scenario_means']), '',
 '### By dataset','',
 tabulate(dataset_summary[['dataset','variant','mean_ARI','paired_delta_mean_ARI']],None), '',
 'Exploratory Wilcoxon tests on the 18 scenario means (Holm adjusted). These p-values MUST NOT be interpreted as based on 18 independent biological studies:', '', tabulate(stat_df), '',
 '## Sensitivity to mechanism strength and MCAR','',
 'For MAR/MNAR, gamma was set independently to 1, 3, and 5 and the observation mask was regenerated for each gamma. Do not treat scores from different gamma values as trained on identical masks.', '',
 tabulate(sens_agg), '',
 tabulate(sens_summary[['dataset','mechanism','gamma','n','ARI_mean','ARI_sd']]), '',
 '## Masked-entry completion comparison','',
 'All 180 scenario–seed masks were shared across the six compared approaches, and normalization parameters were estimated from observed entries. PG values are from its iterative completed input, not decoder reconstruction or observed-entry MSE. IterativeSVD is a simple fixed-rank (rank 5), 15-step reconstruction benchmark, not a reimplementation of nuclear-norm SoftImpute.', '',
 tabulate(imp_overall), '',
 tabulate(imp_sum[['dataset','method','mean_RMSE','sd_RMSE','mean_MAE']]), '',
 '## Prototype/profile consistency','',
 'Pearson correlation compares post-fit decoded prototypes with membership-weighted profiles estimated from observed data. It does not establish metabolite identity, biomarker validity or a biological mechanism.', '',
 tabulate(pp_summary), '',
 'Top recurring descriptive features (count of distinct scenario–seed combinations selected top 10 in either cluster):', '',
 tabulate(sel.groupby('dataset').head(8)[['dataset','feature','n_distinct_scenario_seeds','mean_abs_contrast']]), '',
 '## Exploratory phenotype association (post-fit, not an independent cohort)','',
 'We tested feature-wise differences between the two recorded phenotype groups on the original complete log-reference, after fitting the model, using Mann--Whitney U and Benjamini--Hochberg FDR. This internal statistical association is not independent biological validation. For anonymized M1... features, no verified molecule mapping exists in the data cache.','',
 tabulate(pd.read_csv(ROOT/'phenotype_association'/'phenotype_association_enrichment.csv').drop(columns=['top10_named_features'])), '',
 '## Inference-time feature perturbation (diagnostic)','',
 'Each observed metabolite was permuted among observed samples without changing missingness mask, model weights or other features. The reported effect is sensitivity of fuzzy memberships to this intervention, not causal biological relevance or stability under retraining.', '',
 tabulate(pert_summary), '',
 '## Interpretation caveats and unresolved reviewer items','',
 '- Ablation outcomes are from 3 binary-class metabolomics datasets only. Correlated masks and seeds on the same dataset are not independent biological cohorts.','- No real measured-MNAR dataset or multiclass external cohort has been validated here.','- Only the human_cachexia prepared file contains explicit metabolite names. MTBLS161 and MTBLS404 use M1/M2/... numeric feature identifiers without source-verified metabolite mappings; naming specific molecules requires an audited annotation file.','- Scientific biological pathway/experimental validation remains out of scope of these computational tests.','- Feature perturbation measures model-level sensitivity, not independent experimental confirmation.','- The test settings gamma=1 and 5 are prespecified stress tests and must not be used to silently choose a best-looking gamma after examining phenotype labels.','- Inference based on the 18 scenario means is exploratory because scenarios share three underlying biological datasets.','- Main benchmark Results and Discussion should be updated by replacing placeholders, not retroactively claiming experiments that failed to execute.','']
(OUT/'ADDITIONAL_EXPERIMENTS_REPORT.md').write_text('\n'.join(lines),encoding='utf8')

# Audit manifest including completed files and hashes.
manifest={'main_ablation_fits':int(len(main)),'variants':sorted(variants),'missingness_sensitivity_fits':int(len(sens)),
 'imputation_fits':int(len(imps)),'feature_perturbation_model_fits':int(len(pert)),
 'replayed_control_fits':int(len(reference)),'replay_max_metric_errors':max_err,'replay_max_completion_MSE_error':max_completion_replay,
 'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.csv'))}}
(OUT/'EXPERIMENT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('SUCCEEDED',json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2));print('Report',OUT/'ADDITIONAL_EXPERIMENTS_REPORT.md')
