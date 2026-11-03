#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, numpy as np
R=Path(__file__).resolve().parents[1]/'final_analysis'
A=pd.read_csv(R/'ablation_by_dataset.csv')
O=pd.read_csv(R/'ablation_overall_comparison.csv')
G=pd.read_csv(R/'gamma_and_mcar_by_dataset.csv')
I=pd.read_csv(R/'masked_entry_completion_summary.csv')
P=pd.read_csv(R/'prototype_profile_correlation_summary.csv')
S=pd.read_csv(R/'feature_perturbation_dataset_summary.csv')

def fm(x):return f'{x:.4f}' if pd.notna(x) else '--'
def val(x):return f'{x:.3f}' if pd.notna(x) else '--'
def esc(s):return s.replace('_',r'\_').replace('%',r'\%')
D=['MTBLS161','MTBLS404','human_cachexia']
L=[r'% This insert is generated from executed supplementary experiments. Include within Section 5 after updating main-table references.',
   r'\subsection{Additional Controlled Experiments}',
   'The following component-ablation, missingness-sensitivity, masked-entry recovery, and metabolite-sensitivity experiments were conducted using the same three datasets and observed-entry normalization as the main benchmark. Class labels were not used to fit any model. The analyses provide additional dataset-dependent evidence, not an independent biological validation cohort.',
   '',r'\subsubsection{Component Ablation}',
   'To evaluate the separate influence of the proposed components, the complete PG-JMADKFC configuration was compared against controlled variants in which one component was disabled or replaced. Each configuration was fitted using the original 18 MAR/MNAR scenarios and 10 random seeds (180 fits per variant). All comparisons used identical masks within each dataset--mechanism--rate--seed condition. The tested changes include removing prototype-guided completion, the mask input, gradient safeguarding, prototype separation, latent variance safeguarding, or the clustering gradient, as well as shuffling the mask input, using a global rather than locally adaptive kernel bandwidth, or evaluating a reconstruction objective with zero targets at masked entries. A higher ARI for the complete model in a paired setting is evidence of a favorable contribution under that setting, not proof of universal superiority.',
   '',r'\begin{table*}[t]',r'\centering',r'\caption{Controlled component ablation of PG-JMADKFC on three metabolomics datasets. ARI values are means across 60 fits per dataset (two mechanisms, three rates, ten seeds). The final column reports the paired difference between the complete method and each ablation, averaged across 180 matched fits; positive values favor the complete method.}',r'\label{tab:pg_component_ablation}',r'\scriptsize',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lccccc}',r'\toprule',r'Configuration & MTBLS161 & MTBLS404 & Human Cachexia & Mean ARI & $\Delta\mathrm{ARI}$ (PG--ablation) \\',r'\midrule']
for variant in ['PG-control','without-completion','without-mask-channel','shuffled-mask-channel','without-gradient-guard','without-separation','without-variance-safeguard','global-bandwidth-kernel','zero-target-reconstruction','without-clustering-gradient','mask-scale-0.10','mask-scale-0.50']:
    sub=O[O.variant==variant]
    if sub.empty:continue
    vals=[]
    for d in D:
        row=A[(A.variant==variant)&(A.dataset==d)].iloc[0]
        vals.append(fm(row.mean_ARI))
    r=sub.iloc[0]
    desc={'PG-control':'PG-JMADKFC (complete)','without-completion':'Without prototype completion','without-mask-channel':'Without mask input','shuffled-mask-channel':'Shuffled mask input','without-gradient-guard':'Without gradient safeguard','without-separation':'Without prototype separation','without-variance-safeguard':'Without variance safeguard','global-bandwidth-kernel':'Global bandwidth kernel','zero-target-reconstruction':'Zero-target reconstruction','without-clustering-gradient':'Without clustering gradient','mask-scale-0.10':'Mask scale 0.10','mask-scale-0.50':'Mask scale 0.50'}[variant]
    L.append(esc(desc)+' & '+' & '.join(vals)+f' & {fm(r.mean_ARI)} & {fm(r.mean_delta_ARI_control_minus_variant)} '+r'\\')
L.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}','',
          'Table~\\ref{tab:pg_component_ablation} summarizes the observed ablation effects. Component contributions varied across the three datasets; therefore, the full set of paired scenario-level results is provided in the supplementary experiment files. These findings are descriptive of the evaluated settings and do not establish that every mechanism consistently improves clustering.',
          '',r'\subsubsection{Sensitivity to Missingness Mechanism and Strength}',
          'To extend the main MAR/MNAR analysis beyond a fixed missingness-strength parameter, the proposed method was independently evaluated under $\\gamma\\in\\{1,3,5\\}$ at 10\\%, 20\\%, and 30\\% missingness. For each value of $\\gamma$, masks were newly sampled; consequently, results across different strengths are comparisons of experimental regimes, not matched-mask ablations. Additional missing-completely-at-random (MCAR) experiments were performed using the same three rates and ten seeds. These experiments examine the sensitivity of the proposed method to masking conditions without assuming that any particular missingness mechanism is always informative for clustering.',
          '',r'\begin{table*}[t]',r'\centering',r'\caption{Mean ARI under MAR/MNAR missingness strengths and MCAR. Each dataset cell aggregates 30 fits (three rates and ten seeds). MCAR has no strength parameter.}',r'\label{tab:pg_gamma_mcar}',r'\small',r'\begin{tabular}{lccccc}',r'\toprule',r'Mechanism & $\gamma$ & MTBLS161 & MTBLS404 & Human Cachexia & Overall \\',r'\midrule'])
for mech in ['MAR','MNAR','MCAR']:
    for gam in ([1.,3.,5.] if mech!='MCAR' else [3.]):
        sub=G[(G.mechanism==mech)&(G.gamma==gam)]
        vals=[fm(sub[sub.dataset==d].ARI_mean.iloc[0]) for d in D]
        L.append(mech+' & '+(str(int(gam)) if mech!='MCAR' else '--')+' & '+' & '.join(vals)+' & '+fm(sub.ARI_mean.mean())+r'\\')
L.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}','',
          r'\subsubsection{Masked-Entry Completion Under Matched Masks}',
          'For each of the 180 frozen MAR/MNAR mask realizations, the final prototype-guided completed input was compared with observed-only mean, median, KNN, MICE, and rank-five iterative low-rank SVD completion. RMSE and MAE were calculated only on artificially hidden values after transforming the complete reference with parameters estimated exclusively from observed entries. The reconstruction outputs from the neural decoder are not conflated with the prototype-guided completed inputs. The low-rank SVD comparison is a simple fixed-rank reference and is not presented as a state-of-the-art nuclear-norm solver.',
          '',r'\begin{table*}[t]',r'\centering',r'\caption{Masked-entry completion performance on held-out values. Mean RMSE (MAE) is averaged over 60 masks for each dataset. Lower is better.}',r'\label{tab:pg_completion_baselines}',r'\small',r'\begin{tabular}{lccc}',r'\toprule',r'Method & MTBLS161 & MTBLS404 & Human Cachexia \\',r'\midrule'])
for method in ['PG-JMADKFC','Mean','Median','KNN','MICE','IterativeSVD']:
    vals=[]
    for d in D:
        r=I[(I.dataset==d)&(I.method==method)].iloc[0]
        vals.append(f'{fm(r.mean_RMSE)} ({fm(r.mean_MAE)})')
    L.append(esc(method)+' & '+' & '.join(vals)+r'\\')
L.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}','',
          r'\subsubsection{Prototype Consistency and Inference-Time Feature Sensitivity}',
          'Decoded metabolite-space prototypes were compared with fuzzy membership-weighted profiles computed only from observed measurements. These correlations describe post-fit representational agreement, not biological pathway validation. In a separate perturbation analysis, each observed metabolite was permuted among its observed samples while holding the fitted PG-JMADKFC model, observation mask, and all other features fixed. The resulting change in fuzzy memberships quantifies inference-time model sensitivity; it is not a biological causal effect.',
          '',r'\begin{table*}[t]',r'\centering',r'\caption{Prototype--profile agreement and feature-sensitivity diagnostic. Mean Spearman $\rho$ quantifies association of descriptive metabolite importance with model sensitivity, using five seeds per MAR/MNAR 20\% condition.}',r'\label{tab:pg_feature_sensitivity}',r'\small',r'\begin{tabular}{lcc}',r'\toprule',r'Dataset & Prototype Pearson $r$ & Importance/sensitivity $\rho$ \\',r'\midrule'])
for d in D:
    pp=P[P.dataset==d].iloc[0];s=S[S.dataset==d].iloc[0]
    L.append(esc('Human Cachexia' if d=='human_cachexia' else d)+f' & {fm(pp.mean_r)} & {fm(s.mean_spearman_rho)} '+r'\\')
L.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}','',
 'The two MetaboLights prepared matrices contain anonymized feature labels (e.g., M1, M2) without a verified source-to-metabolite mapping in the benchmark archive. Accordingly, source-specific metabolite names and biological mechanisms should not be inferred from these indices. The named features in \\texttt{human\\_cachexia} enable an exploratory descriptive analysis, but independent biological validation is not established by this computational benchmark.',
 '',r'\paragraph{Limitations of the additional experiments.}',
 'All analyses reuse three binary-class datasets, so the 18 simulated conditions do not constitute 18 independent biological cohorts. The new tests clarify sensitivity to model components and missingness but do not establish performance on real missingness, multiclass phenotypes, or independent clinical cohorts. Any inferential comparison across scenario means must therefore account for this dependence, and the effects should be reported as conditional on the studied data and simulation choices.'])
path=R/'Additional_Experiments_PG_JMADKFC_Manuscript_Insert.tex';path.write_text('\n'.join(L)+'\n');print(path,'lines',len(L))
