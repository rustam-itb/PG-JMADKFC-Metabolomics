# PG-JMADKFC — Additional Experiments: Executed Numerical Report

Generated from new individual-run JSON outputs. Every reported score was calculated from the archived masked reference data after training; labels were excluded from model fitting.

## Reproducibility and integrity

- Frozen control fits reproduced: **180 / 180**. Maximum absolute difference from original final benchmark: ARI 9.71e-17, NMI 9.89e-17, ACC 0. All 180 original mask SHA-256 values matched.

- Component ablation runs at gamma=3: **2160** = 18 scenarios × 10 seeds × 12 variants; 100+200 epochs each.

- Missingness sensitivity runs (including gamma=3 controls): **630**; MCAR is an additional evaluation, not a replacement for MAR/MNAR.

- Missing-entry completion comparison: **1080** = 180 masks × 6 methods, measured on the *same masked entries*.

- Post-fit feature-perturbation models: **30**, at 20% MAR and MNAR, five seeds per dataset–mechanism.

## Component ablation: larger ARI is better

Positive paired delta means that the PG control achieved higher ARI than the corresponding ablation. This is an empirical component contribution, not a guarantee of generalization beyond the three datasets.

| variant                     |   n_fits |   mean_ARI |   sd_ARI |   mean_delta_ARI_control_minus_variant |   n_positive_scenario_means |
|:----------------------------|---------:|-----------:|---------:|---------------------------------------:|----------------------------:|
| without-gradient-guard      |      180 |     0.1239 |   0.0651 |                                -0.0126 |                      6.0000 |
| mask-scale-0.10             |      180 |     0.1172 |   0.0811 |                                -0.0059 |                      9.0000 |
| shuffled-mask-channel       |      180 |     0.1125 |   0.0792 |                                -0.0013 |                      6.0000 |
| without-variance-safeguard  |      180 |     0.1123 |   0.0816 |                                -0.0010 |                      8.0000 |
| without-mask-channel        |      180 |     0.1118 |   0.0799 |                                -0.0006 |                      8.0000 |
| PG-control                  |      180 |     0.1112 |   0.0815 |                                 0.0000 |                    nan      |
| mask-scale-0.50             |      180 |     0.1111 |   0.0789 |                                 0.0001 |                     10.0000 |
| without-completion          |      180 |     0.1101 |   0.0813 |                                 0.0011 |                      9.0000 |
| without-separation          |      180 |     0.1073 |   0.0823 |                                 0.0040 |                     10.0000 |
| without-clustering-gradient |      180 |     0.1042 |   0.0714 |                                 0.0071 |                      7.0000 |
| zero-target-reconstruction  |      180 |     0.0934 |   0.0733 |                                 0.0178 |                     13.0000 |
| global-bandwidth-kernel     |      180 |     0.0750 |   0.0627 |                                 0.0362 |                     16.0000 |

### By dataset

| dataset        | variant                     |   mean_ARI |   paired_delta_mean_ARI |
|:---------------|:----------------------------|-----------:|------------------------:|
| MTBLS161       | PG-control                  |     0.1511 |                nan      |
| MTBLS161       | global-bandwidth-kernel     |     0.1145 |                  0.0366 |
| MTBLS161       | mask-scale-0.10             |     0.1594 |                 -0.0083 |
| MTBLS161       | mask-scale-0.50             |     0.1529 |                 -0.0019 |
| MTBLS161       | shuffled-mask-channel       |     0.1503 |                  0.0008 |
| MTBLS161       | without-clustering-gradient |     0.1007 |                  0.0504 |
| MTBLS161       | without-completion          |     0.1495 |                  0.0016 |
| MTBLS161       | without-gradient-guard      |     0.1063 |                  0.0448 |
| MTBLS161       | without-mask-channel        |     0.1547 |                 -0.0036 |
| MTBLS161       | without-separation          |     0.1554 |                 -0.0044 |
| MTBLS161       | without-variance-safeguard  |     0.1526 |                 -0.0015 |
| MTBLS161       | zero-target-reconstruction  |     0.1279 |                  0.0232 |
| MTBLS404       | PG-control                  |     0.0231 |                nan      |
| MTBLS404       | global-bandwidth-kernel     |     0.0185 |                  0.0046 |
| MTBLS404       | mask-scale-0.10             |     0.0233 |                 -0.0002 |
| MTBLS404       | mask-scale-0.50             |     0.0244 |                 -0.0013 |
| MTBLS404       | shuffled-mask-channel       |     0.0260 |                 -0.0029 |
| MTBLS404       | without-clustering-gradient |     0.0344 |                 -0.0112 |
| MTBLS404       | without-completion          |     0.0232 |                 -0.0001 |
| MTBLS404       | without-gradient-guard      |     0.0845 |                 -0.0613 |
| MTBLS404       | without-mask-channel        |     0.0229 |                  0.0002 |
| MTBLS404       | without-separation          |     0.0221 |                  0.0011 |
| MTBLS404       | without-variance-safeguard  |     0.0225 |                  0.0006 |
| MTBLS404       | zero-target-reconstruction  |     0.0233 |                 -0.0002 |
| human_cachexia | PG-control                  |     0.1595 |                nan      |
| human_cachexia | global-bandwidth-kernel     |     0.0921 |                  0.0674 |
| human_cachexia | mask-scale-0.10             |     0.1688 |                 -0.0093 |
| human_cachexia | mask-scale-0.50             |     0.1560 |                  0.0036 |
| human_cachexia | shuffled-mask-channel       |     0.1613 |                 -0.0017 |
| human_cachexia | without-clustering-gradient |     0.1775 |                 -0.0179 |
| human_cachexia | without-completion          |     0.1578 |                  0.0018 |
| human_cachexia | without-gradient-guard      |     0.1809 |                 -0.0213 |
| human_cachexia | without-mask-channel        |     0.1578 |                  0.0017 |
| human_cachexia | without-separation          |     0.1444 |                  0.0152 |
| human_cachexia | without-variance-safeguard  |     0.1617 |                 -0.0022 |
| human_cachexia | zero-target-reconstruction  |     0.1290 |                  0.0306 |

Exploratory Wilcoxon tests on the 18 scenario means (Holm adjusted). These p-values MUST NOT be interpreted as based on 18 independent biological studies:

| variant                     |   n_scenario_means |   mean_delta_PG_minus_variant |   scenarios_favor_PG |   scenarios_favor_variant |   wilcoxon_p_exploratory |   holm_adjusted_p_exploratory |
|:----------------------------|-------------------:|------------------------------:|---------------------:|--------------------------:|-------------------------:|------------------------------:|
| global-bandwidth-kernel     |                 18 |                        0.0362 |                   16 |                         2 |                   0.0001 |                        0.0012 |
| mask-scale-0.10             |                 18 |                       -0.0059 |                    9 |                         9 |                   0.1964 |                        1.0000 |
| mask-scale-0.50             |                 18 |                        0.0001 |                   10 |                         8 |                   0.7987 |                        1.0000 |
| shuffled-mask-channel       |                 18 |                       -0.0013 |                    6 |                        12 |                   0.3927 |                        1.0000 |
| without-clustering-gradient |                 18 |                        0.0071 |                    7 |                        11 |                   0.9323 |                        1.0000 |
| without-completion          |                 18 |                        0.0011 |                    9 |                         9 |                   0.6397 |                        1.0000 |
| without-gradient-guard      |                 18 |                       -0.0126 |                    6 |                        12 |                   0.3247 |                        1.0000 |
| without-mask-channel        |                 18 |                       -0.0006 |                    8 |                        10 |                   0.6397 |                        1.0000 |
| without-separation          |                 18 |                        0.0040 |                   10 |                         8 |                   0.3247 |                        1.0000 |
| without-variance-safeguard  |                 18 |                       -0.0010 |                    8 |                         9 |                   0.3603 |                        1.0000 |
| zero-target-reconstruction  |                 18 |                        0.0178 |                   13 |                         5 |                   0.0208 |                        0.2081 |

## Sensitivity to mechanism strength and MCAR

For MAR/MNAR, gamma was set independently to 1, 3, and 5 and the observation mask was regenerated for each gamma. Do not treat scores from different gamma values as trained on identical masks.

| mechanism   |   gamma |   n |   ARI_mean |   ARI_sd |   ACC_mean |   completion_RMSE |
|:------------|--------:|----:|-----------:|---------:|-----------:|------------------:|
| MAR         |  1.0000 |  90 |     0.1162 |   0.0879 |     0.6616 |            0.9594 |
| MAR         |  3.0000 |  90 |     0.1083 |   0.0777 |     0.6580 |            0.9594 |
| MAR         |  5.0000 |  90 |     0.1001 |   0.0843 |     0.6519 |            0.9460 |
| MCAR        |  3.0000 |  90 |     0.1156 |   0.0796 |     0.6666 |            0.9735 |
| MNAR        |  1.0000 |  90 |     0.1105 |   0.0839 |     0.6610 |            1.0218 |
| MNAR        |  3.0000 |  90 |     0.1142 |   0.0855 |     0.6642 |            1.2481 |
| MNAR        |  5.0000 |  90 |     0.1212 |   0.0787 |     0.6706 |            1.4900 |

| dataset        | mechanism   |   gamma |   n |   ARI_mean |   ARI_sd |
|:---------------|:------------|--------:|----:|-----------:|---------:|
| MTBLS161       | MAR         |  1.0000 |  30 |     0.1807 |   0.0776 |
| MTBLS404       | MAR         |  1.0000 |  30 |     0.0155 |   0.0128 |
| human_cachexia | MAR         |  1.0000 |  30 |     0.1523 |   0.0369 |
| MTBLS161       | MAR         |  3.0000 |  30 |     0.1678 |   0.0656 |
| MTBLS404       | MAR         |  3.0000 |  30 |     0.0194 |   0.0190 |
| human_cachexia | MAR         |  3.0000 |  30 |     0.1376 |   0.0334 |
| MTBLS161       | MAR         |  5.0000 |  30 |     0.1638 |   0.0985 |
| MTBLS404       | MAR         |  5.0000 |  30 |     0.0207 |   0.0174 |
| human_cachexia | MAR         |  5.0000 |  30 |     0.1159 |   0.0286 |
| MTBLS161       | MCAR        |  3.0000 |  30 |     0.1618 |   0.0785 |
| MTBLS404       | MCAR        |  3.0000 |  30 |     0.0297 |   0.0168 |
| human_cachexia | MCAR        |  3.0000 |  30 |     0.1552 |   0.0390 |
| MTBLS161       | MNAR        |  1.0000 |  30 |     0.1493 |   0.0899 |
| MTBLS404       | MNAR        |  1.0000 |  30 |     0.0244 |   0.0188 |
| human_cachexia | MNAR        |  1.0000 |  30 |     0.1577 |   0.0402 |
| MTBLS161       | MNAR        |  3.0000 |  30 |     0.1344 |   0.0860 |
| MTBLS404       | MNAR        |  3.0000 |  30 |     0.0269 |   0.0118 |
| human_cachexia | MNAR        |  3.0000 |  30 |     0.1814 |   0.0436 |
| MTBLS161       | MNAR        |  5.0000 |  30 |     0.1397 |   0.0720 |
| MTBLS404       | MNAR        |  5.0000 |  30 |     0.0413 |   0.0379 |
| human_cachexia | MNAR        |  5.0000 |  30 |     0.1825 |   0.0395 |

## Masked-entry completion comparison

All 180 scenario–seed masks were shared across the six compared approaches, and normalization parameters were estimated from observed entries. PG values are from its iterative completed input, not decoder reconstruction or observed-entry MSE. IterativeSVD is a simple fixed-rank (rank 5), 15-step reconstruction benchmark, not a reimplementation of nuclear-norm SoftImpute.

| method       |   n |   mean_RMSE |   mean_MAE |   median_RMSE |
|:-------------|----:|------------:|-----------:|--------------:|
| MICE         | 180 |      0.8982 |     0.6330 |        0.8676 |
| KNN          | 180 |      0.9454 |     0.6775 |        0.8548 |
| IterativeSVD | 180 |      0.9798 |     0.6784 |        0.9188 |
| PG-JMADKFC   | 180 |      1.1038 |     0.8110 |        0.9537 |
| Mean         | 180 |      1.2736 |     0.9872 |        1.2322 |
| Median       | 180 |      1.2892 |     0.9924 |        1.2578 |

| dataset        | method       |   mean_RMSE |   sd_RMSE |   mean_MAE |
|:---------------|:-------------|------------:|----------:|-----------:|
| MTBLS161       | IterativeSVD |      1.1394 |    0.1884 |     0.8134 |
| MTBLS161       | KNN          |      1.0786 |    0.1979 |     0.7899 |
| MTBLS161       | MICE         |      1.0718 |    0.2198 |     0.7683 |
| MTBLS161       | Mean         |      1.3060 |    0.1912 |     0.9983 |
| MTBLS161       | Median       |      1.3033 |    0.1846 |     0.9802 |
| MTBLS161       | PG-JMADKFC   |      1.2285 |    0.1920 |     0.9175 |
| MTBLS404       | IterativeSVD |      0.9889 |    0.2729 |     0.6112 |
| MTBLS404       | KNN          |      0.9752 |    0.2599 |     0.6450 |
| MTBLS404       | MICE         |      0.8073 |    0.1996 |     0.5151 |
| MTBLS404       | Mean         |      1.2866 |    0.3329 |     0.9428 |
| MTBLS404       | Median       |      1.3074 |    0.3754 |     0.9558 |
| MTBLS404       | PG-JMADKFC   |      1.2136 |    0.3473 |     0.8425 |
| human_cachexia | IterativeSVD |      0.8110 |    0.0771 |     0.6106 |
| human_cachexia | KNN          |      0.7824 |    0.0546 |     0.5975 |
| human_cachexia | MICE         |      0.8156 |    0.1475 |     0.6158 |
| human_cachexia | Mean         |      1.2283 |    0.1009 |     1.0207 |
| human_cachexia | Median       |      1.2569 |    0.1107 |     1.0411 |
| human_cachexia | PG-JMADKFC   |      0.8692 |    0.0664 |     0.6729 |

## Prototype/profile consistency

Pearson correlation compares post-fit decoded prototypes with membership-weighted profiles estimated from observed data. It does not establish metabolite identity, biomarker validity or a biological mechanism.

| dataset        |   n |   mean_r |   median_r |   min_r |   max_r |
|:---------------|----:|---------:|-----------:|--------:|--------:|
| MTBLS161       | 120 |   0.7326 |     0.7243 |  0.1912 |  0.9827 |
| MTBLS404       | 120 |   0.9104 |     0.9231 |  0.6162 |  0.9787 |
| human_cachexia | 120 |   0.8737 |     0.8901 |  0.6104 |  0.9515 |

Top recurring descriptive features (count of distinct scenario–seed combinations selected top 10 in either cluster):

| dataset        | feature       |   n_distinct_scenario_seeds |   mean_abs_contrast |
|:---------------|:--------------|----------------------------:|--------------------:|
| MTBLS161       | M15           |                          60 |              0.2835 |
| MTBLS161       | M17           |                          58 |              0.3674 |
| MTBLS161       | M19           |                          58 |              0.3314 |
| MTBLS161       | M25           |                          57 |              0.2702 |
| MTBLS161       | M2            |                          52 |              0.2524 |
| MTBLS161       | M22           |                          51 |              0.2563 |
| MTBLS161       | M23           |                          47 |              0.2511 |
| MTBLS161       | M1            |                          46 |              0.2737 |
| MTBLS404       | M116          |                          58 |              0.4655 |
| MTBLS404       | M115          |                          57 |              0.4694 |
| MTBLS404       | M24           |                          56 |              0.4607 |
| MTBLS404       | M103          |                          54 |              0.4443 |
| MTBLS404       | M65           |                          46 |              0.4240 |
| MTBLS404       | M105          |                          41 |              0.4550 |
| MTBLS404       | M56           |                          41 |              0.4221 |
| MTBLS404       | M38           |                          37 |              0.4253 |
| human_cachexia | Creatinine    |                          59 |              0.6544 |
| human_cachexia | Glutamine     |                          54 |              0.6843 |
| human_cachexia | Dimethylamine |                          53 |              0.6618 |
| human_cachexia | cis-Aconitate |                          53 |              0.6426 |
| human_cachexia | Valine        |                          52 |              0.6978 |
| human_cachexia | Alanine       |                          50 |              0.6711 |
| human_cachexia | Tryptophan    |                          48 |              0.6438 |
| human_cachexia | Ethanolamine  |                          46 |              0.6603 |

## Exploratory phenotype association (post-fit, not an independent cohort)

We tested feature-wise differences between the two recorded phenotype groups on the original complete log-reference, after fitting the model, using Mann--Whitney U and Benjamini--Hochberg FDR. This internal statistical association is not independent biological validation. For anonymized M1... features, no verified molecule mapping exists in the data cache.

| dataset        |   n_features |   n_BH_sig |   top10_selected_features_with_BH_sig |   hypergeometric_p_descriptive |
|:---------------|-------------:|-----------:|--------------------------------------:|-------------------------------:|
| MTBLS161       |           29 |          6 |                                     5 |                         0.0105 |
| MTBLS404       |          120 |         51 |                                     7 |                         0.0671 |
| human_cachexia |           63 |         54 |                                    10 |                         0.1872 |

## Inference-time feature perturbation (diagnostic)

Each observed metabolite was permuted among observed samples without changing missingness mask, model weights or other features. The reported effect is sensitivity of fuzzy memberships to this intervention, not causal biological relevance or stability under retraining.

| dataset        |   n |   mean_spearman_rho |   median_spearman_rho |   mean_high_sensitivity |   mean_low_sensitivity |   mean_flip_fraction |
|:---------------|----:|--------------------:|----------------------:|------------------------:|-----------------------:|---------------------:|
| MTBLS161       |  10 |              0.2564 |                0.2123 |                  0.0148 |                 0.0112 |               0.0245 |
| MTBLS404       |  10 |              0.0756 |                0.0850 |                  0.0102 |                 0.0090 |               0.0038 |
| human_cachexia |  10 |             -0.0271 |               -0.0094 |                  0.0166 |                 0.0174 |               0.0015 |

## Interpretation caveats and unresolved reviewer items

- Ablation outcomes are from 3 binary-class metabolomics datasets only. Correlated masks and seeds on the same dataset are not independent biological cohorts.
- No real measured-MNAR dataset or multiclass external cohort has been validated here.
- Only the human_cachexia prepared file contains explicit metabolite names. MTBLS161 and MTBLS404 use M1/M2/... numeric feature identifiers without source-verified metabolite mappings; naming specific molecules requires an audited annotation file.
- Scientific biological pathway/experimental validation remains out of scope of these computational tests.
- Feature perturbation measures model-level sensitivity, not independent experimental confirmation.
- The test settings gamma=1 and 5 are prespecified stress tests and must not be used to silently choose a best-looking gamma after examining phenotype labels.
- Inference based on the 18 scenario means is exploratory because scenarios share three underlying biological datasets.
- Main benchmark Results and Discussion should be updated by replacing placeholders, not retroactively claiming experiments that failed to execute.
