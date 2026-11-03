# PG-JMADKFC — Reproducible Code and Experiments

**Prototype-Guided Joint Missingness-Aware Deep Kernel Fuzzy Clustering for Incomplete Metabolomics Data**

This repository provides the **actual Python implementation** of PG-JMADKFC and the computational materials supporting the major-revision experiments reported in *Uncertainty-Preserving Deep Kernel Fuzzy Clustering with Interpretable Prototypes for Incomplete Metabolomics Data* (Scientific Reports, manuscript under revision). This repository is a research artifact; it does **not** claim paper acceptance or publication.

## What is included

| Location | Purpose |
|---|---|
| `src/` | Original research implementation used for PG-JMADKFC and 11 comparison algorithms, retained without silently modifying the research code |
| `scripts/reproduce_paper_benchmark.py` | New three-dataset, 12-method wrapper for the **reported** benchmark (bypasses historical six-dataset protocol validation) |
| `scripts/prepare_three_datasets.py` | Convert the exact study-prepared source tables into analytic NPZ matrices |
| `scripts/run_ablation.py` | All 12 PG configurations, missingness sensitivity (`gamma` and MCAR) |
| `scripts/run_imputation.py` | Completion comparison against mean, median, KNN, MICE, and iterative SVD |
| `scripts/run_perturbation.py` | Post-training observed-feature permutation sensitivity |
| `scripts/phenotype_assoc.py` | Exploratory *post-fit* association of features with reference phenotypes |
| `scripts/analyze_results.py` | Generate ablation, sensitivity, completion, prototype/profile and significance tables from JSON |
| `scripts/verify_results.py` | Audit included result files without rerunning model training |
| `results/benchmark/` | Fixed main evaluation: 2,160 fit rows and 216 scenario-level mean/SD rows |
| `ablation_runs/`, `imputation_runs/`, `perturbation_runs/` | Saved individual computation outputs (no patient-level intensities) |
| `final_analysis/` | Published-table inputs, numerical audit, and analysis manifest |
| `data/README.md` | Source datasets, provenance and reproduction instructions |

Only **PG-JMADKFC** is presented as the proposed method. The model source also contains historical development classes retained for traceability; they are **not included as separate comparison methods** in the reported 12-method paper benchmark.

## Study protocol

- Datasets: MTBLS161 (59 samples, 29 metabolites), MTBLS404 (184, 120), Human Cachexia (77, 63).
- Primary simulated missingness: MAR / MNAR at rates 10%, 20%, 30%, `gamma=3`.
- Repeated seeds: `1061 2081 3109 4127 5147 6163 7187 8209 9221 10243`.
- Proposed-model configuration: `c=2`, fuzzy exponent `alpha=2`, 100 reconstruction pretraining epochs + 200 joint epochs, burn-in 50 epochs, completion step 0.25.
- Main reported comparison: PG-JMADKFC + **11** published-method benchmark implementations = **2,160 fits** (3 datasets × 2 mechanisms × 3 rates × 10 seeds × 12 methods).
- Additional analyses: 12-component-variant experiment including control (2,160 fits), MAR/MNAR `gamma` sensitivity plus MCAR (630 fits including controls), 1,080 masked-entry completion evaluations, and 30 feature-perturbation models. Counts across experiments overlap; do not add them as unique base fits.
- Class labels are used only after unsupervised training, for ARI/NMI/ACC and post-fit phenotype association.

## Environment

Use Python 3.10+ (preferably 3.11), a CPU-capable PyTorch installation, and install:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements define compatible lower bounds, not a bitwise reproducibility guarantee across all hardware/software stacks. Run `python scripts/print_environment.py` to capture the versions actually installed. The archived metrics pass numerical integrity checks, but an additional fresh PG training run did **not** exactly match one archived score despite an identical mask; consult `docs/REPRODUCIBILITY_AUDIT.md` before making any exact-replay claim. Agreement may depend on execution environment and numerical kernels; the immediate cause is not yet resolved.

## Fast verification (no sample-level data required)

```bash
python scripts/verify_results.py
```

The verification checks archived main benchmark size/methods, ablation records, completion records, numerical means reported in the manuscript, and the JSON-level source/mask identities. **This is a static integrity check, not a fresh training replay.**

## Reproduce one PG-JMADKFC model fit

1. Obtain the **exact study-prepared** input tables, follow `data/README.md`, and prepare the NPZ files.
2. Run:

```bash
python scripts/reproduce_paper_benchmark.py --dataset MTBLS161 --mechanism MAR --rate 0.10 --seed 1061 --method PG-JMADKFC
```

The default output is under `results/reproduced/`. The wrapper runs 100+200 epochs unless `--smoke` is set explicitly; a reduced-epoch smoke run **must not** be used for manuscript-level numeric comparisons.

To run a representative baseline:

```bash
python scripts/reproduce_paper_benchmark.py --dataset MTBLS161 --mechanism MAR --rate 0.10 --seed 1061 --method DEC
```

To run the full 2,160-job main benchmark (will take substantial compute):

```bash
python scripts/reproduce_paper_benchmark.py --all --max-jobs 2160
```

The runner skips existing successful JSON files. The original result rows in `results/benchmark/` are an immutable numerical reference, not files to overwrite.

## Reproduce reviewer-requested additional experiments

After preparing the three input arrays, run commands from the repository root:

```bash
python scripts/run_ablation.py --workers 3
python scripts/run_ablation.py --workers 3 --variants global-bandwidth-kernel zero-target-reconstruction without-clustering-gradient
python scripts/run_ablation.py --workers 3 --variants mask-scale-0.10 mask-scale-0.50
python scripts/run_ablation.py --workers 3 --variants PG-control --gamma 1
python scripts/run_ablation.py --workers 3 --variants PG-control --gamma 5
python scripts/run_ablation.py --workers 3 --variants PG-control --mechanisms MCAR
python scripts/run_imputation.py --workers 3
python scripts/run_perturbation.py --workers 3
python scripts/phenotype_assoc.py
python scripts/analyze_results.py
python scripts/write_manuscript_insert.py
```

Outputs are saved per scenario/seed and can be resumed. For the exact control-run replay and audit, run all commands and compare to `final_analysis/EXPERIMENT_MANIFEST.json`. **Do not** delete or modify archived results in the published branch; use a separate working branch or copy for a fresh run.

## Principal numerical findings (descriptive)

- PG-JMADKFC mean ARI: MTBLS161 **0.1511**, MTBLS404 **0.0231**, Human Cachexia **0.1595**.
- Adaptive-kernel vs global-bandwidth ablation: overall mean ARI **0.1112 vs 0.0750**.
- Masked observed-entry reconstruction vs zero-target ablation: **0.1112 vs 0.0934**.
- Without gradient guard: **0.1239** overall mean ARI, with dataset-dependent trade-offs.
- Masked-entry completion RMSE: PG-JMADKFC **1.1038**, MICE **0.8982**.

Do not claim universal superiority, independent validation of biological markers, or independently replicated cohorts on the basis of repeated seeds on the same three datasets.

## Data and code citation

Cite the associated Scientific Reports manuscript **only after publication** and cite the original data creators. The DOI and permanent repository release URL should be filled in once obtained. See `CITATION.cff` and `docs/CODE_AVAILABILITY.tex`. The repository itself is not yet publicly hosted; do not assert that it is already accessible before uploading it.

## License and public release

No license is assumed on behalf of the authors. Before publishing, the code owners should choose and add a software `LICENSE` file; see `docs/LICENSE_DECISION.md`. Third-party data retain their original terms, and data rehosting requires separate authorization if needed.

## Known scope limits

- Three binary-class source datasets; the two-prototype separation projection is not tested for multi-class cases.
- Baseline hyperparameters are documented but not optimized under identical search budgets.
- Simulated MAR/MNAR/MCAR do not demonstrate accuracy under instrument-confirmed natural missingness.
- MTBLS161 and MTBLS404 source feature IDs are anonymized `M1`, `M2`, etc.; no validated chemical mapping is included.
- Phenotype associations and prototype/observed-profile correlations are **not** independent biochemical validation.
- Friedman/Wilcoxon analysis across 18 scenarios is exploratory because six scenarios reuse each of three source datasets.

## Contact

For correspondence about the research artifact, consult the corresponding author information in the manuscript. Please open a GitHub Issue for reproducibility questions after the repository is published.
