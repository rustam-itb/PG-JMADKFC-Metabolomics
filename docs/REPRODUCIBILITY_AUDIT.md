# Reproducibility and numerical audit (2026-11-03)

## Archived results: validated consistency

The repository's `scripts/verify_results.py` completed successfully. It verifies 2,160 main benchmark rows (12 methods × 180 matched settings), 2,160 main component-ablation fit records, 630 PG-control missingness sensitivity fit records, 180 imputation scenario files (six completed-entry methods including PG), 30 perturbation output files, and SHA-256 checksums of the additional-experiment summary CSVs against the saved manifest. Its checks do **not** independently rerun model training.

## Source and data checks

The source files in `src/` match the source SHA-256 entries of the original frozen protocol. The three analytical arrays were independently prepared from the study-provided `.xlsx`/`.csv` tables using `scripts/prepare_three_datasets.py`, and all five NPZ content arrays (`X_log`, `y`, `features`, `source_rows`, and `source_sha256`) matched the archived prepared arrays exactly. The regenerated NPZ ZIP bytes may differ.

## Fresh training replay: one unexplained discrepancy

A new full-epoch PG-JMADKFC run for `MTBLS161`, MAR, 10% masking, seed 1061 yielded **ARI = 0.19562570295833567**, NMI = 0.14634809661855527, ACC = 0.7288135593220338. The archived benchmark control for that *same mask SHA-256* reports **ARI = 0.16513998458283446**, NMI = 0.12264649465777846, ACC = 0.711864406779661. The mask SHA-256 is `54052ff2926652b9b511aef4b2ad8e22b413a73cab8ab80370c1c11a8c88b399` in both. This discrepancy was reproduced by the original `src/run_one.py` and by the added single-fit wrapper, and it was also observed in a fresh `scripts/run_ablation.py` PG-control replay. Both use the exact same archived source hashes.

For comparison, one fresh DEC run and one fresh FCM-Median run at this setting matched their archived ARI values to floating-point precision. The source of the PG discrepancy is **not yet established**; different CPU math-library implementations, floating-point behavior, and nondeterministic optimization paths are potential explanations but have **not** been proven. The archived run documents Python 3.13.5, PyTorch 2.10.0+cpu and NumPy 2.3.5; these version strings alone do not prove equivalent hardware/backend conditions.

**Therefore:**

- Do **not** describe the code package as establishing exact independent replay of the archived main benchmark.
- The numerical results in the manuscript should be traced to the archived per-fit records and corresponding summaries, which pass static-integrity checks.
- A fresh fit is executable but exact match of metric values across environments is **not yet guaranteed**.
- Before making a definitive reproducibility claim in the submission, identify the cause of this deviation; document the original CPU/BLAS settings, compare deterministic algorithm settings and latent trajectories, and rerun a representative group of seeds on an environment matching the original execution.

The experiment authors previously reported a complete 180/180 replay audit in the additional-experiments manifest. This note documents the outcome of the **new independent attempt during this packaging session**, which differs from that earlier reported replay. Both facts must remain visible until reconciled.
