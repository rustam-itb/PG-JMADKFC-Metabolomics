# Source origins and exactness

The files in `src/` are copied from the original research package, and the reviewer-requested `scripts/` for ablation, imputation, perturbation and statistical summaries are copied without alteration from the additional-experiment archive. These scripts can be compared directly to the author source hashes.

`src/freeze_protocol.py` and `src/run_one.py` retain the **historical six-dataset lock**, including experimental development classes not separately reported in the revised manuscript. This is deliberate: rewriting those original source files would invalidate the frozen fingerprints used to audit earlier runs. Use `scripts/reproduce_paper_benchmark.py` to reproduce the **reported three-dataset, twelve-method** benchmark. This wrapper uses the same original `run_one.run_fit` fit function, without invoking the older six-dataset lock.

The new wrapper, data preparation script, result verifier, environment reporter and documentation are convenience/release tools created for this code-availability package. Results used in the manuscript should be verified against archived `results/benchmark/` CSVs and `final_analysis/EXPERIMENT_MANIFEST.json`.

Data-only transformations may reproduce identical numerical tensors without producing byte-identical `.npz` compressed files; compare source SHA-256 and array contents when re-preparing inputs.
