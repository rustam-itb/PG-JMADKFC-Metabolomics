# Release checklist (complete before submission)

- [ ] Approve public source-code publication with coauthors.
- [ ] Choose and add a software license.
- [ ] Review rights before making any NPZ, XLSX, CSV raw data, or derived patient-level data public.
- [ ] Confirm citations and credits for the three original datasets.
- [ ] Run `python scripts/verify_results.py` and save its report.
- [ ] Run `python scripts/print_environment.py` on the release machine; save version manifest.
- [ ] Run at least one fresh full-epoch PG fit using the exact prepared source tables.
- [ ] Check source tables and processed arrays agree with the declared source hashes.
- [ ] Create GitHub repository, push files, and confirm visibility in an incognito browser.
- [ ] Create tag `v1.0.0`; optionally archive the release at Zenodo for a DOI.
- [ ] Replace `USERNAME/REPOSITORY` in `docs/CODE_AVAILABILITY.tex`.
- [ ] Update Data Availability and ethics-related declarations to include Human Cachexia.
- [ ] Confirm README wording matches final manuscript title/status and results.
