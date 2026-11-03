# Input data and redistribution

This GitHub-ready archive intentionally **does not redistribute sample-level metabolomics data**, even when source studies are publicly accessible. Researchers should download the primary sources and prepare the exact analytical matrices used in the study. Before redistributing any processed sample-level arrays, verify the source datasets' licenses, citation obligations, and consent constraints.

| Analytical dataset | Required source table (study-prepared) | Primary source | n × d |
|---|---|---|---|
| MTBLS161 | `MTBLS161.xlsx` | https://www.ebi.ac.uk/metabolights/MTBLS161 | 59 × 29 |
| MTBLS404 | `MTBLS404.xlsx` | https://www.ebi.ac.uk/metabolights/MTBLS404 | 184 × 120 |
| Human Cachexia | `human_cachexia.csv` | MetaboAnalyst 6.0 sample dataset (see manuscript reference Pang et al. 2024) | 77 × 63 |

**Important:** The two `.xlsx` inputs used in the benchmark are study-prepared analytical tables (with `Data` sheet, `Class` field and `M1...` metabolite columns); they are not necessarily files available verbatim on the corresponding MetaboLights landing pages. Reproducing the exact published fit metrics from a different conversion of original public records is **not guaranteed**. Obtain the same study-prepared inputs through the data release authorized by the authors. The `M1...` identifiers do not establish chemical identities.

## Prepare the analytical arrays

Put the three study-prepared source tables in `data/sources/`, then run:

```bash
python scripts/prepare_three_datasets.py --source-dir data/sources --output-dir data
```

This creates `data/MTBLS161_prepared.npz`, `data/MTBLS404_prepared.npz`, and `data/human_cachexia_prepared.npz`. `X_log` holds the parameter-free `log1p` intensities; reference labels in `y` are used for post-fit metrics only and must not be supplied to model optimization. The script logs source SHA-256 hashes for provenance.

For local auditing, the authors can keep the full data-containing reproduction bundle privately. It is **not part of the public GitHub archive** unless redistribution rights are verified.
