# Licensing checklist before making the repository public

This package does not impose a software license on its authors. Before release:

1. Obtain approval from all code contributors/coauthors to publish the research source.
2. Decide the intended software license (MIT / Apache-2.0 / BSD-3-Clause, or a different permitted license), then place its complete text in a root `LICENSE` file.
3. Check whether any imported/reimplemented source code requires attribution or license notices. Cite corresponding algorithm papers in the manuscript and source comments.
4. Check data source rights. Prepared NPZ files contain participant-level metabolite measurements and post-fit reference labels; do not automatically push them to a public repository.
5. Decide if archived fit-level results and model outputs need additional dataset-specific attribution or redistribution restrictions.
6. Remove personal paths, credentials, or institution-specific secrets before pushing (source files currently do not require secret API keys).

Without a `LICENSE` file, public visibility does not automatically grant downstream redistribution or modification rights.
