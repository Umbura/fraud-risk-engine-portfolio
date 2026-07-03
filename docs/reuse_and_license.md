# Reuse And License Notes

This repository is licensed under the MIT License.

The implementation is original. Public repositories and datasets were used as references for project direction, benchmark selection, and documentation structure. External source code was not copied into this project.

## Dataset Use

The synthetic dataset is generated locally by this repository.

The real benchmark uses the public OpenML/Kaggle ULB credit-card fraud dataset:

- https://www.openml.org/d/42175
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Generated local artifacts are ignored by Git:

- `data/*.csv`
- `data/*.parquet`
- `models/`
- `reports/*.json`
- `reports/*.csv`
- `reports/*.txt`
- `reports/*.log`

## Reference Projects

Reference projects are listed in the README for context. They informed the project direction but were not used as copied code sources.

## Publication Checklist

Before publishing:

- run the test suite;
- run Ruff;
- verify that generated datasets and model artifacts are not staged;
- confirm that no `.env` file is present in Git;
- confirm that the CI badge points to the final GitHub repository path.
