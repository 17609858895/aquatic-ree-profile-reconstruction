# Rare earth element profile reconstruction in aquatic organisms

Core analysis code for estimating 14-element rare earth element (REE) profiles from partial measurements. The scripts prepare the aquatic biota records, fit the hierarchical profile model, compare completion methods across held-out studies, and estimate group-specific distributions of total REE concentrations.

## Data and environment

Download and extract the source workbook from the compilation by Pereto, Baudrimont and Coynel (2024), distributed under CC BY 4.0. [PANGAEA dataset](https://doi.org/10.1594/PANGAEA.963631)

Place `Global_compilation_of_Rare_Earth_Element_PERETO-etal_2023.xlsx` in `data_extracted/` at the repository root. The SHA-256 digest of the extracted workbook used for the analysis is `7934025ede6b637fb98455683c15f66d9a049c48cd9eeb0f4bf13cedd7469816`.

Use a Python 3.12 environment and run the following commands from the repository root. `requirements.txt` records the analysis package versions.

```sh
python -m pip install -r requirements.txt
python scripts/01_prepare.py
```

The preparation script writes `data_processed/biota_model.csv`. The analysis retains 1,430 biota records from 96 source studies. Of these, 556 records contain all 14 elements.

## Analyses

`scripts/lchg.py` implements the model. `scripts/ree_common.py` defines the element order, normalization constants, workbook reader and tissue groupings. The analyses below use the prepared table.

| Analysis | Command | Main output in `results/` |
| --- | --- | --- |
| Metadata benchmark with nested tuning | `python scripts/02c_benchmark20_tuned.py 4` | `benchmark20_tuned_summary.csv` |
| Study-held-out profile completion | `python scripts/03_completion.py 0,1,2` | `completion_predictions_seeds0-1-2.parquet` |
| Tuned completion baselines | `python scripts/03b_completion_tuned_baselines.py 0,1,2 4` | `completion_tuned_predictions_seeds0-1-2.parquet` |
| Expanded baseline search | `python scripts/03b_completion_tuned_baselines.py 0,1,2 4 r2` | `completion_tuned_r2_predictions_seeds0-1-2.parquet` |
| Summary of completion runs | `python scripts/09_summaries.py` | `completion_summary_full.csv` |
| Measurement panels and interval calibration | `python scripts/04_uncertainty_panels.py` | Panel-design and coverage tables |
| Sensitivity analyses and matched panel comparisons | `python scripts/11_sensitivity.py 4` | Sensitivity and matched-panel tables |
| Full-data fit, harmonised totals and organism-group summaries | `python scripts/06_application.py` | `harmonised_biota_REE.csv`, `ext_baseline_shift.csv` |

For the completion comparison, run the two baseline searches and the study-held-out completion script before the summary script. It pools every matching prediction file in `results/`, so each method and seed should occur only once across input files. The argument `4` sets the number of parallel workers in scripts that accept a worker count.

The application script can run directly after data preparation. For each record it preserves the reported element measurements, simulates the missing elements, and computes the median and interval for the 14-element total. Its final group comparison uses dry-weight records classified as uncontaminated in the source compilation.
