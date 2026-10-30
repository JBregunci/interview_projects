# Pandas Data-Wrangling Workout

A pandas refresher for interview preparation: **59 hands-on exercises** on a
deliberately messy synthetic e-commerce dataset, written in a functional pandas style.

The workbook covers the full data-wrangling loop, from inspection and cleaning to
aggregation, joins, reshaping, time series, MultiIndexes, performance, and a
pure-function capstone.

## What is inside

- 59 exercises with prompts, starter cells, assertion-based self-checks, and
  collapsible worked solutions.
- Five related CSV files committed under `datasets/`; no downloads are required.
- A functional emphasis: pure helper functions, `.pipe` composition, explicit dtypes,
  `assign`, `query`, `transform`, named aggregation, validated merges, and vectorized
  operations.
- A cumulative capstone that turns raw tables into a customer analytics report.
- Five challenge problems: RFM segmentation, cohort retention, channel funnels,
  `merge_asof`, and deduplication.
- A cheat sheet and code-review checklist at the end.

## Project structure

```text
.
├── README.md
└── pandas_workout/
    ├── pandas_workout.ipynb
    ├── requirements.txt
    └── datasets/
        ├── customers.csv
        ├── orders.csv
        ├── product_prices.csv
        ├── products.csv
        └── returns.csv
```

## Workbook contents

| Section | Focus | Exercises |
| --- | --- | ---: |
| 1 | Foundations and inspection | 4 |
| 2 | Loading and typing | 5 |
| 3 | Cleaning and validation | 8 |
| 4 | Features, strings, and dates | 7 |
| 5 | Aggregation | 7 |
| 6 | Combining and reshaping | 7 |
| 7 | Time series | 5 |
| 8 | Advanced indexing | 3 |
| 9 | Performance and review | 3 |
| 10 | Functional capstone | 5 |
| 11 | Challenge set | 5 |
| 12 | Cheat sheet and self-review | — |

## Dataset

| Table | Grain | Rows |
| --- | --- | ---: |
| `customers.csv` | one row per customer | 215 |
| `products.csv` | one row per product | 30 |
| `product_prices.csv` | price changes by product and effective date | 62 |
| `orders.csv` | order lines, including duplicate updates | 1,770 |
| `returns.csv` | return events, including duplicates | 170 |

The data intentionally includes inconsistent labels, mixed date formats, missing and
orphan foreign keys, duplicate rows, currency-formatted prices, invalid quantities,
missing discounts, and outliers. Each issue is documented and used by the exercises.

## Setup

Create a conda environment and install the pinned packages:

```bash
conda create -n pandas-workout python=3.11 -y
conda activate pandas-workout
pip install -r pandas_workout/requirements.txt
```

Then open `pandas_workout/pandas_workout.ipynb` in your local notebook environment and
run the first **Setup** cell. The setup cell resolves `datasets/` from either the
repository root or the project directory.

`requirements.txt` contains pandas, numpy, JupyterLab, ipykernel, and pyarrow (used by
the Parquet exercise).

## Working through the workbook

Work through the sections in order. Each exercise has a self-check that prints
`passed` when the contract is met. Try the exercise before opening the collapsible
solution; alternative correct implementations are welcome.

All 59 self-checks have been executed successfully.

## Branch layout

This repository uses one branch per interview project:

- `pandas-workout` contains this workbook and its datasets.
- `BIS_Case_Study` contains the portfolio credit-risk case study.
- `main` contains only repository-level files.
