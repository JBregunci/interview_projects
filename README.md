# Portfolio credit risk — Monte Carlo case study

An interactive Streamlit app that estimates **Value-at-Risk (VaR)** and **Tail
Value-at-Risk (TVaR)** for a one-year credit portfolio of 100 obligors under three
modelling assumptions, and benchmarks every simulation against a **deterministic
numerical reference distribution**: closed-form for setup 1, loss-grid convolution
with Gauss–Jacobi quadrature over the Beta density for setups 2 and 3, accurate to
about USD 0.01.

The case study requires Monte Carlo estimation at confidence levels 97%, 99% and
99.99%, analytical expected loss and loss volatility, and a discussion of how the
modelling assumptions change portfolio risk.

## Model setups

All setups share the same total notional (USD 1,000) and unconditional default
probability (1%), which makes the effect of dependence directly comparable.

| Setup | Exposures | Dependence |
| --- | --- | --- |
| 1 | Equal exposures of USD 10 | Independent defaults |
| 2 | From `exposures_model_setup_2_and_3.csv` | Independent defaults |
| 3 | Same CSV | Common factor $q \sim \mathrm{Beta}(0.2, 19.8)$, conditionally independent defaults |

## What the app shows

- **Problem statement** — the rewritten case-study brief and model definitions.
- **Convergence** — for each setup, the true Monte Carlo error of running VaR and TVaR
  at every simulation count from 100 to $N$, measured against the deterministic
  reference. Charts can show USD or percent errors, signed or absolute, on linear or
  log axes.
- **Analytical solution** — full derivations of expected loss, loss volatility, the
  implied default correlation, the loss distributions and the numerical evaluation of
  the reference values.
- **Results** — simulated vs reference/analytical VaR, TVaR, expected loss and
  volatility at the chosen $N$, with relative errors and a CSV download.

## Project structure

```
.
├── app.py                            # Streamlit front-end (UI only)
├── risk_engine.py                    # Simulations, analytical moments, reference distribution
├── exposures_model_setup_2_and_3.csv # 100 exposures totalling USD 1,000
├── environment.yml                   # Conda environment
└── .streamlit/config.toml            # Light theme
```

`app.py` contains no calculation logic; everything numerical lives in
`risk_engine.py`, and the full derivations are presented in the app's
**Analytical solution** tab.

## Running the app

```bash
conda env create -f environment.yml   # first time only
conda activate portfolio-credit-risk
streamlit run app.py
```

The app runs entirely locally and needs no data downloads.

## Methods at a glance

- **Monte Carlo** — vectorised sampling of the hierarchical model. Setup 3 draws one
  common default probability $q$ per simulated year, applied to all 100 obligors.
- **Running risk measures** — VaR and TVaR are computed at every prefix of the sample
  with an order-statistic Fenwick tree, so per-simulation convergence curves are
  cheap even at $N = 10^6$.
- **Deterministic reference** — the loss distribution is computed without simulation
  by convolving the 100 two-point Bernoulli laws on a fine loss grid (65,536 points,
  step ≈ USD 0.015). Setup 3 averages the conditional laws over the Beta density with
  32-point Gauss–Jacobi quadrature using $a = \beta - 1$ and $b = \alpha - 1$. VaR is
  a binary search on the resulting CDF; TVaR is the tie-inclusive tail mean
  $\mathbb{E}[L \mid L \geq \mathrm{VaR}]$. Except for the closed-form setup 1, this
  reference is a numerical approximation accurate to about USD 0.01.
- **Analytical moments** — closed forms via linearity of expectation and the law of
  total (expectation and) variance for the Beta-mixture setup.

## Key results (deterministic reference)

| Risk measure | Setup 1 | Setup 2 | Setup 3 |
| --- | --- | --- | --- |
| Expected loss | USD 10.00 | USD 10.00 | USD 10.00 |
| Loss volatility | USD 9.95 | USD 13.46 | USD 25.37 |
| TVaR 99.99% | USD 61.50 | USD 110.61 | USD 354.68 |

The expected loss is identical by construction, but the common factor in setup 3
creates a ~4.8% pairwise default correlation that makes the deep-tail risk roughly
3× larger than under independence. Run `app.py` for the interactive comparison and
the full derivations.
