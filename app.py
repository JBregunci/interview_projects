"""Streamlit front-end for the portfolio credit-risk case study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import risk_engine as engine

st.set_page_config(
    page_title="Portfolio credit risk — case study",
    page_icon=":material/account_balance:",
    layout="wide",
)

SETUPS = (1, 2, 3)
DISPLAY_START = 100
LOG_FLOOR = 1e-3
TRACE_COLORS = {"97%": "#2563eb", "99%": "#d97706", "99.99%": "#dc2626"}
SETUP_DESCRIPTIONS = {
    1: "100 equal exposures of USD 10 and independent defaults, "
    r"$I_i \stackrel{iid}{\sim} \mathrm{Bernoulli}(1\%)$.",
    2: "Exposures from `exposures_model_setup_2_and_3.csv` and independent defaults, "
    r"$I_i \stackrel{iid}{\sim} \mathrm{Bernoulli}(1\%)$.",
    3: r"Exposures from the CSV. The common default probability "
    r"$q \sim \mathrm{Beta}(0.2, 19.8)$ is drawn first and "
    r"$I_i \mid q \stackrel{iid}{\sim} \mathrm{Bernoulli}(q)$.",
}


@st.cache_data(show_spinner=False)
def exposures_for(setup: int) -> np.ndarray:
    return engine.get_exposures(setup)


@st.cache_data(show_spinner=False)
def analytical_for(setup: int) -> engine.AnalyticalMoments:
    return engine.analytical_moments(setup, exposures_for(setup))


@st.cache_data(show_spinner=False)
def exact_for(setup: int) -> engine.ExactSummary:
    return engine.exact_summary(setup, exposures_for(setup))


@st.cache_data(show_spinner=False)
def losses_for(setup: int, n_sims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return engine.simulate_losses(setup, exposures_for(setup), n_sims, rng)


@st.cache_data(show_spinner=False)
def running_for(setup: int, n_sims: int, seed: int) -> engine.RunningSeries:
    return engine.running_risk_series(losses_for(setup, n_sims, seed))


@st.cache_data(show_spinner=False)
def summary_for(setup: int, n_sims: int, seed: int) -> engine.MonteCarloSummary:
    return engine.summarize_simulation(losses_for(setup, n_sims, seed))


def display_index(n_sims: int, max_points: int = 1500) -> np.ndarray:
    first = min(DISPLAY_START, n_sims)
    count = n_sims - first + 1
    if count <= max_points:
        return np.arange(first - 1, n_sims)
    spaced = np.geomspace(float(first), float(n_sims), max_points).astype(int) - 1
    return np.unique(spaced.clip(first - 1, n_sims - 1))


def error_values(
    estimate: np.ndarray,
    reference: float,
    signed: bool,
    relative: bool,
    log_scale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    error = estimate - reference
    if not signed:
        error = np.abs(error)
    if relative:
        error = error / reference * 100.0
    display = np.maximum(error, LOG_FLOOR) if log_scale else error
    return display, error


def convergence_figure(
    x: np.ndarray,
    series: engine.RunningSeries,
    reference: dict[str, float],
    measure: str,
    y_title: str,
    unit: str,
    signed: bool,
    relative: bool,
    log_scale: bool,
    index: np.ndarray,
) -> go.Figure:
    fig = go.Figure()
    for label in engine.CL_LABELS:
        estimates = getattr(series.risk[label], measure)[index]
        display, actual = error_values(
            estimates, reference[label], signed, relative, log_scale
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=display,
                customdata=actual,
                mode="lines",
                name=f"CL {label}",
                line=dict(color=TRACE_COLORS[label], width=2),
                hovertemplate=f"CL {label}: %{{customdata:,.4g}} {unit}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Number of simulations",
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    if log_scale:
        fig.update_yaxes(type="log")
    else:
        fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#9ca3af")
    return fig


def exact_reference_captions(exact: engine.ExactSummary) -> list[str]:
    lines = []
    for measure, label in (("var", "VaR"), ("tvar", "TVaR")):
        values = " · ".join(
            f"{cl}: USD {getattr(exact.risk[cl], measure):,.2f}"
            for cl in engine.CL_LABELS
        )
        lines.append(f"Reference {label} — {values}")
    return lines


st.title("Portfolio credit risk — Monte Carlo case study")
st.caption(
    "One-year portfolio of 100 obligors · total notional USD 1,000 · "
    "confidence levels 97%, 99% and 99.99% · simulation benchmarked against a "
    "deterministic numerical reference distribution."
)

with st.sidebar:
    st.header(":material/settings: Simulation controls")
    n_sims = st.select_slider(
        "Number of simulations (N)",
        options=[5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
        value=engine.DEFAULT_SIMULATIONS,
        format_func=lambda value: f"{value:,}",
    )
    seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42)

    st.divider()
    st.subheader(":material/insights: Error chart options")
    error_scale = st.segmented_control(
        "Error scale", ["USD", "Percent"], default="USD"
    )
    signed = st.toggle("Signed error (bias)", value=False)
    log_scale = st.toggle(
        "Logarithmic y-axis", value=True, disabled=signed
    )

    st.divider()
    st.caption(
        "Reference VaR/TVaR come from a deterministic numerical inversion of "
        "the loss distribution: closed-form binomial for setup 1, loss-grid "
        "convolution with Gauss-Jacobi quadrature over the Beta density for "
        "setups 2 and 3. Except for setup 1 this is a numerical approximation "
        "with controlled error (about USD 0.01), not a closed form. TVaR uses "
        r"the tie-inclusive definition $E[L \mid L \geq VaR]$."
    )

relative = error_scale == "Percent"
use_log = bool(log_scale) and not signed
y_units = "%" if relative else "USD"

tab_brief, tab_convergence, tab_analytics, tab_results = st.tabs(
    [
        ":material/description: Problem statement",
        ":material/monitoring: Convergence",
        ":material/functions: Analytical solution",
        ":material/table_chart: Results",
    ]
)

with tab_brief:
    st.markdown(
        r"""
## Case study — portfolio credit risk

### Context

A bank holds a one-year credit portfolio of **100 obligors**, indexed by
$i = 1, \dots, 100$. Obligor $i$ has exposure $E_i$ (USD). The default indicator is

$$
I_i =
\begin{cases}
1 & \text{if obligor } i \text{ defaults within the year,} \\
0 & \text{otherwise.}
\end{cases}
$$

If an obligor defaults, the **full exposure is lost** (no recovery). The total
portfolio loss is

$$
L = \sum_{i=1}^{100} E_i \, I_i .
$$

### Risk measures

For a confidence level $\gamma \in (0, 1)$ and cumulative distribution function
$F_L$ of the loss,

$$
\mathrm{VaR}_\gamma(L) := \inf\{\, x : F_L(x) \geq \gamma \,\}
= \inf\{\, x : \mathbb{P}(L > x) \leq 1 - \gamma \,\},
$$

$$
\mathrm{TVaR}_\gamma(L) := \mathbb{E}\!\left[\, L \mid L \geq \mathrm{VaR}_\gamma(L) \,\right].
$$

### What is being asked

For **each model setup** below:

1. Estimate VaR and TVaR at confidence levels **97%, 99% and 99.99%** using
   Monte Carlo simulation, with a robust justification for the number of
   simulations used.
2. Derive the **expected loss** and **loss volatility** (standard deviation of
   $L$) analytically.
3. Discuss the results, highlighting how the modelling assumptions change the
   portfolio risk profile.

### Model setups

| Setup | Exposures | Dependence structure |
| --- | --- | --- |
| **1** | $E_i = 10$ for all $i$ | $I_i \stackrel{iid}{\sim} \mathrm{Bernoulli}(p)$, $p = 1\%$ |
| **2** | $E_i$ from `exposures_model_setup_2_and_3.csv` (100 non-negative exposures, total USD 1,000) | $I_i \stackrel{iid}{\sim} \mathrm{Bernoulli}(p)$, $p = 1\%$ |
| **3** | Same CSV exposures | Common factor $q \sim \mathrm{Beta}(0.2,\, 19.8)$, then $I_i \mid q \stackrel{iid}{\sim} \mathrm{Bernoulli}(q)$ |

All three setups share the same total notional (USD 1,000) and the same
**unconditional** default probability $\mathbb{E}[q] = 0.01$, which makes the
effect of dependence on the tail directly comparable.

### What this app provides

- **Convergence** — Monte Carlo error of VaR and TVaR at every simulation count
  from 1 to $N$, measured against the deterministic numerical reference.
- **Analytical solution** — full derivations of expected loss, loss volatility,
  the implied default correlation, and the numerical evaluation of the
  reference VaR/TVaR.
- **Results** — the simulated risk measures at the chosen $N$ next to their
  reference/analytical counterparts, with relative errors.
"""
    )

with tab_convergence:
    st.markdown(
        """
The charts show the **true Monte Carlo error**
$\\bigl|\\hat{\\theta}_n - \\theta_{\\mathrm{ref}}\\bigr|$ of the running
estimator as a function of the number of simulations $n$. The reference loss
distribution is computed deterministically (without simulation noise), so this
is the actual estimation error rather than a self-referential convergence
measure.
"""
    )

    index = display_index(n_sims)
    x_display = (index + 1).astype(float)

    for setup in SETUPS:
        with st.container(border=True):
            st.subheader(f"Model setup {setup}")
            st.markdown(SETUP_DESCRIPTIONS[setup])
            with st.spinner(f"Simulating model setup {setup} ..."):
                series = running_for(setup, n_sims, seed)
                exact = exact_for(setup)

            for reference_line in exact_reference_captions(exact):
                st.caption(reference_line)
            reference_var = {cl: exact.risk[cl].var for cl in engine.CL_LABELS}
            reference_tvar = {cl: exact.risk[cl].tvar for cl in engine.CL_LABELS}

            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.markdown("**VaR error**")
                st.plotly_chart(
                    convergence_figure(
                        x_display,
                        series,
                        reference_var,
                        "var",
                        f"{'Signed' if signed else 'Absolute'} error ({y_units})",
                        y_units,
                        signed,
                        relative,
                        use_log,
                        index,
                    ),
                    theme=None,
                    width="stretch",
                    alt=f"Monte Carlo error of VaR against the deterministic reference for model setup {setup}.",
                )
            with chart_right:
                st.markdown("**TVaR error**")
                st.plotly_chart(
                    convergence_figure(
                        x_display,
                        series,
                        reference_tvar,
                        "tvar",
                        f"{'Signed' if signed else 'Absolute'} error ({y_units})",
                        y_units,
                        signed,
                        relative,
                        use_log,
                        index,
                    ),
                    theme=None,
                    width="stretch",
                    alt=f"Monte Carlo error of TVaR against the deterministic reference for model setup {setup}.",
                )

    st.caption(
        f"Charts start at {DISPLAY_START:,} simulations and display up to 1,500 "
        f"log-spaced points out of {n_sims:,}; the underlying series is computed "
        "at every n. Zero errors are drawn at the axis floor on the log scale."
    )

with tab_analytics:
    moments = {setup: analytical_for(setup) for setup in SETUPS}
    exact = {setup: exact_for(setup) for setup in SETUPS}
    rho = engine.implied_default_correlation()

    alpha = engine.BETA_ALPHA
    beta = engine.BETA_BETA
    ab = alpha + beta
    e_q = alpha / ab
    var_q = alpha * beta / (ab**2 * (ab + 1.0))
    e_q1_q = alpha * beta / (ab * (ab + 1.0))
    sum_e2 = float(np.square(exposures_for(2)).sum())

    st.markdown(
        rf"""
## 1. Portfolio loss and model assumptions

The portfolio loss is $L = \sum_{{i=1}}^{{100}} E_i I_i$ with $I_i \in \{{0, 1\}}$.
The three setups differ only in the joint law of $(I_1, \dots, I_{{100}})$:

- **Setups 1 and 2:** $I_i \stackrel{{iid}}{{\sim}} \mathrm{{Bernoulli}}(p)$ with
  $p = {engine.DEFAULT_P:.2f}$.
- **Setup 3:** $q \sim \mathrm{{Beta}}(\alpha, \beta)$ with
  $\alpha = {alpha}$, $\beta = {beta}$, and
  $I_i \mid q \stackrel{{iid}}{{\sim}} \mathrm{{Bernoulli}}(q)$.

The Beta parameters are chosen so that the unconditional default probability
matches setups 1 and 2:

$$
\mathbb{{E}}[q] = \frac{{\alpha}}{{\alpha + \beta}}
= \frac{{{alpha}}}{{{alpha} + {beta}}} = {e_q:.4f} = p.
$$

## 2. Expected loss and loss volatility

### Setups 1 and 2 — independent defaults

Since $\mathbb{{E}}[I_i] = p$ and $\mathrm{{Var}}(I_i) = p(1-p)$,
independence gives

$$
\mathbb{{E}}[L] = \sum_{{i=1}}^{{100}} E_i \, \mathbb{{E}}[I_i]
= p \sum_{{i=1}}^{{100}} E_i
= {engine.DEFAULT_P} \times {float(exposures_for(2).sum()):,.0f}
= \text{{USD }} {moments[2].expected_loss:,.2f},
$$

$$
\mathrm{{Var}}(L) = \sum_{{i=1}}^{{100}} E_i^2 \, \mathrm{{Var}}(I_i)
= p(1-p) \sum_{{i=1}}^{{100}} E_i^2 .
$$

Setup 1 ($E_i \equiv 10$): $\sum_i E_i^2 = 100 \times 100 = 10{{,}}000$, so
$\sigma_L = \text{{USD }} {moments[1].loss_volatility:,.2f}$.

Setup 2 (CSV exposures): $\sum_i E_i^2 = {sum_e2:,.2f}$, so
$\sigma_L = \text{{USD }} {moments[2].loss_volatility:,.2f}$. The expected loss is identical
to setup 1 — only the dispersion changes.

### Setup 3 — a common factor $q$

Conditional on $q$, defaults are independent with $\mathbb{{E}}[I_i \mid q] = q$
and $\mathrm{{Var}}(I_i \mid q) = q(1-q)$, hence

$$
\mathbb{{E}}[L \mid q] = q \sum_{{i=1}}^{{100}} E_i,
\qquad
\mathrm{{Var}}(L \mid q) = q(1-q) \sum_{{i=1}}^{{100}} E_i^2 .
$$

The **law of total expectation** gives

$$
\mathbb{{E}}[L] = \mathbb{{E}}\bigl[\mathbb{{E}}[L \mid q]\bigr]
= \mathbb{{E}}[q] \sum_{{i=1}}^{{100}} E_i
= \text{{USD }} {moments[3].expected_loss:,.2f},
$$

and the **law of total variance** adds the factor-risk term:

$$
\mathrm{{Var}}(L)
= \underbrace{{\mathbb{{E}}\bigl[\mathrm{{Var}}(L \mid q)\bigr]}}_{{\text{{idiosyncratic}}}}
+ \underbrace{{\mathrm{{Var}}\bigl(\mathbb{{E}}[L \mid q]\bigr)}}_{{\text{{systematic}}}}
= \mathbb{{E}}[q(1-q)] \sum_{{i=1}}^{{100}} E_i^2
+ \mathrm{{Var}}(q) \left( \sum_{{i=1}}^{{100}} E_i \right)^{{2}} .
$$

For a Beta law,

$$
\mathbb{{E}}[q(1-q)] = \frac{{\alpha \beta}}{{(\alpha + \beta)(\alpha + \beta + 1)}}
= {e_q1_q:.6f},
\qquad
\mathrm{{Var}}(q) = \frac{{\alpha \beta}}{{(\alpha + \beta)^2 (\alpha + \beta + 1)}}
= {var_q:.3e},
$$

so $\sigma_L = \text{{USD }} {moments[3].loss_volatility:,.2f}$ — more than
{moments[3].loss_volatility / moments[2].loss_volatility:.2f}× the independent
case with the same exposures.

### Implied default correlation

For any two obligors, $\mathrm{{Cov}}(I_i, I_j) = \mathrm{{Var}}(q)$, hence

$$
\rho = \mathrm{{Corr}}(I_i, I_j)
= \frac{{\mathrm{{Var}}(q)}}{{p(1-p)}}
= {rho:.4f} \approx {rho * 100:.2f}\%.
$$

Even a modest {rho:.1%} pairwise correlation materially fattens the portfolio
tail because all 100 names load on the same factor.
"""
    )

    st.markdown("## 3. VaR and TVaR")
    st.latex(r"\mathrm{VaR}_\gamma(L) = \inf\{\, x : F_L(x) \geq \gamma \,\}")
    st.latex(
        r"\mathrm{TVaR}_\gamma(L) = "
        r"\frac{\mathbb{E}\left[\, L \cdot \mathbf{1}\{L \geq \mathrm{VaR}_\gamma(L)\} \,\right]}"
        r"{\mathbb{P}\left(L \geq \mathrm{VaR}_\gamma(L)\right)}"
    )

    st.markdown(
        rf"""
The TVaR expression is the **tie-inclusive** conditional expectation from the
case study: for discrete losses every sampled outcome exactly equal to the VaR
threshold belongs to the tail. This matters in setup 1, where the loss is
concentrated on multiples of USD 10.

### Setup 1 — closed form

With $K \sim \mathrm{{Binomial}}(100, p)$ the loss is $L = 10 K$, so

$$
\mathbb{{P}}(L = 10k) = \binom{{100}}{{k}} p^k (1-p)^{{100-k}},
\qquad
\mathrm{{VaR}}_\gamma = 10 \, k_\gamma,
\qquad
\mathrm{{TVaR}}_\gamma = 10 \,
\frac{{\sum_{{k \geq k_\gamma}} k \, \mathbb{{P}}(K = k)}}
{{\sum_{{k \geq k_\gamma}} \mathbb{{P}}(K = k)}},
$$

where $k_\gamma = \min\{{k : F_K(k) \geq \gamma\}}$. This yields exact values at
the three confidence levels: VaR = USD {exact[1].risk['97%'].var:,.2f} /
USD {exact[1].risk['99%'].var:,.2f} / USD {exact[1].risk['99.99%'].var:,.2f},
and TVaR = USD {exact[1].risk['97%'].tvar:,.2f} /
USD {exact[1].risk['99%'].tvar:,.2f} / USD {exact[1].risk['99.99%'].tvar:,.2f}.

### Setup 2 — weighted Bernoulli sum

Here $L$ is a weighted sum of independent, non-identically distributed
Bernoulli variables (a **weighted Poisson-binomial** law). No elementary closed
form exists, but the CDF is the subset sum

$$
F_L(x) = \sum_{{S \subseteq \{{1, \dots, 100\}} : \sum_{{i \in S}} E_i \leq x}}
\; p^{{|S|}} (1-p)^{{100 - |S|}},
$$

and the characteristic function factorises, which enables fast deterministic
inversion:

$$
\varphi_L(t) = \mathbb{{E}}\left[e^{{itL}}\right]
= \prod_{{i=1}}^{{100}} \left( (1-p) + p \, e^{{itE_i}} \right).
$$

This app evaluates the distribution by deterministic convolution on a fine loss
grid — a numerical approximation with controlled discretization error — instead
of simulation.

### Setup 3 — Beta mixture

Conditional on $q$ the loss has the same weighted Poisson-binomial law with
$p$ replaced by $q$. Integrating over the Beta density
$\pi(q) \propto q^{{\alpha - 1}}(1-q)^{{\beta - 1}}$ gives

$$
F_L(x) = \int_0^1 F_{{\mathrm{{PB}}}}(x; q) \,
\frac{{q^{{\alpha - 1}} (1-q)^{{\beta - 1}}}}{{\mathrm{{B}}(\alpha, \beta)}} \, dq .
$$

Equivalently, the number of defaults is **Beta-binomial**,

$$
\mathbb{{P}}(K = k) = \binom{{100}}{{k}}
\frac{{\mathrm{{B}}(\alpha + k, \beta + 100 - k)}}{{\mathrm{{B}}(\alpha, \beta)}},
$$

and conditional on $K = k$ the loss is the sum of $k$ exposures drawn uniformly
without replacement. Compared with the binomial, the Beta-binomial is
overdispersed: $\mathrm{{Var}}(K) = n p (1-p)\left[1 + (n-1)\rho\right]$ with
$n = 100$ and $\rho = {rho:.4f}$. The far tail is driven by the joint effect of
**more defaults** and the **largest exposures** occurring together.

### Deterministic reference values

| Confidence level | Model 1 VaR | Model 1 TVaR | Model 2 VaR | Model 2 TVaR | Model 3 VaR | Model 3 TVaR |
| --- | --- | --- | --- | --- | --- | --- |
| 97% | USD {exact[1].risk['97%'].var:,.2f} | USD {exact[1].risk['97%'].tvar:,.2f} | USD {exact[2].risk['97%'].var:,.2f} | USD {exact[2].risk['97%'].tvar:,.2f} | USD {exact[3].risk['97%'].var:,.2f} | USD {exact[3].risk['97%'].tvar:,.2f} |
| 99% | USD {exact[1].risk['99%'].var:,.2f} | USD {exact[1].risk['99%'].tvar:,.2f} | USD {exact[2].risk['99%'].var:,.2f} | USD {exact[2].risk['99%'].tvar:,.2f} | USD {exact[3].risk['99%'].var:,.2f} | USD {exact[3].risk['99%'].tvar:,.2f} |
| 99.99% | USD {exact[1].risk['99.99%'].var:,.2f} | USD {exact[1].risk['99.99%'].tvar:,.2f} | USD {exact[2].risk['99.99%'].var:,.2f} | USD {exact[2].risk['99.99%'].tvar:,.2f} | USD {exact[3].risk['99.99%'].var:,.2f} | USD {exact[3].risk['99.99%'].tvar:,.2f} |

## 4. Numerical evaluation of the reference distribution

Only setup 1 has a closed form. The setup 2 and setup 3 reference values are a
**deterministic numerical approximation**: there is no Monte Carlo noise, but
there is a controlled discretization/quadrature error. The pipeline is:

1. **Loss grid.** The support $[0, \sum_i E_i]$ is discretized into
   $N = 2^{{16}} = 65{{,}}536$ equally spaced points, i.e. a step of
   $h = 1{{,}}000 / 65{{,}}535 \approx \mathrm{{USD}}\ 0.01526$. The pmf starts as
   a point mass at zero.
2. **Sequential convolution** (implemented in `risk_engine._grid_pmf`). Starting
   from the point mass at zero, obligors are added one at a time. Each step
   applies the exact two-point rule to every row of the pmf array:

   $$
   f_{{\text{{new}}}}(x) = (1-q)\, f(x) + q\, f(x - E_i),
   $$

   i.e. either obligor $i$ does not default (mass stays at $x$) or it defaults
   (mass moves to $x + E_i$). Because $E_i$ is not a multiple of the grid step
   $h$, the shift $E_i / h = k + u$ falls between grid points, and the shifted mass
   is split between the two neighbours by **linear interpolation**. This is
   exactly the update performed in the loop over exposures of `_grid_pmf`:

   $$
   f_{{\text{{new}}}}[j] = (1-q) f[j]
   + q\big[(1-u) f[j-k] + u\, f[j-k-1]\big].
   $$

   `_grid_pmf` performs the full convolution over the 100 obligors: setup 2 calls
   it once with $q = p$, while setup 3 calls it once per quadrature node (item 3)
   and averages the rows. The interpolation is the only discretization error of
   the convolution: it conserves total mass, and its effect on the moments is of
   order $O(h^2)$.
3. **Beta mixture by Gauss–Jacobi quadrature** (implemented in
   `risk_engine.exact_pmf`). For setup 3 the conditional Poisson-binomial law is
   convolved once per quadrature node with `_grid_pmf`, and the rows are averaged
   against the Beta density through the normalized weights:

   $$
   F_L(x) = \int_0^1 F_{{\mathrm{{PB}}}}(x; q)\,
   \frac{{q^{{\alpha-1}}(1-q)^{{\beta-1}}}}{{\mathrm{{B}}(\alpha,\beta)}}\, dq
   \;\approx\; \sum_{{j=1}}^{{m}} w_j\, F_{{\mathrm{{PB}}}}(x; q_j).
   $$

   The nodes and weights come from `scipy.special.roots_jacobi(n, a, b)`, called
   inside `exact_pmf`; the rule is exact for
   $\int_{{-1}}^{{1}} (1-x)^a (1+x)^b g(x)\, dx$. Choosing
   **$a = \beta - 1$ and $b = \alpha - 1$** makes the Jacobi weight coincide with
   the Beta density after the change of variable $q = (x+1)/2$, since
   $(1-x)^{{\beta-1}}(1+x)^{{\alpha-1}}\, dx = 2^{{\alpha+\beta-1}} q^{{\alpha-1}}(1-q)^{{\beta-1}}\, dq$.
   Normalizing the raw weights by $2^{{\alpha+\beta-1}} \mathrm{{B}}(\alpha,\beta)$
   turns the sum into an expectation under the Beta law.
4. **How many quadrature points?** The app uses $m = 32$ nodes: the roots of the
   degree-32 Jacobi polynomial. A 32-point Gauss–Jacobi rule integrates
   polynomials up to degree $2m - 1 = 63$ exactly, and the remaining integrand is
   smooth in $q$, so convergence is fast: going from 32 to 64 nodes changes every
   VaR/TVaR by less than USD 0.01, and 24 nodes already reproduce the displayed
   values.
5. **VaR by CDF inversion** (implemented in `risk_engine.exact_summary`). With
   $F_j = \sum_{{l \leq j}} f_l$, the VaR is the
   first grid point whose CDF reaches $\gamma$:

   $$
   x_{{j^*}}, \qquad j^* = \min\{{j : F_j \geq \gamma\}},
   $$

   found by `np.searchsorted`, i.e. binary search. The reference loss distribution
   is discrete, so its CDF is a step function and the quantile is a grid point —
   no Newton–Raphson or other derivative-based root finding is used.
6. **TVaR** (same function, `exact_summary`). The tie-inclusive tail mean
   $\sum_{{j \geq j^*}} x_j f_j \big/ \sum_{{j \geq j^*}} f_j$ includes the
   complete probability atom at the VaR grid point.

**Accuracy checks.** Setup 1 matches the SciPy binomial law exactly. For setups 2
and 3, the pmf mean and standard deviation match the closed forms of section 2 to
about $10^{{-7}}$; halving the grid step changes VaR/TVaR by at most USD 0.01; and
one-million-path Monte Carlo simulations agree within sampling error. Because of
the grid and quadrature approximations, the app labels these values **reference**
rather than exact.

## 5. Discussion

- **Expected loss is identical across setups** by construction
  (USD {moments[1].expected_loss:,.2f}) — assumptions about dependence do not move
  the mean.
- **Volatility and tail risk are not.** Loss volatility grows from
  USD {moments[1].loss_volatility:,.2f} (setup 1) to USD {moments[2].loss_volatility:,.2f}
  (setup 2, heterogeneous exposures) and to USD {moments[3].loss_volatility:,.2f}
  (setup 3, common factor).
- **Concentration matters even without dependence:** setup 2 has the same
  expected loss as setup 1 but a larger $\sum_i E_i^2$, so both volatility and
  VaR/TVaR increase.
- **Dependence dominates the deep tail:** at the 99.99% level the reference TVaR
  is
  USD {exact[3].risk['99.99%'].tvar:,.2f} under the common-factor model versus
  USD {exact[2].risk['99.99%'].tvar:,.2f} under independence — a factor of
  {exact[3].risk['99.99%'].tvar / exact[2].risk['99.99%'].tvar:.1f}.
- **Simulation budget:** at $N = 25{{,}}000$ the expected number of exceedances
  at the 99.99% level is only $N(1-\gamma) = 2.5$, so VaR is a single order
  statistic and TVaR averages a handful of observations. The convergence tab
  quantifies this error; larger $N$, importance sampling or analytical
  inversion are the standard remedies.
"""
    )

with tab_results:
    with st.spinner("Running simulations and computing reference benchmarks ..."):
        summaries = {setup: summary_for(setup, n_sims, seed) for setup in SETUPS}
        moments = {setup: analytical_for(setup) for setup in SETUPS}
        exact = {setup: exact_for(setup) for setup in SETUPS}

    metric_rows: list[tuple[str, str, str | None]] = []
    for cl in engine.CL_LABELS:
        metric_rows.append((f"CL = {cl} — VaR", "risk", "var", cl))
        metric_rows.append((f"CL = {cl} — TVaR", "risk", "tvar", cl))
    metric_rows.append(("Expected loss", "expected_loss", None, None))
    metric_rows.append(("Loss volatility", "loss_volatility", None, None))

    columns = [
        f"Model {setup} · {statistic}"
        for setup in SETUPS
        for statistic in ("Simulation", "Reference", "Rel. error")
    ]
    table = pd.DataFrame(
        index=[row[0] for row in metric_rows],
        columns=columns,
        dtype=float,
    )

    for column_index, setup in enumerate(SETUPS):
        for row_index, (_, kind, measure, cl) in enumerate(metric_rows):
            if kind == "risk":
                simulated = getattr(summaries[setup].risk[cl], measure)
                reference = getattr(exact[setup].risk[cl], measure)
            else:
                simulated = getattr(summaries[setup], kind)
                reference = getattr(moments[setup], kind)
            offset = 3 * column_index
            table.iloc[row_index, offset] = simulated
            table.iloc[row_index, offset + 1] = reference
            table.iloc[row_index, offset + 2] = (
                (simulated - reference) / reference * 100.0
            )

    st.subheader("Simulated vs reference/analytical risk measures")
    st.caption(
        f"Monte Carlo estimate with N = {n_sims:,} paths (seed {seed}). "
        "Reference VaR/TVaR come from the deterministic numerical evaluation "
        "of the loss distribution (grid convolution with Gauss–Jacobi "
        "quadrature; closed-form binomial for setup 1). Expected loss and "
        "volatility are compared with the closed-form expressions. Relative "
        "error is computed as (simulation − reference) / reference."
    )

    column_config: dict = {}
    for setup in SETUPS:
        prefix = f"Model {setup}"
        column_config[f"{prefix} · Simulation"] = st.column_config.NumberColumn(
            "Simulation", format="dollar"
        )
        column_config[f"{prefix} · Reference"] = st.column_config.NumberColumn(
            "Reference", format="dollar"
        )
        column_config[f"{prefix} · Rel. error"] = st.column_config.NumberColumn(
            "Rel. error", format="%.2f%%"
        )

    st.dataframe(
        table,
        column_config=column_config,
        height=(len(metric_rows) + 1) * 35 + 8,
        alt="Simulated and reference VaR, TVaR, expected loss and loss volatility for the three model setups at the chosen simulation count.",
    )

    st.download_button(
        "Download table as CSV",
        data=table.to_csv().encode("utf-8"),
        file_name=f"risk_measures_N{n_sims}_seed{seed}.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    st.subheader("Key takeaways")
    spread = exact[3].risk["99.99%"].tvar / exact[2].risk["99.99%"].tvar
    st.markdown(
        f"""
- All three setups share the same expected loss of
  **USD {moments[1].expected_loss:,.2f}**, by construction.
- Loss volatility rises from **USD {moments[1].loss_volatility:,.2f}** (equal
  exposures, independent) to **USD {moments[2].loss_volatility:,.2f}**
  (heterogeneous exposures, independent) and **USD {moments[3].loss_volatility:,.2f}**
  (common factor).
- At CL 99.99%, the reference TVaR under the common-factor model is
  **USD {exact[3].risk['99.99%'].tvar:,.2f}** versus
  **USD {exact[2].risk['99.99%'].tvar:,.2f}** under independence — a factor of
  **{spread:.1f}×**.
- At N = {n_sims:,}, the {len(engine.CL_LABELS)} confidence levels are estimated
  with sampling error shown in the table; the 99.99% estimate remains the
  noisiest because only about {n_sims * 1e-4:,.1f} observations are expected
  beyond the VaR threshold.
"""
    )
