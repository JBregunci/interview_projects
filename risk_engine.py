"""Portfolio credit-risk case study: simulation, exact distributions and analytical moments.

Model setups
------------
1. 100 obligors, equal exposures of USD 10, independent Bernoulli(p = 1%).
2. Exposures from exposures_model_setup_2_and_3.csv, independent Bernoulli(p = 1%).
3. Same exposures, but the common default probability q ~ Beta(0.2, 19.8) is drawn
   first and defaults are independent Bernoulli(q) conditional on q.

Portfolio loss and risk measures
--------------------------------
The portfolio loss is L = sum_i E_i * I_i, where E_i is the exposure to obligor i
and I_i is its default indicator. For a confidence level gamma:

    VaR_gamma(L)  = inf{x : F_L(x) >= gamma}
    TVaR_gamma(L) = E[L | L >= VaR_gamma(L)]

TVaR is evaluated with the tie-inclusive definition of the case study: when the
loss distribution is discrete, every outcome exactly equal to the VaR threshold is
part of the tail. This differs from expected shortfall for atoms in the loss law.

Exact reference
---------------
Rather than relying on simulation, the app benchmarks Monte Carlo estimates against
a deterministic loss distribution:

* setup 1 is the closed-form Binomial(100, p) law of the default count;
* setups 2 and 3 are obtained by sequential convolution of the 100 two-point
  Bernoulli laws on a fine loss grid (linear interpolation handles the fractional
  grid shift);
* setup 3 additionally integrates the conditional distributions over the Beta
  density with Gauss-Jacobi quadrature.

VaR is then obtained by a binary search on the exact CDF (the distribution is
discrete, so the quantile is a grid point, not a root of a smooth equation), and
TVaR is the tail-conditional mean of the resulting probability mass function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import special, stats

SetupId = Literal[1, 2, 3]

# Portfolio and model constants fixed by the case study.
N_OBLIGORS = 100
PORTFOLIO_NOTIONAL = 1_000.0
EQUAL_EXPOSURE = PORTFOLIO_NOTIONAL / N_OBLIGORS
DEFAULT_P = 0.01
BETA_ALPHA = 0.2
BETA_BETA = 19.8
DEFAULT_CSV = Path(__file__).with_name("exposures_model_setup_2_and_3.csv")

# Numerical settings for the deterministic reference distribution. The loss grid
# has 2^16 + 1 points over [0, 1000], i.e. a step of about USD 0.015; the Beta
# integral uses 32 Gauss-Jacobi nodes. Both were checked for convergence:
# doubling either changes VaR/TVaR by at most USD 0.01.
DEFAULT_GRID_POINTS = 1 << 16
DEFAULT_QUADRATURE_NODES = 32

DEFAULT_SIMULATIONS = 25_000
CHUNK_SIZE = 50_000

# Confidence levels stored as exact rational pairs (numerator, denominator) so
# that the order-statistic rank ceil(gamma * n) can be computed with integer
# arithmetic. Using floats here would make e.g. ceil(0.99 * 100) ambiguous.
CL_FRACTIONS: dict[str, tuple[int, int]] = {
    "97%": (97, 100),
    "99%": (99, 100),
    "99.99%": (9_999, 10_000),
}
CL_LABELS: tuple[str, ...] = tuple(CL_FRACTIONS)


@dataclass(frozen=True)
class AnalyticalMoments:
    """Closed-form expected loss E[L] and loss volatility sqrt(Var(L))."""

    expected_loss: float
    loss_volatility: float


@dataclass(frozen=True)
class RiskAtCL:
    """Estimated VaR and TVaR at a single confidence level."""

    var: float
    tvar: float


@dataclass(frozen=True)
class MonteCarloSummary:
    """Sample estimates from one simulated loss vector.

    expected_loss and loss_volatility are the sample mean and sample standard
    deviation of the losses; risk holds the empirical VaR/TVaR per confidence
    level.
    """

    n_sims: int
    expected_loss: float
    loss_volatility: float
    risk: dict[str, RiskAtCL]


@dataclass(frozen=True)
class ExactSummary:
    """Deterministic reference values from the exact loss distribution.

    expected_loss and loss_volatility are the mean and standard deviation of the
    discretized pmf; risk holds the exact VaR/TVaR per confidence level.
    """

    expected_loss: float
    loss_volatility: float
    risk: dict[str, RiskAtCL]


@dataclass(frozen=True)
class RunningRisk:
    """Moving VaR and TVaR as the sample grows, one entry per simulation count.

    Entry i corresponds to the first i + 1 simulated losses.
    """

    var: NDArray[np.float64]
    tvar: NDArray[np.float64]


@dataclass(frozen=True)
class RunningSeries:
    """Per-confidence-level running risk measures over all sample prefixes."""

    n_sims: int
    risk: dict[str, RunningRisk]


def exposures_setup_1() -> NDArray[np.float64]:
    """Equal exposures of USD 10 for all 100 obligors (total notional USD 1,000)."""
    return np.full(N_OBLIGORS, EQUAL_EXPOSURE, dtype=np.float64)


def load_exposures(path: str | Path = DEFAULT_CSV) -> NDArray[np.float64]:
    """Read the exposure file and validate the case-study requirements.

    The file must contain 100 non-negative exposures summing to USD 1,000.
    Only the first column is used, so both ``Exposure`` and ``exposure_usd``
    headers work.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Exposure file not found: {path}")
    frame = pd.read_csv(path)
    if frame.shape[1] < 1:
        raise ValueError("The exposure file must contain at least one column.")
    exposures = frame.iloc[:, 0].to_numpy(dtype=np.float64)
    if exposures.size != N_OBLIGORS:
        raise ValueError(f"Expected {N_OBLIGORS} exposures, found {exposures.size}.")
    if np.any(exposures < 0):
        raise ValueError("Exposures must be non-negative.")
    if not np.isclose(exposures.sum(), PORTFOLIO_NOTIONAL, rtol=1e-6):
        raise ValueError(
            f"Exposures must sum to USD {PORTFOLIO_NOTIONAL:,.0f}, "
            f"found {exposures.sum():,.2f}."
        )
    return exposures


def get_exposures(
    setup: SetupId, csv_path: str | Path = DEFAULT_CSV
) -> NDArray[np.float64]:
    """Return the exposure vector for a model setup.

    Setup 1 uses equal exposures; setups 2 and 3 share the CSV exposures.
    """
    if setup == 1:
        return exposures_setup_1()
    if setup in (2, 3):
        return load_exposures(csv_path)
    raise ValueError(f"Unknown setup: {setup}")


def analytical_moments(
    setup: SetupId,
    exposures: NDArray[np.float64],
    p: float = DEFAULT_P,
    alpha: float = BETA_ALPHA,
    beta: float = BETA_BETA,
) -> AnalyticalMoments:
    """Evaluate the closed-form expected loss and loss volatility.

    Setups 1 and 2 (independent defaults):

        E[L]  = p * sum_i E_i
        Var(L) = p (1 - p) * sum_i E_i^2

    because the default indicators are independent Bernoulli(p) variables.

    Setup 3 (common factor q ~ Beta(alpha, beta)):

        E[L]  = E[q] * sum_i E_i
        Var(L) = E[q(1-q)] * sum_i E_i^2 + Var(q) * (sum_i E_i)^2

    The first variance term is the idiosyncratic (conditional) part, the second
    is the systematic part created by the random factor q. The Beta moments are

        E[q]        = alpha / (alpha + beta)
        Var(q)      = alpha beta / ((alpha + beta)^2 (alpha + beta + 1))
        E[q(1 - q)] = alpha beta / ((alpha + beta) (alpha + beta + 1))
    """
    exposures = np.asarray(exposures, dtype=np.float64)
    sum_exposure = float(exposures.sum())
    sum_exposure_sq = float(np.square(exposures).sum())

    if setup in (1, 2):
        expected_loss = p * sum_exposure
        variance = p * (1.0 - p) * sum_exposure_sq
    elif setup == 3:
        ab = alpha + beta
        e_q = alpha / ab
        e_q1_q = alpha * beta / (ab * (ab + 1.0))
        var_q = alpha * beta / (ab**2 * (ab + 1.0))
        expected_loss = e_q * sum_exposure
        variance = e_q1_q * sum_exposure_sq + var_q * sum_exposure**2
    else:
        raise ValueError(f"Unknown setup: {setup}")

    return AnalyticalMoments(
        expected_loss=expected_loss,
        loss_volatility=float(math.sqrt(max(variance, 0.0))),
    )


def implied_default_correlation(
    p: float = DEFAULT_P, alpha: float = BETA_ALPHA, beta: float = BETA_BETA
) -> float:
    """Pairwise default correlation implied by the Beta common-factor model.

    For i != j, Cov(I_i, I_j) = E[I_i I_j] - p^2 = Var(q), hence

        Corr(I_i, I_j) = Var(q) / (p (1 - p)).
    """
    ab = alpha + beta
    var_q = alpha * beta / (ab**2 * (ab + 1.0))
    return float(var_q / (p * (1.0 - p)))


def simulate_losses(
    setup: SetupId,
    exposures: NDArray[np.float64],
    n_sims: int,
    rng: np.random.Generator,
    p: float = DEFAULT_P,
    alpha: float = BETA_ALPHA,
    beta: float = BETA_BETA,
    chunk_size: int = CHUNK_SIZE,
) -> NDArray[np.float64]:
    """Simulate independent paths of the portfolio loss L = sum_i E_i * I_i.

    Setups 1 and 2: each obligor defaults independently with probability p.

    Setup 3: first draw one common default probability q ~ Beta(alpha, beta) per
    path, then let each obligor default independently with probability q. Drawing
    q once per path is what introduces positive default correlation.

    Paths are generated in chunks to bound peak memory for large n_sims. The
    returned array preserves the simulation order, which matters for the running
    risk-measure series.
    """
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")
    exposures = np.asarray(exposures, dtype=np.float64)
    losses = np.empty(n_sims, dtype=np.float64)

    for start in range(0, n_sims, chunk_size):
        size = min(chunk_size, n_sims - start)
        if setup in (1, 2):
            # Boolean matrix of default events; the matrix product applies the
            # exposures to the defaulted obligors and sums the losses.
            defaults = rng.random((size, exposures.size)) < p
        elif setup == 3:
            q = rng.beta(alpha, beta, size=size)
            defaults = rng.random((size, exposures.size)) < q[:, None]
        else:
            raise ValueError(f"Unknown setup: {setup}")
        losses[start : start + size] = defaults @ exposures

    return losses


def _order_statistic_rank(n_sims: int, fraction: tuple[int, int]) -> int:
    """Return ceil(gamma * n_sims) for gamma = numerator / denominator.

    Used to locate the sample quantile: the rank-th smallest loss estimates
    VaR = inf{x : F_n(x) >= gamma}. The ceiling is computed with integer
    arithmetic so that boundary cases such as gamma = 0.99 and n = 100 are
    exact rather than dependent on floating-point representation.
    """
    num, den = fraction
    return -((-num * n_sims) // den)


def sample_risk_measures(
    losses: NDArray[np.float64],
    cl_fractions: dict[str, tuple[int, int]] = CL_FRACTIONS,
) -> dict[str, RiskAtCL]:
    """Sample VaR and TVaR from a loss vector (tie-inclusive definition).

    VaR at level gamma is the rank-th order statistic with rank = ceil(gamma n).
    TVaR is the mean of all sampled losses greater than or equal to that VaR, so
    observations tied exactly at the threshold are included.
    """
    losses = np.asarray(losses, dtype=np.float64)
    n_sims = losses.size
    if n_sims == 0:
        raise ValueError("At least one loss observation is required.")

    measures: dict[str, RiskAtCL] = {}
    for label, fraction in cl_fractions.items():
        rank = _order_statistic_rank(n_sims, fraction)
        # np.partition places the rank-th smallest value at index rank - 1 in
        # O(n) without fully sorting the sample.
        var = float(np.partition(losses, rank - 1)[rank - 1])
        tail = losses[losses >= var]
        measures[label] = RiskAtCL(var=var, tvar=float(tail.mean()))
    return measures


def summarize_simulation(
    losses: NDArray[np.float64],
) -> MonteCarloSummary:
    """Sample mean, sample volatility and VaR/TVaR of one simulated loss vector."""
    losses = np.asarray(losses, dtype=np.float64)
    return MonteCarloSummary(
        n_sims=int(losses.size),
        expected_loss=float(losses.mean()),
        loss_volatility=float(losses.std(ddof=1)) if losses.size > 1 else 0.0,
        risk=sample_risk_measures(losses),
    )


def _grid_pmf(
    exposures: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    weights: NDArray[np.float64],
    n_grid: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convolve the 100 two-point Bernoulli laws on a uniform loss grid.

    The loss grid is x_j = j * step for j = 0, ..., n_grid - 1 with
    step = sum(exposures) / (n_grid - 1), so the support [0, sum E_i] is covered.

    `f` is a batch of pmf vectors, one row per probability value: setup 2 uses a
    single row with probability p, while setup 3 uses one row per Gauss-Jacobi
    node q_j. Each row is updated with the exact two-point convolution

        f_new(x) = (1 - q) f(x) + q f(x - E_i),

    starting from the point mass f(0) = 1. Because a grid shift E_i / step is
    generally fractional (whole + frac), the shifted mass is split by linear
    interpolation between the two neighbouring grid points:

        f_new[j] = (1 - q) f[j]
                 + q * ((1 - frac) f[j - whole] + frac f[j - whole - 1])

    This is the only approximation of the method; it conserves total mass
    exactly and its effect on the moments is O(step^2).

    The returned pmf is the quadrature-weighted average over the rows,
    weights @ f, which for setup 2 is just the single row.
    """
    max_loss = float(exposures.sum())
    step = max_loss / (n_grid - 1)
    f = np.zeros((probabilities.size, n_grid), dtype=np.float64)
    f[:, 0] = 1.0
    g = np.empty_like(f)
    probability = probabilities[:, None]

    for exposure in exposures:
        # Split the shift into an integer part and a fractional interpolation
        # weight. `whole` can be zero for exposures smaller than the grid step.
        shift = exposure / step
        whole = int(math.floor(shift))
        frac = shift - whole

        # g receives the shifted row(s); then f <- f + q (g - f) computes the
        # convex combination (1 - q) f + q g in place for every row at once.
        g.fill(0.0)
        np.multiply(f[:, : n_grid - whole], 1.0 - frac, out=g[:, whole:])
        if whole + 1 < n_grid:
            g[:, whole + 1 :] += frac * f[:, : n_grid - whole - 1]
        np.subtract(g, f, out=g)
        g *= probability
        f += g

    pmf = weights @ f
    grid = np.arange(n_grid, dtype=np.float64) * step
    return grid, pmf


def exact_pmf(
    setup: SetupId,
    exposures: NDArray[np.float64],
    p: float = DEFAULT_P,
    alpha: float = BETA_ALPHA,
    beta: float = BETA_BETA,
    n_grid: int = DEFAULT_GRID_POINTS,
    n_quadrature: int = DEFAULT_QUADRATURE_NODES,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the exact loss pmf on a uniform grid; returns (grid, pmf).

    Setup 1 uses the closed-form Binomial(100, p) distribution of the default
    count K, with L = 10 K. No discretization is needed.

    Setup 2 convolves the two-point laws with the single probability p.

    Setup 3 is a mixture: conditional on q the loss is a weighted
    Poisson-binomial law, and the unconditional law averages those conditional
    laws over q ~ Beta(alpha, beta). The integral

        F_L(x) = integral_0^1 F_PB(x; q) * q^(alpha-1) (1-q)^(beta-1)
                 / B(alpha, beta) dq

    is evaluated with Gauss-Jacobi quadrature. `scipy.special.roots_jacobi(n, a, b)`
    returns nodes x in (-1, 1) and weights w for

        integral_{-1}^{1} (1 - x)^a (1 + x)^b g(x) dx ~= sum_j w_j g(x_j).

    Choosing a = beta - 1 and b = alpha - 1 gives the Beta weight after mapping
    x -> q = (x + 1) / 2, because

        (1 - x)^(beta-1) (1 + x)^(alpha-1) dx
            = 2^(alpha + beta - 1) q^(alpha-1) (1 - q)^(beta-1) dq.

    Dividing the raw weights by 2^(alpha + beta - 1) * B(alpha, beta) therefore
    turns the sum into an expectation under the Beta density:

        integral_0^1 g(q) Beta_pdf(q) dq ~= sum_j weights_j g(q_j).

    The mapped nodes q_j = (x_j + 1) / 2 are the per-path default probabilities
    fed into the same grid convolution as setup 2.
    """
    exposures = np.asarray(exposures, dtype=np.float64)

    if setup == 1:
        # Closed form: K ~ Binomial(100, p) and L = exposure * K (equal exposures).
        counts = np.arange(N_OBLIGORS + 1)
        return exposures[0] * counts, stats.binom.pmf(counts, N_OBLIGORS, p)

    if setup == 2:
        # One convolution pass with the fixed default probability.
        probabilities = np.array([p], dtype=np.float64)
        weights = np.array([1.0], dtype=np.float64)
    elif setup == 3:
        # Gauss-Jacobi nodes and weights for the weight (1 - x)^(beta-1)
        # (1 + x)^(alpha-1), then map to q in (0, 1) and normalize so that the
        # sum approximates an integral against the Beta(alpha, beta) density.
        nodes, quadrature_weights = special.roots_jacobi(
            n_quadrature, beta - 1.0, alpha - 1.0
        )
        probabilities = 0.5 * (nodes + 1.0)
        weights = quadrature_weights / (
            2.0 ** (alpha + beta - 1.0) * special.beta(alpha, beta)
        )
    else:
        raise ValueError(f"Unknown setup: {setup}")

    # The batch rows are the quadrature nodes; _grid_pmf returns their weighted
    # average, i.e. the mixture pmf.
    return _grid_pmf(exposures, probabilities, weights, n_grid)


def exact_summary(
    setup: SetupId,
    exposures: NDArray[np.float64],
    p: float = DEFAULT_P,
    alpha: float = BETA_ALPHA,
    beta: float = BETA_BETA,
    n_grid: int = DEFAULT_GRID_POINTS,
    n_quadrature: int = DEFAULT_QUADRATURE_NODES,
) -> ExactSummary:
    """Evaluate exact VaR/TVaR, expected loss and volatility from the pmf.

    The pmf returned by :func:`exact_pmf` is renormalized (guarding against tiny
    negative values from grid interpolation), the CDF is built by cumulative
    summation, and VaR is obtained by inverting the CDF:

        VaR_gamma = x_{i*},  i* = min{i : F_i >= gamma},

    which is a binary search (`np.searchsorted`) because the distribution is
    discrete and its CDF is a step function. There is no Newton-Raphson or other
    root-finding step: the quantile is exactly a grid point, not the zero of a
    smooth equation.

    TVaR is the tie-inclusive tail mean of the pmf, including the full
    probability atom at the VaR grid point:

        TVaR_gamma = sum_{j >= i*} x_j f_j / sum_{j >= i*} f_j.

    Expected loss and volatility are computed from the same pmf and serve as a
    cross-check of the closed forms in :func:`analytical_moments` (they agree to
    roughly 1e-7).
    """
    grid, pmf = exact_pmf(
        setup,
        exposures,
        p=p,
        alpha=alpha,
        beta=beta,
        n_grid=n_grid,
        n_quadrature=n_quadrature,
    )
    # Guard against tiny numerical negatives and renormalize so the pmf sums to 1.
    pmf = np.clip(pmf, 0.0, None)
    pmf = pmf / pmf.sum()
    cdf = np.cumsum(pmf)

    risk: dict[str, RiskAtCL] = {}
    for label, (num, den) in CL_FRACTIONS.items():
        gamma = num / den
        # First grid index whose CDF reaches gamma; side="left" matches the
        # inf{x : F(x) >= gamma} definition when several points straddle gamma.
        index = int(np.searchsorted(cdf, gamma, side="left"))
        tail = pmf[index:]
        tail_mass = float(tail.sum())
        var = float(grid[index])
        tvar = float(np.dot(grid[index:], tail) / tail_mass)
        risk[label] = RiskAtCL(var=var, tvar=tvar)

    expected_loss = float(np.dot(grid, pmf))
    variance = float(np.dot(np.square(grid - expected_loss), pmf))

    return ExactSummary(
        expected_loss=expected_loss,
        loss_volatility=float(math.sqrt(max(variance, 0.0))),
        risk=risk,
    )


def running_risk_series(
    losses: NDArray[np.float64],
    cl_fractions: dict[str, tuple[int, int]] = CL_FRACTIONS,
) -> RunningSeries:
    """Compute VaR and TVaR for every prefix of the simulated loss sequence.

    Entry i of each array uses only the first i + 1 losses, which produces the
    convergence curves shown in the app. Re-sorting each prefix would be O(N^2);
    instead, two Fenwick trees (binary indexed trees) over the compressed loss
    values maintain, in O(log V) per observation:

    * `counts`: how many observations fall on each unique loss value;
    * `sums`:   the total loss contributed by each unique value.

    For a prefix of size n and level gamma, the target rank is k = ceil(gamma n).
    Binary lifting on the `counts` tree finds the largest compressed index whose
    cumulative count is still below k, so the next index holds the k-th smallest
    loss: that value is the sample VaR. The same prefix queries on `sums` give the
    total loss strictly below the VaR, and therefore the tie-inclusive tail

        TVaR = (total loss - loss below VaR) / (n - count below VaR),

    which averages every observation greater than or equal to the sample VaR,
    replicating :func:`sample_risk_measures` at each prefix.
    """
    losses = np.asarray(losses, dtype=np.float64)
    n_sims = losses.size
    if n_sims == 0:
        raise ValueError("At least one loss observation is required.")

    # Compress the (possibly repeated) loss values to 1-based ranks so the
    # Fenwick trees stay as small as the number of distinct losses.
    values = np.unique(losses)
    n_values = values.size
    compressed = np.searchsorted(values, losses)
    counts = [0] * (n_values + 1)
    sums = [0.0] * (n_values + 1)
    highest_bit = 1 << (n_values.bit_length() - 1)

    var_arrays = {label: np.empty(n_sims) for label in cl_fractions}
    tvar_arrays = {label: np.empty(n_sims) for label in cl_fractions}

    total = 0.0
    for i in range(n_sims):
        # Fenwick point update at the compressed index of the new loss.
        position = compressed[i] + 1
        value = losses[i]
        total += value
        while position <= n_values:
            counts[position] += 1
            sums[position] += value
            position += position & -position

        size = i + 1
        for label, fraction in cl_fractions.items():
            rank = _order_statistic_rank(size, fraction)

            # Binary lifting: find the largest prefix `position` whose count is
            # < rank. The rank-th smallest value then sits at `values[position]`
            # (0-based), because `position` counts how many unique slots lie
            # strictly before it.
            position = 0
            below_count = 0
            bit = highest_bit
            while bit:
                candidate = position + bit
                if candidate <= n_values and below_count + counts[candidate] < rank:
                    position = candidate
                    below_count += counts[candidate]
                bit >>= 1

            # Fenwick prefix query for the total loss strictly below the VaR.
            below_sum = 0.0
            index = position
            while index > 0:
                below_sum += sums[index]
                index -= index & -index

            var_arrays[label][i] = values[position]
            # Every observation >= VaR is in the tail, including ties at the
            # threshold, hence the exact tie-inclusive conditional mean.
            tvar_arrays[label][i] = (total - below_sum) / (size - below_count)

    risk = {
        label: RunningRisk(var=var_arrays[label], tvar=tvar_arrays[label])
        for label in cl_fractions
    }
    return RunningSeries(n_sims=n_sims, risk=risk)
