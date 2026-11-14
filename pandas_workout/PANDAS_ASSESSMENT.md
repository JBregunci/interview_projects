# Pandas Readiness Assessment

**Interview:** Jane Street - Data Engineer (1 hour)  
**Date:** 2026-11-09  
**Reviewed:** `pandas_workout/pandas_workout_solutions.ipynb`  
**Method:** reviewed the completed cells and ran the completed self-checks in memory. The workbook contains 59 exercises; 27 have answers.

## Verdict

| Metric | Count |
|---|---:|
| Exercises attempted | 27 |
| Clean | 14 |
| Must fix | 7 |
| Fragile | 6 |
| Untouched | 32 |
| Self-checks passing | 24 / 27 |

- **Strengths:** filtering, typed loading, currency and mixed-date parsing, basic cleaning, validated left joins, indicator/anti-joins, and one named aggregation.
- **Highest risk:** functions pass their self-check while violating the contract; aggregation uses the wrong column or function; missing-value and index semantics; method chaining is mostly `.assign` chains with no `.pipe`.
- **Biggest untouched areas:** aggregation extensions (`transform`, `filter`, weighted averages), reshaping (`melt`, `crosstab`), all of time series, MultiIndex, performance, and the capstone.
- **Readiness:** enough foundation to start a data-engineering interview, but slow down on contracts and edge cases. Do not try to finish all 32 untouched exercises before tomorrow.

## Scorecard

| Exercise | Status | Finding | Fix |
|---|---|---|---|
| 1.1 | FIX | Uses global `ORDERS` instead of `df`; `missing_pct` is a fraction, not a percent. | Use `df` throughout; `df.isna().mean().mul(100).round(2)`. |
| 1.2 | FRAGILE | Uses `.iloc` positions instead of index labels; no `.copy()`. | `df.loc[10:19, cols].copy()`. |
| 1.3 | PASS | Correct. | - |
| 1.4 | PASS | Correct; add `astype("string")` before `.str` for robustness. | `df["status"].astype("string").str...` |
| 2.1 | PASS | Correct. | - |
| 2.2 | PASS | Correct; add `astype("string")` for non-string inputs. | `series.astype("string").str...` |
| 2.3 | PASS | Correct. | - |
| 2.4 | FIX | Returns a new 7-column frame, dropping `order_date`, `unit_price`, `updated_at`; customer/product IDs are `object`, not `string`. | `df.copy().astype({...})`. |
| 2.5 | UNTOUCHED | JSON/Parquet round-trip. | Do after core gaps. |
| 3.1 | FIX | Rebuilds a Series with a fresh index; empty/`None` becomes the string `"missing"` instead of `NaN`. | Use `.map(mapping).astype("string")`; preserve index. |
| 3.2 | FRAGILE | `na_position="last"` plus `keep="last"` would keep a missing timestamp if a duplicate group had both. Current data has no such group. | `na_position="first"` then `keep="last"`. |
| 3.3 | PASS | Correct. `fillna(medians)` is cleaner than `np.where`. | `df.assign(**{value_col: df[value_col].fillna(medians)})` |
| 3.4 | FIX | Returns original date strings; does not assign parsed dates or reset indexes. Self-check fails. | Assign parsed values, partition with one mask, reset both. |
| 3.5 | FIX | Rebuilds a frame without `product_id`; downstream category joins break. | Use `df.assign(...)` and keep all columns. |
| 3.6 | PASS | Correct. | - |
| 3.7 | UNTOUCHED | Foreign-key audit. | High priority: common interview task. |
| 3.8 | PASS | Correct. | - |
| 4.1 | PASS | Correct; use one `assign` and `.round(2)`. | `df.assign(revenue=(...).round(2))` |
| 4.2 | FRAGILE | Returns a fresh `RangeIndex`; loses a non-default index. | `pd.Series(values, index=series.index, name="discount_band")` |
| 4.3 | FRAGILE | `qcut` on raw revenue can fail on duplicate bin edges; current data happens to pass. | `pd.qcut(df["revenue"].rank(method="first"), 4, ...)` |
| 4.4 | FIX | Invalid emails keep local/domain parts; `is_valid` uses `contains(".")` as a regex and `.astype(bool)` turns NA into `True`; `a@b` is marked valid. Self-check fails. | Use a regex `match`, then `.where(valid)`. |
| 4.5 | FRAGILE | Row-wise `apply` for year/month; not vectorized. | `dates.dt.year.astype("Int16")`, `dates.dt.month.astype("Int8")`. |
| 4.6 | UNTOUCHED | `map` versus validated `merge`. | High priority: common lookup pattern. |
| 4.7 | FRAGILE | Does not use `.pipe`; closes over the outer `df` inside lambdas. Self-check passes but the contract is not met. | Named steps + `df.pipe(...)`. |
| 5.1 | PASS | Correct; use `.groupby("status", observed=True)` on the assigned column. | - |
| 5.2 | FIX | `avg_order_value` averages `unit_price`; `first_order`/`last_order` use `order_id`; date dtype is wrong. Self-check fails. | Average `revenue`; use `order_date` min/max. |
| 5.3 | UNTOUCHED | `transform` for customer share. | High priority. |
| 5.4 | UNTOUCHED | `groupby(...).filter(...)`. | High priority. |
| 5.5 | UNTOUCHED | Weighted average with vectorized group operations. | High priority. |
| 5.6 | UNTOUCHED | Multi-key summary and within-channel shares. | High priority. |
| 5.7 | UNTOUCHED | `apply` versus `transform` for grouped z-scores. | Medium priority. |
| 6.1 | UNTOUCHED | Concatenate batches with provenance. | High priority. |
| 6.2 | PASS | Correct. | - |
| 6.3 | PASS | Correct. | - |
| 6.4 | PASS | Correct. | - |
| 6.5 | PASS | Correct. | - |
| 6.6 | UNTOUCHED | `melt` wide back to long. | High priority. |
| 6.7 | UNTOUCHED | Row-normalized `crosstab` with margins. | High priority. |
| 7.1-7.5 | UNTOUCHED | Full calendar, resample, rolling, growth, business days, quarters. | High priority: time-series questions are common. |
| 8.1-8.3 | UNTOUCHED | MultiIndex slicing, `xs`, `unstack`, ordered categoricals. | Medium priority. |
| 9.1-9.3 | UNTOUCHED | Vectorization, downcasting, pure refactor. | High priority: Jane Street cares about efficiency and clean code. |
| 10.1-10.5 | UNTOUCHED | End-to-end pure pipeline capstone. | Highest priority after the fixes. |
| 11.1-11.5 | UNTOUCHED | RFM, cohort retention, funnel, `merge_asof`, dedupe. | Medium priority. |

## What the self-checks did not catch

These passed but are not safe to reuse:

- **1.1:** the check calls `inspect_table(ORDERS)`, so using the global `ORDERS` hides the missing `df` argument.
- **2.4:** the check verifies memory and selected dtypes but not column preservation or ID dtypes.
- **3.1:** the check uses a `RangeIndex`, so rebuilding the Series hides the index loss.
- **3.5:** the check verifies cleaned columns but not `product_id`, which was dropped.
- **4.2:** the check uses the default index, so rebuilding the Series hides the index loss.
- **4.3:** the check fixture has unique revenue values, so `qcut` does not hit duplicate edges.
- **4.7:** the check verifies output values, not that `.pipe` was used.

The lesson: a passing check is necessary, not sufficient. Add your own invariants: column list, index, dtypes, row counts, uniqueness, and totals.

## Recurring gaps

### 1. Contract drift

Your functions sometimes satisfy the example but not the stated contract.

Examples:

- `inspect_table(df)` ignores `df`.
- `optimize_dtypes(df)` and `clean_products(df)` silently drop columns.
- `clean_country(series)` returns a new index.
- `customer_metrics` returns the wrong columns with the right names.

In an interview, read the contract as a test list. Before returning, check:

```python
assert list(out.columns) == expected_columns
assert out.index.equals(expected_index)
assert out["id"].is_unique
assert out["quantity"].dtype == "Int32"
```

### 2. Missing values, index, and dtypes

- Prefer `NaN`/`pd.NA` over sentinel strings such as `"missing"`.
- Never call `.astype(bool)` on a nullable boolean or a `contains` result; NA becomes `True`.
- `.str.contains(".")` uses a regex. Use `regex=False` for a literal dot or a full `str.match` pattern.
- `pd.Series(values)` creates a new `RangeIndex`; pass `index=series.index` when broadcasting back.
- `.astype(str)` turns missing values into the string `"nan"` and produces `object`. Use `"string"` for text IDs.
- Preserve extension dtypes: `"string"`, `"Int32"`, `"Float64"`, `"boolean"`, `"category"`.

### 3. Aggregation correctness

Named aggregation is only safe when the column-function pairs are correct.

- `avg_order_value=("unit_price", "mean")` computes the mean unit price, not the average order value.
- `first_order=("order_id", "min")` returns the smallest order ID, not the first date.
- `size` counts rows; `count` counts non-null values; `nunique` counts distinct values.
- `agg` reduces to one row per group; `transform` returns a Series aligned to the original rows; `filter` keeps or drops whole groups.

Corrected `customer_metrics`:

```python
def customer_metrics(orders):
    return (
        orders.groupby("customer_id", observed=True)
        .agg(
            n_orders=("order_id", "nunique"),
            total_revenue=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            first_order=("order_date", "min"),
            last_order=("order_date", "max"),
        )
        .sort_index()
    )
```

### 4. Method chaining and purity

Your chains are mostly repeated `.assign` calls. That is acceptable, but you are missing the interview signal that `.pipe` provides: named, independently testable steps.

Current issues:

- `4.7` does not use `.pipe`, despite the contract.
- Lambdas close over the outer `df` instead of using their argument. If an earlier step changes the frame, the lambda reads stale data.
- `3.1` and `4.2` rebuild Series and lose the index.
- `np.where` is used where `.fillna()` or `.where()` expresses intent.
- Attribute access (`x.quantity`) works for simple names but is fragile; prefer `x["quantity"]`.

Target pattern:

```python
def prepare_orders(df):
    def with_price(frame):
        return frame.assign(unit_price=parse_currency(frame["unit_price"]))

    def with_revenue(frame):
        return frame.assign(
            revenue=(
                frame["quantity"] * frame["unit_price"] * (1 - frame["discount"])
            ).round(2)
        )

    def with_flag(frame):
        return frame.assign(is_high_value=frame["revenue"].ge(20))

    return df.pipe(with_price).pipe(with_revenue).pipe(with_flag)
```

Rules:

1. Normalize and parse once, at the top of the pipeline.
2. One responsibility per helper.
3. The lambda/helper argument is the current frame, not the outer frame.
4. Return new objects; do not mutate inputs.
5. Use `[]` for column access.
6. Keep a chain readable. If it needs more than about five steps or complex conditionals, name the steps.
7. Do not force `.pipe` for a single operation; clarity beats ceremony.
8. Validate at boundaries: after parsing, after joins, after filtering.

### 5. Coverage gaps

The 32 untouched exercises include the highest-yield interview material:

- `transform`, `filter`, weighted averages, multi-key shares (5.3-5.6).
- `concat`, `melt`, `crosstab` (6.1, 6.6, 6.7).
- Full-calendar time series, `resample`, rolling, growth, business days (7.1-7.5).
- Vectorized refactors and downcasting (9.1-9.3).
- The pure, composed capstone (10.3-10.5).
- `merge_asof`, RFM, and cohort retention (11.1, 11.2, 11.4).

## Fix recipes

### 1.1 Column audit

```python
def inspect_table(df):
    sample = df.iloc[0] if len(df) else pd.Series(index=df.columns, dtype=object)
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing_pct": df.isna().mean().mul(100).round(2),
            "n_unique": df.nunique(dropna=True),
            "sample": sample,
        }
    )
```

### 2.4 Dtype optimization

```python
def optimize_dtypes(df):
    return df.copy().astype(
        {
            "order_id": "string",
            "customer_id": "string",
            "product_id": "string",
            "quantity": "Int32",
            "discount": "Float32",
            "status": "category",
            "channel": "category",
        }
    )
```

### 3.1 Country normalization

```python
def clean_country(series):
    mapping = {
        "united states": "United States", "usa": "United States",
        "us": "United States", "u.s.": "United States",
        "united kingdom": "United Kingdom", "uk": "United Kingdom",
        "u.k.": "United Kingdom", "great britain": "United Kingdom",
        "canada": "Canada", "ca": "Canada", "can": "Canada",
        "germany": "Germany", "de": "Germany", "deutschland": "Germany",
        "france": "France", "fr": "France",
    }
    return (
        series.astype("string").str.strip().str.lower()
        .map(mapping).astype("string")
    )
```

### 3.4 Date parsing and quarantine

```python
def clean_order_dates(df):
    parsed = pd.to_datetime(df["order_date"], format="mixed", errors="coerce")
    bad_mask = parsed.isna()
    clean = (
        df.loc[~bad_mask]
        .assign(order_date=parsed[~bad_mask])
        .reset_index(drop=True)
    )
    bad = (
        df.loc[bad_mask]
        .assign(order_date=parsed[bad_mask])
        .reset_index(drop=True)
    )
    return clean, bad
```

### 3.5 Product cleaning

```python
def clean_products(df):
    category_map = {
        "electronics": "Electronics", "electronic": "Electronics",
        "home office": "Home Office", "home & office": "Home Office",
        "accessories": "Accessories", "accessory": "Accessories",
        "furniture": "Furniture",
    }
    active_map = {
        "y": True, "yes": True, "1": True,
        "n": False, "no": False, "0": False,
    }
    active = (
        df["active"].astype("string").str.strip().str.lower()
        .map(active_map).astype("boolean")
    )
    return df.assign(
        product_name=df["product_name"].str.strip(),
        category=(
            df["category"].astype("string").str.strip().str.lower()
            .map(category_map).astype("string")
        ),
        unit_cost=pd.to_numeric(df["unit_cost"], errors="coerce"),
        launch_date=pd.to_datetime(df["launch_date"], format="mixed", errors="coerce"),
        active=active,
    )
```

### 4.4 Email parsing

```python
def extract_email_parts(series):
    normalized = series.astype("string").str.strip().str.lower()
    valid = normalized.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False)
    parts = normalized.str.split("@", n=1, expand=True)
    local_part = parts[0].where(valid)
    if 1 in parts.columns:
        domain = parts[1].where(valid)
    else:
        domain = pd.Series(pd.NA, index=series.index, dtype="string")
    return pd.DataFrame(
        {
            "email_norm": normalized,
            "local_part": local_part,
            "domain": domain,
            "is_valid": valid,
        }
    )
```

### 5.2 Customer metrics

```python
def customer_metrics(orders):
    return (
        orders.groupby("customer_id", observed=True)
        .agg(
            n_orders=("order_id", "nunique"),
            total_revenue=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            first_order=("order_date", "min"),
            last_order=("order_date", "max"),
        )
        .sort_index()
    )
```

### 4.7 Pure pipeline with `.pipe`

```python
def prepare_orders(df):
    def with_price(frame):
        return frame.assign(unit_price=parse_currency(frame["unit_price"]))

    def with_revenue(frame):
        return frame.assign(
            revenue=(
                frame["quantity"] * frame["unit_price"] * (1 - frame["discount"])
            ).round(2)
        )

    def with_flag(frame):
        return frame.assign(is_high_value=frame["revenue"].ge(20))

    return df.pipe(with_price).pipe(with_revenue).pipe(with_flag)
```

### Quick fixes for the fragile exercises

- **1.2:** `df.loc[10:19, ["order_id", "customer_id", "quantity"]].copy()`
- **3.2:** sort with `na_position="first"`, then `drop_duplicates("order_id", keep="last")`.
- **4.2:** `pd.Series(values, index=series.index, name="discount_band")`.
- **4.3:** `pd.qcut(df["revenue"].rank(method="first"), 4, labels=[...])`.
- **4.5:** `dates.dt.year.astype("Int16")`, `dates.dt.month.astype("Int8")`.
- **4.1:** one `assign` with `.round(2)`.
- **5.1:** `.groupby("status", observed=True)` after `assign`.

## Know cold

### Selection and indexing

| Pattern | Why |
|---|---|
| `df.loc[10:19, ["a", "b"]]` | Label-based; end label is inclusive. |
| `df.iloc[10:20, 0:2]` | Position-based; end position is exclusive. |
| `df.loc[mask, cols]` | Boolean filtering with explicit columns. |
| `mask = (a) & (b) & ~(c)` | Parenthesize every condition. |
| `df.loc[lambda d: d["x"] > 0]` | Chaining-friendly filter. |
| `.copy()` | Return an independent frame. |
| `reset_index(drop=True)` | Fresh `RangeIndex` after filtering/dedup. |

### Types and missing values

| Pattern | Why |
|---|---|
| `read_csv(dtype={"id": "string"}, na_values=["", "N/A"])` | Explicit schema and missing tokens. |
| `astype({"quantity": "Int32", "discount": "Float32"})` | Nullable, memory-aware dtypes. |
| `pd.to_numeric(..., errors="coerce")` | Invalid values become missing. |
| `pd.to_datetime(..., format="mixed", errors="coerce")` | Mixed formats, invalid dates become `NaT`. |
| `df.isna()`, `notna()`, `fillna()`, `dropna(subset=...)` | Missing-value operations. |
| `.where(cond)`, `.mask(cond)` | Keep/replace conditionally without losing index. |
| `"string"` not `str` | Avoids `"nan"` and keeps nullable text. |

### Strings and dates

| Pattern | Why |
|---|---|
| `.str.strip().str.lower()` | Normalize labels before mapping/grouping. |
| `.str.replace("$", "", regex=False)` | Literal replacement. |
| `.str.split("@", n=1, expand=True)` | Split into columns. |
| `.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False)` | Email validation without regex-dot traps. |
| `.dt.year`, `.dt.month`, `.dt.day_name()`, `.dt.weekday` | Date parts. |
| `.dt.normalize()`, `.dt.to_period("M"/"Q")` | Day/month/quarter keys. |
| `pd.date_range(start, end, freq="D"/"B"/"ME")` | Complete calendar. |
| `resample("ME").sum()`, `rolling(7, min_periods=1).mean()` | Time aggregation. |

### GroupBy

| Pattern | Why |
|---|---|
| `groupby(key, observed=True).agg(total=("col", "sum"), n=("id", "nunique"))` | Named aggregation; no MultiIndex columns. |
| `size` vs `count` vs `nunique` | Rows vs non-null vs distinct. |
| `transform("sum")` | Broadcast a group result to every row. |
| `agg(...)` | Reduce to one row per group. |
| `filter(lambda g: ...)` | Keep or drop whole groups. |
| `sort_values([...], ascending=[...])` | Multi-key ordering. |
| `rank(method="first")` | Stable ranks before `qcut`. |

### Joins and reshaping

| Pattern | Why |
|---|---|
| `merge(..., how="left", validate="many_to_one", indicator=True)` | Detect row multiplication and unmatched keys. |
| `_merge` values `both`, `left_only`, `right_only` | Audit join results. |
| Anti-join: left merge + `_merge == "left_only"` | Find missing relationships. |
| `pd.concat(frames, ignore_index=True)` | Stack batches with a fresh index. |
| `pivot_table(index, columns, values, aggfunc, fill_value=0, margins=True)` | Wide report with totals. |
| `melt(id_vars, var_name, value_name)` | Wide to long. |
| `crosstab(..., normalize="index", margins=True)` | Row shares plus totals. |
| `merge_asof(left, right, on, by, direction="backward")` | Attach latest prior value. |
| `.xs(...)`, `.unstack(...)`, `.stack()` | MultiIndex slicing and reshaping. |

### Purity, performance, validation

| Pattern | Why |
|---|---|
| `df.assign(a=..., b=...)` | New frame, multiple columns. |
| `df.pipe(f).pipe(g)` | Named pure steps. |
| Avoid `inplace=True`, chained assignment, `iterrows`, row-wise `apply` | Mutation and performance traps. |
| `pd.cut(...)`, `pd.qcut(series.rank(method="first"), 4)` | Explicit bins and tie-safe quartiles. |
| `pd.testing.assert_frame_equal`, `assert_series_equal` | Prove purity and equality. |
| Assert column list, index, dtypes, uniqueness, totals | Catch contract drift. |

## Study plan for tonight

If you have about 3.5 hours:

1. **30 min:** Fix the three hard failures: `3.4`, `4.4`, `5.2`. Re-read each prompt and rewrite from scratch.
2. **25 min:** Fix the silent contract bugs: `1.1`, `2.4`, `3.1`, `3.5`; rewrite `4.7` with `.pipe`.
3. **45 min:** Do `5.3`, `5.4`, `5.6`, `6.6`, `6.7`. These are high-yield and quick.
4. **45 min:** Do `7.1`, `7.2`, `9.3`, `10.4`; read the `10.5` solution and explain it aloud.
5. **60 min:** Take the mock notebook under a timer.
6. **15 min:** Review the checklist and write down three patterns you missed.

If you have only two hours:

1. Fix `3.4`, `4.4`, `5.2`, `2.4`, `3.5`.
2. Rewrite `4.7` with `.pipe`.
3. Take the mock.
4. Review the checklist.

## Interview advice

- Clarify the grain, keys, missing-value policy, dedupe rule, date/timezone assumptions, and expected output before coding.
- State your pipeline in words, then write named steps.
- Validate as you go: row counts, uniqueness, totals, and dtypes.
- Mention trade-offs: `map` vs `merge`, `agg` vs `transform`, vectorization vs `apply`, memory vs precision.
- Keep chains readable. Jane Street will care more about correct reasoning and clear code than about clever one-liners.
- If you get stuck, write the naive version, verify it, then optimize and explain the change.
- Do not spend the whole hour on pandas trivia. The interview is about data engineering judgment.

## Practice material

- `pandas_workout/PANDAS_CHEATSHEET.pdf` - three-page printable A4 reference with the core commands, function parameters, syntax-highlighted code, and a short regex section.
- `pandas_workout/MOCK_INTERVIEW_60MIN.ipynb` - timed run with isolated tasks, exact self-checks, and collapsible reference solutions.
