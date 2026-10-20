# P0-01: SERIES_VERSION Investigation Report

## 1. Exact Location and Value of SERIES_VERSION

**Location:** [`store.py:30`](backend/src/alpha_harness/vault/store.py)

```python
SERIES_VERSION = 3
```

**The value has been 3 since the file was first committed** (verified via `git log -S`). There has never been a version 2 in the repository's history. My earlier report of `SERIES_VERSION = 2` was an error.

---

## 2. Every Writer and Reader of `series_version` and `pnl_version`

### Writers (2 paths, same file)

| Path | File:Line | What it sets |
|---|---|---|
| **`save_series`** | [`store.py:423-434`](backend/src/alpha_harness/vault/store.py) | Upserts `series_version = 3`, `pnl_version = 3` |
| **`save_pnl_only`** | [`store.py:444-449`](backend/src/alpha_harness/vault/store.py) | Upserts `series_version = None`, `pnl_version = 3` |

Both always use the constant `SERIES_VERSION` (= 3). There is no other code that writes to either column.

### Readers (4 queries, same file)

| Query | File:Line | Behavior |
|---|---|---|
| **`PNL_STORED`** | [`store.py:33`](backend/src/alpha_harness/vault/store.py) | `coalesce(pnl_version, series_version) = ?` — accepts a row if *either* version column equals 3 |
| **`pnl_grid`** | [`store.py:700-707`](backend/src/alpha_harness/vault/store.py) | Uses `PNL_STORED` sub-select with `SERIES_VERSION` |
| **`lacking_series`** | [`store.py:750-755`](backend/src/alpha_harness/vault/store.py) | `WHERE series_version = ?` (strict: requires full series, not PnL-only) |
| **`lacking_pnl`** | [`store.py:760-767`](backend/src/alpha_harness/vault/store.py) | Uses `PNL_STORED` sub-select |

### Summary

- `series_version` and `pnl_version` are **two separate columns** on the DuckDB `alpha` table, but both are filled from the **same constant** (`SERIES_VERSION = 3`).
- `series_version` marks a full PnL + turnover series.
- `pnl_version` marks a PnL-only download (no turnover), used for correlation.

---

## 3. Are These Separate Markers or the Same Constant?

**Separate columns, same constant.**

- `series_version` is set by `save_series` (full data) — set to 3.
- `pnl_version` is set by both `save_series` (= 3) and `save_pnl_only` (= 3, with `series_version` left `None`).
- The reader `PNL_STORED` coalesces them: it accepts a row if *either* is current (version 3).
- The reader `lacking_series` requires `series_version = 3` specifically (full data).

---

## 4. Which Versions Exist in the Database?

This cannot be determined safely without running queries against the user's database. However, from the git history:

- Version 3 has been the constant since the first commit (commit `4329f84`, "feat: introduce Alpha Harness").
- There is **no evidence** of version 1 or 2 ever being written by this codebase.
- Any rows with `series_version IS NULL` or `pnl_version IS NULL` represent alphas that have never been synced with `save_series` or `save_pnl_only`.

---

## 5. Whether the Implementation Requires a Version Bump

**No. A version bump is not required for P0-01.**

The rationale:

1. `SERIES_VERSION` controls **vault data freshness** (DuckDB stored PnL series), not **evaluation provenance** (SQLite `ResearchEvaluation` rows).
2. The P0-01 fix operates on the `EvaluationEngine` and its callers in the SQLite research database — completely separate from the DuckDB vault versioning.
3. Bumping `SERIES_VERSION` to 4 would force a **re-download of every alpha's PnL series** from BRAIN, which is expensive, quota-consuming, and solves nothing. The vault data is genuine BRAIN recordset data regardless of version.
4. The provenance problem is that `evaluate_candidate` accepts arbitrary PnL arrays (including synthetic ones) and persists them in `ResearchEvaluation`. This is an evaluation-layer issue, not a vault-versioning issue.

---

## 6. How Existing Genuine Version 3 Series Are Treated

Since the version has **always been 3**, every genuine series in the database is already version 3. There is nothing to migrate.

The P0-01 implementation should:

- **Leave `SERIES_VERSION = 3` unchanged.**
- **Not introduce any new version constant for vault series.**
- Focus exclusively on the evaluation-layer provenance boundary: making `EvaluationEngine.evaluate_candidate` retrieve verified series through `AlphaVault` internally, rather than accepting raw arrays from any caller.

---

## Correction of Earlier Discrepancy

> [!IMPORTANT]
> My earlier statement that `SERIES_VERSION = 2` was **incorrect**. The value has been 3 in every commit of this repository. No proposal should depend on bumping it.
