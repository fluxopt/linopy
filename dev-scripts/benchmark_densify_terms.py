"""
Benchmark: densify_terms() vectorized vs old O(n²) implementation,
and memory impact of sum(drop_zeros=True) vs sum().

Shows when drop_zeros=True helps (sparse masked variables) and when it
doesn't (dense expressions with no masked variables).

Outputs:
  - dev-scripts/benchmark_densify_speed.html   (old vs new densify time)
  - dev-scripts/benchmark_densify_memory.html   (memory with/without drop_zeros)
  - dev-scripts/benchmark_densify_when.html     (helps vs doesn't help)
"""

import time

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr

import linopy
from linopy.expressions import TERM_DIM

OUTDIR = "dev-scripts"
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Old O(n²) densify_terms for comparison
# ---------------------------------------------------------------------------
def densify_terms_old(expr):
    """Original O(n²) implementation — uses coeffs!=0 (broken for masked vars)."""
    data = expr.data.transpose(..., TERM_DIM)
    cdata = data.coeffs.data
    axis = cdata.ndim - 1
    nnz = np.nonzero(cdata)
    nterm = (cdata != 0).sum(axis).max()

    mod_nnz = list(nnz)
    mod_nnz.pop(axis)

    remaining_axes = np.vstack(mod_nnz).T
    _, idx_ = np.unique(remaining_axes, axis=0, return_inverse=True)

    idx = list(idx_)
    new_index = np.array([idx[:i].count(j) for i, j in enumerate(idx)])

    mod_nnz.insert(axis, new_index)

    vdata = np.full_like(cdata, -1)
    vdata[tuple(mod_nnz)] = data.vars.data[nnz]
    data.vars.data = vdata

    cdata_new = np.zeros_like(cdata)
    cdata_new[tuple(mod_nnz)] = data.coeffs.data[nnz]
    data.coeffs.data = cdata_new

    return expr.__class__(data.sel({TERM_DIM: slice(0, nterm)}), expr.model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_masked_expr(n_other, n_contrib, active_per_row, model, name_suffix=""):
    """Create expression from masked variables (uniform column sparsity)."""
    active_cols = RNG.choice(n_contrib, active_per_row, replace=False)
    mask = xr.DataArray(
        np.zeros((n_other, n_contrib), dtype=bool),
        dims=["other", "contrib"],
        coords={"other": range(n_other), "contrib": range(n_contrib)},
    )
    mask.data[:, active_cols] = True

    name = f"masked_{n_other}_{n_contrib}_{active_per_row}{name_suffix}"
    v = model.add_variables(
        lower=0,
        upper=1,
        mask=mask,
        name=name,
        dims=["other", "contrib"],
        coords={"other": range(n_other), "contrib": range(n_contrib)},
    )
    return 1 * v


def make_dense_expr(n_other, n_contrib, model, name_suffix=""):
    """Create expression from fully dense variables (no masking)."""
    name = f"dense_{n_other}_{n_contrib}{name_suffix}"
    v = model.add_variables(
        lower=0,
        upper=1,
        name=name,
        dims=["other", "contrib"],
        coords={"other": range(n_other), "contrib": range(n_contrib)},
    )
    return 1 * v


def nbytes_expr(expr):
    return expr.data.coeffs.data.nbytes + expr.data.vars.data.nbytes


def bench_sum(expr, drop_zeros, n_repeats=3):
    """Time sum('contrib', drop_zeros=...) and return best time in ms."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        expr.sum("contrib", drop_zeros=drop_zeros)
        times.append(time.perf_counter() - t0)
    return min(times) * 1000


def bench_old_densify(expr, n_repeats=3):
    """Time old densify on a pre-summed expression. Returns ms."""
    times = []
    for _ in range(n_repeats):
        raw = expr.sum("contrib")
        t0 = time.perf_counter()
        densify_terms_old(raw)
        times.append(time.perf_counter() - t0)
    return min(times) * 1000


# ---------------------------------------------------------------------------
# Benchmark 1: Speed — old vs new densify across problem sizes
# ---------------------------------------------------------------------------
def bench_speed():
    print("=== Benchmark: old vs new densify_terms speed ===")

    configs = [
        (50, 100, 10),
        (100, 200, 20),
        (500, 500, 50),
        (1000, 500, 50),
        (2000, 500, 50),
    ]

    rows = []
    for i, (n_other, n_contrib, active) in enumerate(configs):
        total = n_other * n_contrib
        print(f"  {n_other}×{n_contrib} ({active} active) ...")

        m = linopy.Model()
        expr = make_masked_expr(n_other, n_contrib, active, m, name_suffix=f"_sp{i}")

        t_new = bench_sum(expr, drop_zeros=True)
        rows.append(
            {
                "total_terms": total,
                "Implementation": "New (vectorized)",
                "Time (ms)": t_new,
            }
        )

        # Old: skip if too slow
        nnz = int((expr.sum("contrib").data.vars.data != -1).sum())
        if nnz <= 10_000:
            t_old = bench_old_densify(expr)
            rows.append(
                {
                    "total_terms": total,
                    "Implementation": "Old (O(n²) loop)",
                    "Time (ms)": t_old,
                }
            )
            print(f"    new={t_new:.1f}ms, old={t_old:.1f}ms")
        else:
            print(f"    new={t_new:.1f}ms, old=skipped (too slow)")

    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="total_terms",
        y="Time (ms)",
        color="Implementation",
        log_y=True,
        log_x=True,
        title="densify_terms() speed: old O(n²) vs new vectorized<br>"
        "<sup>10% active contributors, varying problem size</sup>",
        labels={"total_terms": "Total terms (rows × contributors)"},
        color_discrete_map={
            "Old (O(n²) loop)": "#ef553b",
            "New (vectorized)": "#636efa",
        },
    )
    fig.update_traces(marker=dict(size=12), mode="lines+markers")
    path = f"{OUTDIR}/benchmark_densify_speed.html"
    fig.write_html(path)
    print(f"  -> {path}\n")
    return df


# ---------------------------------------------------------------------------
# Benchmark 2: Memory — raw sum vs drop_zeros=True
# ---------------------------------------------------------------------------
def bench_memory():
    print("=== Benchmark: memory with/without drop_zeros ===")

    configs = [
        (1000, 500, 25),
        (1000, 500, 50),
        (1000, 500, 125),
        (1000, 500, 250),
        (1000, 500, 500),
    ]

    rows = []
    for i, (n_other, n_contrib, active) in enumerate(configs):
        pct = int(100 * active / n_contrib)

        m = linopy.Model()
        if active == n_contrib:
            expr = make_dense_expr(n_other, n_contrib, m, name_suffix=f"_mem{i}")
        else:
            expr = make_masked_expr(
                n_other, n_contrib, active, m, name_suffix=f"_mem{i}"
            )

        raw = expr.sum("contrib")
        compact = expr.sum("contrib", drop_zeros=True)

        mem_raw = nbytes_expr(raw) / 1024
        mem_compact = nbytes_expr(compact) / 1024
        nterm_raw = raw.data.sizes[TERM_DIM]
        nterm_compact = compact.data.sizes[TERM_DIM]

        rows.append(
            {
                "active_pct": pct,
                "Method": "sum()",
                "Memory (KB)": mem_raw,
                "_term size": nterm_raw,
            }
        )
        rows.append(
            {
                "active_pct": pct,
                "Method": "sum(drop_zeros=True)",
                "Memory (KB)": mem_compact,
                "_term size": nterm_compact,
            }
        )

        print(
            f"  {pct}% active: _term {nterm_raw}->{nterm_compact}, "
            f"mem {mem_raw:.0f}->{mem_compact:.0f} KB"
        )

    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="active_pct",
        y="Memory (KB)",
        color="Method",
        text="_term size",
        title="Memory after .sum('contrib'): raw vs drop_zeros=True<br>"
        "<sup>1000 rows × 500 contributors. Labels = _term dimension size.</sup>",
        labels={"active_pct": "Active contributors (%)"},
        color_discrete_map={
            "sum()": "#ef553b",
            "sum(drop_zeros=True)": "#636efa",
        },
    )
    fig.update_traces(
        marker=dict(size=12), mode="lines+markers+text", textposition="top center"
    )
    path = f"{OUTDIR}/benchmark_densify_memory.html"
    fig.write_html(path)
    print(f"  -> {path}\n")
    return df


# ---------------------------------------------------------------------------
# Benchmark 3: When does drop_zeros help vs hurt?
# ---------------------------------------------------------------------------
def bench_when_helps():
    print("=== Benchmark: when does drop_zeros=True help? ===")

    # Sweep active fraction from 5% to 100% at realistic scale
    n_other, n_contrib = 2000, 500
    active_values = [25, 50, 100, 150, 250, 350, 500]

    rows = []
    for i, active in enumerate(active_values):
        pct = int(100 * active / n_contrib)

        m = linopy.Model()
        if active == n_contrib:
            expr = make_dense_expr(n_other, n_contrib, m, name_suffix=f"_wh{i}")
        else:
            expr = make_masked_expr(
                n_other, n_contrib, active, m, name_suffix=f"_wh{i}"
            )

        t_plain = bench_sum(expr, drop_zeros=False)
        t_drop = bench_sum(expr, drop_zeros=True)

        raw = expr.sum("contrib")
        compact = expr.sum("contrib", drop_zeros=True)
        mem_raw = nbytes_expr(raw) / 1024
        mem_compact = nbytes_expr(compact) / 1024

        for method, t, mem in [
            ("sum()", t_plain, mem_raw),
            ("sum(drop_zeros=True)", t_drop, mem_compact),
        ]:
            rows.append(
                {
                    "active_pct": pct,
                    "Method": method,
                    "Time (ms)": t,
                    "Memory (KB)": mem,
                }
            )

        print(
            f"  {pct}% active: plain={t_plain:.1f}ms, drop_zeros={t_drop:.1f}ms, "
            f"mem {mem_raw:.0f}->{mem_compact:.0f} KB"
        )

    df = pd.DataFrame(rows)

    df_time = df[["active_pct", "Method", "Time (ms)"]].copy()
    df_time = df_time.rename(columns={"Time (ms)": "value"})
    df_time["metric"] = "Time (ms)"

    df_mem = df[["active_pct", "Method", "Memory (KB)"]].copy()
    df_mem = df_mem.rename(columns={"Memory (KB)": "value"})
    df_mem["metric"] = "Memory (KB)"

    df_long = pd.concat([df_time, df_mem])

    fig = px.scatter(
        df_long,
        x="active_pct",
        y="value",
        color="Method",
        facet_col="metric",
        title="When does drop_zeros=True help?<br>"
        "<sup>2000 rows × 500 contributors, sweep active fraction</sup>",
        labels={"active_pct": "Active contributors (%)", "value": ""},
        color_discrete_map={
            "sum()": "#ef553b",
            "sum(drop_zeros=True)": "#636efa",
        },
    )
    fig.update_traces(marker=dict(size=10), mode="lines+markers")
    fig.update_yaxes(matches=None)
    # Set each facet's y-axis title to its metric name
    fig.update_yaxes(title_text="Time (ms)", col=1)
    fig.update_yaxes(title_text="Memory (KB)", col=2)
    path = f"{OUTDIR}/benchmark_densify_when.html"
    fig.write_html(path)
    print(f"  -> {path}\n")
    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_speed = bench_speed()
    df_memory = bench_memory()
    df_when = bench_when_helps()
    print("Done.")
