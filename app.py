import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

# ============================================================
# App Config
# ============================================================
APP_NAME = "Difference-in-Differences Studio"
APP_SUBTITLE = "Production-ready DiD estimator with design checks, plots, and placebo diagnostics."

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Global Styling (LIGHT)
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

:root{
  --bg: #ffffff;
  --text: #111827;
  --muted: #6b7280;
  --border: #e5e7eb;
  --panel: #f9fafb;
}

.stApp{
  background: var(--bg);
  color: var(--text);
  font-family: "Plus Jakarta Sans", system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}

.block-container{
  max-width: 1400px;
  padding-top: 1.0rem;
  padding-bottom: 2.0rem;
}

header, footer{ visibility: hidden; }

h1, h2, h3{ letter-spacing: -0.02em; color: #030712 !important; }
p, li{ color: #1f2937; }

section[data-testid="stSidebar"]{
  background: var(--panel);
  border-right: 1px solid var(--border);
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] > div{
  background: #ffffff !important;
  border: 1px solid #d1d5db !important;
  color: #030712 !important;
  border-radius: 10px !important;
}

div[data-testid="stTabs"] button{
  color: #4b5563 !important;
  font-weight: 650;
}
div[data-testid="stTabs"] button[aria-selected="true"]{
  color: #000000 !important;
  font-weight: 800;
  border-bottom: 3px solid #111827;
}
div[data-testid="stDataFrame"]{
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 12px;
}
div[data-testid="stMetric"]{
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px;
}
div[data-testid="stMetricValue"]{
  color: #000000 !important;
  font-weight: 850;
}
section[data-testid="stFileUploaderDropzone"]{
  background: var(--panel) !important;
  border: 1px dashed #9ca3af !important;
  border-radius: 14px !important;
}
.small-muted{
  color: var(--muted);
  font-size: 0.94rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Helpers
# ============================================================
def default_index(cols: pd.Index, preferred: str) -> int:
    return int(cols.get_loc(preferred)) if preferred in cols else 0


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def coerce_binary01(series: pd.Series) -> pd.Series:
    sn = pd.to_numeric(series, errors="coerce")
    vals = set(sn.dropna().unique().tolist())
    if vals.issubset({0, 1}):
        return sn.astype(float)
    # fallback: treat >0 as 1
    return (sn.fillna(0) > 0).astype(int).astype(float)


def try_parse_time_key(s: pd.Series):
    """
    Returns:
      time_key: series that sorts correctly (numeric or datetime if largely parseable; else string)
      time_type: "numeric" | "datetime" | "string"
    """
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().mean() >= 0.90:
        return sn, "numeric"

    sd = pd.to_datetime(s, errors="coerce")
    if sd.notna().mean() >= 0.90:
        return sd, "datetime"

    return s.astype(str), "string"


def sorted_unique_times(time_key: pd.Series, time_type: str) -> list:
    vals = time_key.dropna().unique()
    if time_type == "numeric":
        return sorted(vals.tolist())
    if time_type == "datetime":
        return sorted(vals.tolist())
    # string
    return sorted(vals.tolist(), key=lambda x: str(x))


def infer_post_start(time_key: pd.Series, post: pd.Series, time_type: str):
    mask = post == 1
    if mask.sum() == 0:
        return None
    tk = time_key.loc[mask].dropna()
    if tk.empty:
        return None
    if time_type in ("numeric", "datetime"):
        return tk.min()
    # string
    return sorted_unique_times(tk, "string")[0]


def build_2x2_cell_counts(df: pd.DataFrame, treated_col: str, post_col: str) -> pd.DataFrame:
    cell = df.groupby([treated_col, post_col]).size().reset_index(name="n")
    # ensure all 4 show up (fill 0)
    base = pd.MultiIndex.from_product([[0.0, 1.0], [0.0, 1.0]], names=[treated_col, post_col]).to_frame(index=False)
    out = base.merge(cell, on=[treated_col, post_col], how="left").fillna({"n": 0})
    out["n"] = out["n"].astype(int)
    return out


def did_design_checklist(dfm: pd.DataFrame) -> tuple:
    """
    Returns: (status, messages, cell_counts)
    status: "ok"|"warn"|"fail"
    messages: list of (level, text)
    """
    msgs = []
    status = "ok"

    if dfm.empty:
        return "fail", [("fail", "No usable rows after cleaning. Check missing values / column selections.")], build_2x2_cell_counts(
            pd.DataFrame({"treated": [], "post": []}), "treated", "post"
        )

    # Outcome numeric already coerced, but check still
    if dfm["y"].notna().mean() < 0.90:
        msgs.append(("fail", "Outcome (Y) is not reliably numeric after coercion. Choose a numeric outcome column."))
        status = "fail"

    if dfm["treated"].nunique(dropna=True) < 2:
        msgs.append(("fail", "Treated indicator has no variation (need both 0 and 1)."))
        status = "fail"

    if dfm["post"].nunique(dropna=True) < 2:
        msgs.append(("fail", "Post indicator has no variation (need both 0 and 1)."))
        status = "fail"

    if dfm["unit"].nunique(dropna=True) < 2:
        msgs.append(("fail", "Too few units (need ≥ 2 unique units)."))
        status = "fail"

    if dfm["time_key"].nunique(dropna=True) < 2:
        msgs.append(("fail", "Too few time periods (need ≥ 2 unique time values)."))
        status = "fail"

    cell_counts = build_2x2_cell_counts(dfm, "treated", "post")
    missing = cell_counts.loc[cell_counts["n"] == 0, ["treated", "post"]]
    if not missing.empty:
        pairs = [tuple(map(float, r)) for r in missing.to_numpy().tolist()]
        msgs.append(("fail", f"Missing 2×2 cells (treated/control × pre/post): {pairs}."))
        status = "fail"
    else:
        if (cell_counts["n"] < 5).any():
            msgs.append(("warn", "Small cell counts: some treated/control × pre/post cells have < 5 observations."))
            if status != "fail":
                status = "warn"

    n = len(dfm)
    if n < 30:
        msgs.append(("warn", f"Only {n} usable rows after cleaning (results may be unstable)."))
        if status == "ok":
            status = "warn"

    return status, msgs, cell_counts


def build_formula(covariates: list) -> str:
    base = "y ~ treated + post + treated:post"
    if covariates:
        base += " + " + " + ".join(covariates)
    return base


def fit_with_se(df: pd.DataFrame, formula: str, se_mode: str):
    res_base = smf.ols(formula, data=df).fit()

    if se_mode == "Robust (HC1)":
        return res_base, res_base.get_robustcov_results(cov_type="HC1"), "Robust (HC1)"

    if se_mode == "Cluster by unit":
        return res_base, res_base.get_robustcov_results(cov_type="cluster", groups=df["unit"]), "Cluster-robust (by unit)"

    if se_mode == "Cluster by time":
        return res_base, res_base.get_robustcov_results(cov_type="cluster", groups=df["time_cluster"]), "Cluster-robust (by time)"

    if se_mode == "Two-way cluster (unit & time)":
        return res_base, None, "Two-way cluster (unit & time)"

    return res_base, res_base.get_robustcov_results(cov_type="HC1"), "Robust (HC1)"


def term_stats_from_res(res_any, term: str, alpha: float):
    # statsmodels results: params, bse, pvalues are pandas series with index
    if (not hasattr(res_any, "params")) or (term not in res_any.params.index):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    coef = float(res_any.params.loc[term])
    se = float(res_any.bse.loc[term])
    pval = float(res_any.pvalues.loc[term])
    ci = res_any.conf_int(alpha=alpha)
    ci_low = float(ci.loc[term, 0])
    ci_high = float(ci.loc[term, 1])
    return coef, se, pval, ci_low, ci_high


def term_stats_two_way(res_base, df: pd.DataFrame, term: str, alpha: float):
    if (not hasattr(res_base, "params")) or (term not in res_base.params.index):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    V = cov_cluster_2groups(res_base, df["unit"], df["time_cluster"])
    idx = list(res_base.params.index).index(term)
    coef = float(res_base.params.loc[term])
    se = float(np.sqrt(max(V[idx, idx], 0.0)))

    tval = coef / se if se > 0 else np.nan
    df_resid = float(getattr(res_base, "df_resid", np.nan))
    if np.isfinite(df_resid) and np.isfinite(tval):
        pval = float(2 * stats.t.sf(np.abs(tval), df=df_resid))
        tcrit = float(stats.t.ppf(1 - alpha / 2, df=df_resid))
        ci_low = float(coef - tcrit * se)
        ci_high = float(coef + tcrit * se)
    else:
        pval, ci_low, ci_high = np.nan, np.nan, np.nan

    return coef, se, pval, ci_low, ci_high


def compact_reg_table(res_any, alpha: float, top_terms: list = None) -> pd.DataFrame:
    if top_terms is None:
        top_terms = []
    idx = list(res_any.params.index)
    df_out = pd.DataFrame(
        {
            "term": idx,
            "coef": [float(x) for x in res_any.params.values],
            "se": [float(x) for x in res_any.bse.values],
            "t": [float(x) for x in res_any.tvalues.values],
            "p": [float(x) for x in res_any.pvalues.values],
        }
    )
    ci = res_any.conf_int(alpha=alpha)
    df_out["ci_low"] = [float(ci.iloc[i, 0]) for i in range(ci.shape[0])]
    df_out["ci_high"] = [float(ci.iloc[i, 1]) for i in range(ci.shape[0])]

    # Ordering: show key terms first if present
    if top_terms:
        order = []
        for t in top_terms:
            if t in df_out["term"].values:
                order.append(t)
        rest = [t for t in df_out["term"].tolist() if t not in order]
        df_out["__order"] = df_out["term"].apply(lambda x: order.index(x) if x in order else (len(order) + rest.index(x)))
        df_out = df_out.sort_values("__order").drop(columns="__order")

    return df_out


def summary_stats_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    desc = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    # reorder
    want = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    present = [c for c in want if c in desc.columns]
    desc = desc[present]
    return desc


def safe_fig_show(fig):
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


# ============================================================
# Synthetic Dataset
# ============================================================
def generate_synthetic(seed=7, n_units=200, n_periods=10, cutoff=6, effect=2.0):
    rng = np.random.default_rng(seed)
    rows = []
    unit_fe = rng.normal(0, 1.0, size=n_units)
    for u in range(n_units):
        treated = int(u < n_units / 2)
        for t in range(n_periods):
            post = int(t >= cutoff)
            x1 = rng.normal(0, 1)
            x2 = rng.normal(0, 1)
            y = (
                5.0
                + 0.25 * t
                + 0.6 * treated
                + effect * (treated * post)
                + 0.4 * x1
                - 0.3 * x2
                + unit_fe[u]
                + rng.normal(0, 1.0)
            )
            rows.append(
                {
                    "unit_id": u,
                    "time": t,
                    "treated": treated,
                    "post": post,
                    "y": y,
                    "x1": x1,
                    "x2": x2,
                }
            )
    return pd.DataFrame(rows)


# ============================================================
# Header
# ============================================================
st.markdown(f"# {APP_NAME}")
st.caption(APP_SUBTITLE)

# ============================================================
# Data Input
# ============================================================
st.subheader("Data input")
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
use_synth = st.checkbox("Use synthetic dataset (demo)", value=(uploaded is None))

df_raw = None
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception:
        st.error("Could not read the uploaded CSV. Please check the file format and try again.")
        st.stop()
else:
    if use_synth:
        df_raw = generate_synthetic()
    else:
        st.info("Upload a CSV or enable the synthetic dataset checkbox to run a demo.")
        st.stop()

if df_raw is None or df_raw.empty:
    st.error("Dataset is empty or could not be loaded.")
    st.stop()

# ============================================================
# Sidebar Controls
# ============================================================
def numeric_like_columns(df: pd.DataFrame, min_non_na_share: float = 0.90) -> list:
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= min_non_na_share:
            cols.append(c)
    return cols


with st.sidebar:
    st.header("Controls")

    cols = df_raw.columns
    if len(cols) == 0:
        st.error("No columns found in dataset.")
        st.stop()

    outcome_col = st.selectbox("Outcome (Y) (numeric)", cols, index=default_index(cols, "y"))
    unit_col = st.selectbox("Unit ID", cols, index=default_index(cols, "unit_id"))
    time_col = st.selectbox("Time column", cols, index=default_index(cols, "time"))
    treated_col = st.selectbox("Treated indicator (0/1)", cols, index=default_index(cols, "treated"))
    post_col = st.selectbox("Post indicator (0/1)", cols, index=default_index(cols, "post"))

    st.divider()
    num_candidates = numeric_like_columns(df_raw, min_non_na_share=0.90)
    covariate_options = [c for c in num_candidates if c not in {outcome_col, treated_col, post_col}]
    covariates = st.multiselect("Optional covariates (numeric only)", covariate_options, default=[c for c in ["x1", "x2"] if c in covariate_options])

    st.divider()
    se_mode = st.selectbox(
        "Standard errors",
        ["Robust (HC1)", "Cluster by unit", "Cluster by time", "Two-way cluster (unit & time)"],
        index=1,
    )
    alpha = st.selectbox("Significance level α", [0.10, 0.05, 0.01], index=1)

# ============================================================
# Build modeling dataframe
# ============================================================
needed_cols = [outcome_col, unit_col, time_col, treated_col, post_col] + list(covariates)
try:
    dfm = df_raw[needed_cols].copy()
except Exception:
    st.error("Column selection error. Please re-check your chosen columns.")
    st.stop()

dfm = dfm.rename(
    columns={
        outcome_col: "y",
        unit_col: "unit",
        time_col: "time_raw",
        treated_col: "treated_raw",
        post_col: "post_raw",
    }
)

# coerce key columns
dfm["y"] = coerce_numeric(dfm["y"])
dfm["treated"] = coerce_binary01(dfm["treated_raw"])
dfm["post"] = coerce_binary01(dfm["post_raw"])

# parse time for sorting and placebo mechanics
dfm["time_key"], time_type = try_parse_time_key(dfm["time_raw"])
dfm["time_cluster"] = dfm["time_key"]

# covariates coercion
clean_covs = []
for c in covariates:
    # make safe column names for formula (statsmodels formula handles underscores well)
    safe = c
    # coerce
    dfm[safe] = coerce_numeric(dfm[safe])
    clean_covs.append(safe)

# drop missing essentials
essential = ["y", "unit", "time_key", "treated", "post"] + clean_covs
dfm = dfm.dropna(subset=essential)

# ensure treated/post are floats (0/1)
dfm["treated"] = dfm["treated"].astype(float)
dfm["post"] = dfm["post"].astype(float)

# ============================================================
# DiD Design Checklist Panel (BEFORE estimation)
# ============================================================
st.subheader("DiD Design Checklist")
status, msgs, cell_counts = did_design_checklist(dfm)

if status == "ok":
    st.success("✅ Checklist passed: dataset looks DiD-ready.")
elif status == "warn":
    st.warning("⚠️ Checklist passed with warnings: proceed carefully.")
else:
    st.error("❌ Checklist failed: fix the issues below before estimating.")

if not msgs:
    st.markdown('<div class="small-muted">No issues detected by the checklist.</div>', unsafe_allow_html=True)
else:
    for level, text in msgs:
        icon = "❌" if level == "fail" else "⚠️"
        st.markdown(f"- {icon} {text}")

with st.expander("Show 2×2 cell counts (treated/control × pre/post)", expanded=False):
    # reshape to a readable 2x2 table
    pivot = cell_counts.pivot(index="treated", columns="post", values="n").reindex(index=[0.0, 1.0], columns=[0.0, 1.0])
    pivot.index = ["Control (treated=0)", "Treated (treated=1)"]
    pivot.columns = ["Pre (post=0)", "Post (post=1)"]
    st.dataframe(pivot, use_container_width=True)

if status == "fail":
    st.info(
        "Fix guidance:\n"
        "- Ensure Treated and Post are binary (0/1) with both values present.\n"
        "- Ensure at least 2 units and 2 time periods.\n"
        "- Ensure you have observations in all four cells: treated/control × pre/post.\n"
        "- Ensure outcome and covariates are numeric (or can be coerced to numeric)."
    )
    st.stop()

# ============================================================
# Main Estimation
# ============================================================
formula_main = build_formula(clean_covs)
term_main = "treated:post"

try:
    res_base, res_rep, se_label = fit_with_se(dfm, formula_main, se_mode)
    if se_mode == "Two-way cluster (unit & time)":
        did_coef, did_se, did_p, did_ciL, did_ciH = term_stats_two_way(res_base, dfm, term_main, alpha)
        # build an "as-if" table from base, then replace SE/p/CI for the key term with 2-way values
        res_for_table = res_base
    else:
        did_coef, did_se, did_p, did_ciL, did_ciH = term_stats_from_res(res_rep, term_main, alpha)
        res_for_table = res_rep
except Exception:
    st.error("Model failed to run. Please verify your selections and data types (no stack trace shown).")
    st.stop()

# ============================================================
# Tabs Layout
# ============================================================
tab_preview, tab_est, tab_plots, tab_placebo, tab_summary = st.tabs(
    ["📄 Preview", "📌 Estimate", "📈 Plots", "🧪 Placebos", "📝 Summary"]
)

# ============================================================
# Preview Tab
# ============================================================
with tab_preview:
    st.subheader("Preview (first 25 rows)")
    st.dataframe(df_raw.head(25), use_container_width=True)

# ============================================================
# Estimate Tab
# ============================================================
with tab_est:
    st.subheader("DiD Estimate")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DiD coef (treated×post)", f"{did_coef:.4f}" if np.isfinite(did_coef) else "NA")
    c2.metric("Std. Error", f"{did_se:.4f}" if np.isfinite(did_se) else "NA")
    c3.metric("p-value", f"{did_p:.4g}" if np.isfinite(did_p) else "NA")
    c4.metric(f"{int((1 - alpha) * 100)}% CI", f"[{did_ciL:.4f}, {did_ciH:.4f}]" if np.isfinite(did_ciL) else "NA")

    st.caption(f"SE type: {se_label} | Model: {formula_main}")

    # one-line interpretation
    direction = "increases" if (np.isfinite(did_coef) and did_coef > 0) else ("decreases" if (np.isfinite(did_coef) and did_coef < 0) else "does not change")
    st.markdown(
        f"**Interpretation:** Under the **parallel trends assumption**, the intervention {direction} the treated group’s outcome **relative to the control group** by **{did_coef:.4f}** units on average."
        if np.isfinite(did_coef)
        else "**Interpretation:** Could not compute a valid DiD estimate for the interaction term."
    )

    st.divider()

    # Regression table (compact)
    st.subheader("Regression table (compact)")
    try:
        reg_tbl = compact_reg_table(res_for_table, alpha=alpha, top_terms=["treated:post", "treated", "post"])
        # If two-way: overwrite key term stats in the table so KPIs and table align
        if se_mode == "Two-way cluster (unit & time)" and "treated:post" in reg_tbl["term"].values:
            mask = reg_tbl["term"] == "treated:post"
            reg_tbl.loc[mask, "se"] = did_se
            reg_tbl.loc[mask, "p"] = did_p
            reg_tbl.loc[mask, "ci_low"] = did_ciL
            reg_tbl.loc[mask, "ci_high"] = did_ciH
            # recompute t for display
            reg_tbl.loc[mask, "t"] = did_coef / did_se if (np.isfinite(did_coef) and np.isfinite(did_se) and did_se > 0) else np.nan

        show_tbl = reg_tbl.copy()
        for col in ["coef", "se", "t", "p", "ci_low", "ci_high"]:
            show_tbl[col] = pd.to_numeric(show_tbl[col], errors="coerce")
        st.dataframe(show_tbl, use_container_width=True)
    except Exception:
        st.warning("Could not render the regression table (data may be too irregular).")

    st.divider()

    # Summary statistics
    st.subheader("Summary statistics")
    stat_cols = ["y"] + clean_covs
    overall = summary_stats_table(dfm, stat_cols)
    if overall.empty:
        st.info("No numeric columns available for summary statistics.")
    else:
        st.markdown("**Overall**")
        st.dataframe(overall, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            by_treat = st.checkbox("Also show by group: treated vs control", value=True)
        with colB:
            by_post = st.checkbox("Also show by period: pre vs post", value=False)

        if by_treat:
            st.markdown("**By treated (0/1)**")
            out = []
            for gval in [0.0, 1.0]:
                sub = dfm.loc[dfm["treated"] == gval]
                tbl = summary_stats_table(sub, stat_cols)
                if not tbl.empty:
                    tbl.insert(0, "group", f"treated={int(gval)}")
                    out.append(tbl.reset_index().rename(columns={"index": "variable"}))
            if out:
                st.dataframe(pd.concat(out, ignore_index=True), use_container_width=True)

        if by_post:
            st.markdown("**By post (0/1)**")
            out = []
            for pval in [0.0, 1.0]:
                sub = dfm.loc[dfm["post"] == pval]
                tbl = summary_stats_table(sub, stat_cols)
                if not tbl.empty:
                    tbl.insert(0, "period", f"post={int(pval)}")
                    out.append(tbl.reset_index().rename(columns={"index": "variable"}))
            if out:
                st.dataframe(pd.concat(out, ignore_index=True), use_container_width=True)

# ============================================================
# Plots Tab
# ============================================================
with tab_plots:
    st.subheader("Plots")

    # prep time axis
    t_unique = sorted_unique_times(dfm["time_key"], time_type)
    post_start = infer_post_start(dfm["time_key"], dfm["post"], time_type)

    # 1) Parallel trends plot
    st.markdown("### 1) Parallel trends: average outcome over time (treated vs control)")
    try:
        g = dfm.groupby(["time_key", "treated"])["y"].mean().reset_index()
        pt = g.pivot(index="time_key", columns="treated", values="y").reindex(index=t_unique)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        if 0.0 in pt.columns:
            ax.plot(pt.index, pt[0.0], marker="o", linewidth=2, label="Control (treated=0)")
        if 1.0 in pt.columns:
            ax.plot(pt.index, pt[1.0], marker="o", linewidth=2, label="Treated (treated=1)")

        if post_start is not None:
            ax.axvline(post_start, linestyle="--", linewidth=2)
            ax.text(
                post_start,
                ax.get_ylim()[1],
                "  Post starts",
                va="top",
                ha="left",
                fontsize=10,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Average Y")
        ax.set_title("Parallel trends check (visual)")
        ax.legend()
        ax.grid(True, alpha=0.25)
        safe_fig_show(fig)
    except Exception:
        st.warning("Could not generate the parallel trends plot.")

    # 2) Pre vs Post comparison plot
    st.markdown("### 2) Pre vs post comparison: averages by group")
    try:
        agg = (
            dfm.assign(period=dfm["post"].map({0.0: "Pre", 1.0: "Post"}))
            .groupby(["treated", "period"])["y"]
            .mean()
            .reset_index()
        )
        # order
        period_order = ["Pre", "Post"]
        treated_order = [0.0, 1.0]

        yvals = {}
        for tr in treated_order:
            yvals[tr] = []
            for per in period_order:
                row = agg[(agg["treated"] == tr) & (agg["period"] == per)]
                yvals[tr].append(float(row["y"].iloc[0]) if not row.empty else np.nan)

        x = np.arange(len(period_order))
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(x, yvals[0.0], marker="o", linewidth=2, label="Control (treated=0)")
        ax.plot(x, yvals[1.0], marker="o", linewidth=2, label="Treated (treated=1)")
        ax.set_xticks(x)
        ax.set_xticklabels(period_order)
        ax.set_ylabel("Average Y")
        ax.set_title("Average outcome by group (Pre vs Post)")
        ax.legend()
        ax.grid(True, alpha=0.25)
        safe_fig_show(fig)
    except Exception:
        st.warning("Could not generate the pre vs post comparison plot.")

    # 3) Difference plot: treated - control over time
    st.markdown("### 3) Difference plot: (treated − control) over time")
    try:
        g = dfm.groupby(["time_key", "treated"])["y"].mean().reset_index()
        wide = g.pivot(index="time_key", columns="treated", values="y").reindex(index=t_unique)
        diff = None
        if (0.0 in wide.columns) and (1.0 in wide.columns):
            diff = wide[1.0] - wide[0.0]
        else:
            diff = pd.Series(index=wide.index, data=np.nan)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(diff.index, diff.values, marker="o", linewidth=2, label="Treated − Control")

        if post_start is not None:
            ax.axvline(post_start, linestyle="--", linewidth=2)
            ax.text(post_start, ax.get_ylim()[1], "  Post starts", va="top", ha="left", fontsize=10)

        ax.axhline(0.0, linewidth=1.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Difference in average Y")
        ax.set_title("Difference over time (visual DiD)")
        ax.legend()
        ax.grid(True, alpha=0.25)
        safe_fig_show(fig)
    except Exception:
        st.warning("Could not generate the difference plot.")

# ============================================================
# Placebos Tab
# ============================================================
with tab_placebo:
    st.subheader("Placebo diagnostics")

    st.markdown(
        """
- **Placebo A (shift earlier):** move the policy cutoff earlier by **K** time steps and re-estimate.
- **Placebo sweep curve:** repeat placebo A for **K=1..max feasible**, plot placebo estimates with CI.
- **Interpretation:** Placebo effects should be near **0** and typically **not significant**. Significant placebo effects can indicate **pre-trends** that threaten causal interpretation.
"""
    )

    # time mechanics for placebo
    t_unique = sorted_unique_times(dfm["time_key"], time_type)
    post_start = infer_post_start(dfm["time_key"], dfm["post"], time_type)

    if post_start is None:
        st.info("Placebos unavailable: could not infer a post start time (no rows with post==1).")
    elif len(t_unique) < 3:
        st.info("Placebos unavailable: need at least 3 unique time points.")
    else:
        try:
            cutoff_idx = t_unique.index(post_start)
        except ValueError:
            cutoff_idx = max(1, len(t_unique) // 2)

        max_k = int(max(1, cutoff_idx))  # ensure placebo cutoff still exists
        if max_k < 1:
            st.info("Placebos unavailable: not enough pre periods before the post start.")
        else:
            st.markdown("### Placebo A — Shift the cutoff earlier")
            K = st.slider("K (time steps earlier than the real cutoff)", min_value=1, max_value=max_k, value=1, step=1)
            placebo_cutoff = t_unique[max(0, cutoff_idx - K)]

            dfA = dfm.copy()
            dfA["placebo_post"] = (dfA["time_key"] >= placebo_cutoff).astype(int).astype(float)

            # check 2x2 cells for placebo
            placebo_cells = build_2x2_cell_counts(dfA, "treated", "placebo_post")
            missing = placebo_cells.loc[placebo_cells["n"] == 0]
            if not missing.empty:
                st.warning("Placebo A not estimable: missing treated/control × placebo pre/post cells.")
            else:
                formulaA = build_formula(clean_covs).replace(" post ", " placebo_post ").replace("treated:post", "treated:placebo_post")
                # safer: build directly
                baseA = "y ~ treated + placebo_post + treated:placebo_post"
                if clean_covs:
                    baseA += " + " + " + ".join(clean_covs)
                formulaA = baseA
                termA = "treated:placebo_post"

                try:
                    A_base, A_rep, A_se_label = fit_with_se(dfA, formulaA, se_mode)
                    if se_mode == "Two-way cluster (unit & time)":
                        A_coef, A_se, A_p, A_ciL, A_ciH = term_stats_two_way(A_base, dfA, termA, alpha)
                    else:
                        A_coef, A_se, A_p, A_ciL, A_ciH = term_stats_from_res(A_rep, termA, alpha)

                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Placebo coef (treated×placebo_post)", f"{A_coef:.4f}" if np.isfinite(A_coef) else "NA")
                    a2.metric("Std. Error", f"{A_se:.4f}" if np.isfinite(A_se) else "NA")
                    a3.metric("p-value", f"{A_p:.4g}" if np.isfinite(A_p) else "NA")
                    a4.metric(f"{int((1 - alpha) * 100)}% CI", f"[{A_ciL:.4f}, {A_ciH:.4f}]" if np.isfinite(A_ciL) else "NA")
                    st.caption(f"Real cutoff: {post_start} | Placebo cutoff: {placebo_cutoff} | SE type: {A_se_label}")

                    sig_txt = "SIGNIFICANT" if (np.isfinite(A_p) and A_p < alpha) else "not significant"
                    st.markdown(f"**Placebo A is {sig_txt} at α={alpha}.**")
                except Exception:
                    st.warning("Placebo A regression failed (no stack trace shown).")

            st.divider()

            # Sweep curve
            st.markdown("### Placebo sweep curve (K = 1..max feasible)")
            st.caption("Shows placebo interaction estimates when shifting the cutoff earlier. Includes 0-line and the main DiD estimate.")

            results = []
            for k in range(1, max_k + 1):
                cutoff_k = t_unique[max(0, cutoff_idx - k)]
                dft = dfm.copy()
                dft["placebo_post"] = (dft["time_key"] >= cutoff_k).astype(int).astype(float)

                placebo_cells = build_2x2_cell_counts(dft, "treated", "placebo_post")
                if (placebo_cells["n"] == 0).any():
                    continue

                formula_k = "y ~ treated + placebo_post + treated:placebo_post"
                if clean_covs:
                    formula_k += " + " + " + ".join(clean_covs)

                try:
                    k_base, k_rep, _ = fit_with_se(dft, formula_k, se_mode)
                    if se_mode == "Two-way cluster (unit & time)":
                        k_coef, k_se, k_p, k_ciL, k_ciH = term_stats_two_way(k_base, dft, "treated:placebo_post", alpha)
                    else:
                        k_coef, k_se, k_p, k_ciL, k_ciH = term_stats_from_res(k_rep, "treated:placebo_post", alpha)

                    results.append(
                        {
                            "K": k,
                            "placebo_cutoff": cutoff_k,
                            "coef": k_coef,
                            "se": k_se,
                            "p": k_p,
                            "ci_low": k_ciL,
                            "ci_high": k_ciH,
                        }
                    )
                except Exception:
                    continue

            if not results:
                st.info("No feasible placebo shifts produced estimable regressions (likely due to missing cells).")
            else:
                sweep = pd.DataFrame(results).sort_values("K")

                # Plot sweep
                try:
                    fig, ax = plt.subplots(figsize=(9, 4.7))
                    ax.plot(sweep["K"], sweep["coef"], marker="o", linewidth=2, label="Placebo coef")
                    ax.fill_between(sweep["K"], sweep["ci_low"], sweep["ci_high"], alpha=0.20, label=f"{int((1-alpha)*100)}% CI band")
                    ax.axhline(0.0, linewidth=1.5, linestyle="-")
                    if np.isfinite(did_coef):
                        ax.axhline(did_coef, linewidth=2, linestyle="--", label="Main DiD coef")

                    ax.set_xlabel("K (shift earlier)")
                    ax.set_ylabel("Placebo interaction estimate")
                    ax.set_title("Placebo sweep: shifting cutoff earlier")
                    ax.legend()
                    ax.grid(True, alpha=0.25)
                    safe_fig_show(fig)
                except Exception:
                    st.warning("Could not render the placebo sweep plot.")

                show_table = st.checkbox("Show placebo sweep table", value=False)
                if show_table:
                    st.dataframe(sweep, use_container_width=True)

# ============================================================
# Summary Tab
# ============================================================
with tab_summary:
    st.subheader("Narrative summary")

    sig = "Yes" if (np.isfinite(did_p) and did_p < alpha) else "No"
    direction = "increase" if (np.isfinite(did_coef) and did_coef > 0) else ("decrease" if (np.isfinite(did_coef) and did_coef < 0) else "no change")

    st.markdown(
        f"""
**Main effect (treated×post):** {did_coef:.4f if np.isfinite(did_coef) else "NA"}  
**SE type:** {se_label}  
**p-value:** {did_p:.4g if np.isfinite(did_p) else "NA"}  
**Significant at α={alpha}:** {sig}

**Interpretation:**  
After the policy starts, the treated group shows an average **{direction}** in the outcome **relative to the control group**, by about **{did_coef:.4f}** units, **under the parallel trends assumption**.

**Assumptions to keep in mind:**  
- **Parallel trends:** in the absence of treatment, treated and control would have moved similarly over time.  
- No major concurrent shocks that affect treated differently exactly at the intervention time.

**What placebos mean:**  
If shifting the cutoff earlier yields significant “effects,” it suggests your estimate may be picking up **pre-trends** rather than a causal treatment effect.
"""
    )













   
  

