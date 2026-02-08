# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from dataclasses import dataclass

# =============================
# App Identity + Styling
# =============================
APP_NAME = "DiD Insight Studio"
APP_SUBTITLE = "Difference-in-Differences estimator with diagnostics, plots, and placebo checks."

st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Dark UI that keeps header + sidebar toggle intact
st.markdown(
    """
<style>
:root{
  --bg: #0B0F14;
  --panel: #111827;
  --text: #E5E7EB;
  --muted: #9CA3AF;
  --border: rgba(255,255,255,0.10);
  --border2: rgba(255,255,255,0.08);
}

/* App background */
.stApp { background-color: var(--bg); color: var(--text); }

/* Container spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2.0rem; max-width: 1400px; }

/* Sidebar (do NOT hide it) */
section[data-testid="stSidebar"]{
  background: #0F172A;
  border-right: 1px solid var(--border);
}

/* Metric cards */
div[data-testid="stMetric"] {
  background-color: var(--panel);
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid var(--border2);
}

/* Dataframe polish */
div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--border2);
}

/* Inputs */
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] textarea{
  background: #0B1220 !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: var(--text) !important;
  border-radius: 10px !important;
}

/* File uploader */
section[data-testid="stFileUploaderDropzone"]{
  background: rgba(255,255,255,0.03) !important;
  border: 1px dashed rgba(255,255,255,0.20) !important;
  border-radius: 14px !important;
}

/* Code blocks */
pre { border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.10) !important; }

/* Subtle divider */
hr { border-top: 1px solid var(--border) !important; }

/* Small muted text helper */
.small-muted { color: var(--muted); font-size: 0.94rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# Helpers
# =============================
def default_index(cols: pd.Index, preferred: str) -> int:
    return int(cols.get_loc(preferred)) if preferred in cols else 0

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def coerce_binary01(series: pd.Series, name: str):
    """
    Coerce a series to 0/1 if possible.
    Returns (coerced_series, warning_messages_list).
    """
    msgs = []
    s = series.copy()

    if s.dtype == bool:
        return s.astype(int).astype(float), msgs

    sn = pd.to_numeric(s, errors="coerce")
    if sn.isna().any():
        msgs.append(f"'{name}' has non-numeric values; some became missing (NaN) after coercion.")

    vals = set(sn.dropna().unique().tolist())
    if vals.issubset({0, 1}):
        return sn.astype(float), msgs

    vals_round = set(np.round(sn.dropna()).unique().tolist())
    if vals_round.issubset({0, 1}):
        msgs.append(f"'{name}' was not strictly 0/1; values were rounded to 0/1.")
        return np.round(sn).astype(float), msgs

    msgs.append(f"'{name}' is not strictly 0/1. Coercing using rule: value > 0 becomes 1, else 0.")
    return (sn.fillna(0) > 0).astype(int).astype(float), msgs

def try_parse_time_key(s: pd.Series):
    """
    Create a time key that supports sorting + clustering-by-time.
    Returns (time_key, time_type)
    """
    if np.issubdtype(s.dtype, np.datetime64):
        return s, "datetime"

    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().mean() >= 0.90:
        return sn, "numeric"

    sd = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if sd.notna().mean() >= 0.90:
        return sd, "datetime"

    return s.astype(str), "string"

def sorted_unique_times(time_key: pd.Series, time_type: str) -> list:
    vals = pd.Series(pd.unique(time_key.dropna()))
    if time_type in ("numeric", "datetime"):
        return vals.sort_values().tolist()
    return sorted(vals.tolist(), key=lambda x: str(x))

def infer_post_start(time_key: pd.Series, post: pd.Series, time_type: str):
    mask = post == 1
    if mask.sum() == 0:
        return None
    tk = pd.Series(time_key.loc[mask]).dropna()
    if tk.empty:
        return None
    if time_type in ("numeric", "datetime"):
        return tk.min()
    return sorted_unique_times(tk, "string")[0]

def build_2x2_cell_counts(df: pd.DataFrame, treated_col: str, post_col: str) -> pd.DataFrame:
    cell = df.groupby([treated_col, post_col]).size().reset_index(name="n")
    base = pd.MultiIndex.from_product([[0.0, 1.0], [0.0, 1.0]], names=[treated_col, post_col]).to_frame(index=False)
    out = base.merge(cell, on=[treated_col, post_col], how="left").fillna({"n": 0})
    out["n"] = out["n"].astype(int)
    return out

def did_design_checklist(dfm: pd.DataFrame):
    msgs = []
    status = "ok"

    if dfm.empty:
        return "fail", [("fail", "No usable rows after cleaning. Check missing values / column selections.")], build_2x2_cell_counts(
            pd.DataFrame({"treated": [], "post": []}), "treated", "post"
        )

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

def build_formula(clean_covs: list) -> str:
    base = "y ~ treated + post + treated:post"
    if clean_covs:
        base += " + " + " + ".join(clean_covs)
    return base

def compact_reg_table(res_any, alpha: float, top_terms=None) -> pd.DataFrame:
    if top_terms is None:
        top_terms = []
    params = getattr(res_any, "params", None)
    if params is None:
        return pd.DataFrame()

    if hasattr(params, "index"):
        names = list(params.index)
        coef = np.asarray(params.values, dtype=float)
        se = np.asarray(getattr(res_any, "bse").values, dtype=float)
        tvals = np.asarray(getattr(res_any, "tvalues").values, dtype=float)
        pvals = np.asarray(getattr(res_any, "pvalues").values, dtype=float)
    else:
        names = list(res_any.model.exog_names)
        coef = np.asarray(getattr(res_any, "params"), dtype=float)
        se = np.asarray(getattr(res_any, "bse"), dtype=float)
        tvals = np.asarray(getattr(res_any, "tvalues"), dtype=float)
        pvals = np.asarray(getattr(res_any, "pvalues"), dtype=float)

    try:
        ci = np.asarray(res_any.conf_int(alpha=alpha), dtype=float)
        ci_low = ci[:, 0]
        ci_high = ci[:, 1]
    except Exception:
        ci_low = np.full_like(coef, np.nan, dtype=float)
        ci_high = np.full_like(coef, np.nan, dtype=float)

    df_out = pd.DataFrame(
        {"term": names, "coef": coef, "std_err": se, "t": tvals, "p_value": pvals, "ci_low": ci_low, "ci_high": ci_high}
    )

    if top_terms:
        order = [t for t in top_terms if t in df_out["term"].values]
        rest = [t for t in df_out["term"].tolist() if t not in order]
        df_out["__order"] = df_out["term"].apply(lambda x: order.index(x) if x in order else (len(order) + rest.index(x)))
        df_out = df_out.sort_values("__order").drop(columns="__order")

    return df_out

def safe_show(fig):
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)

def term_stats_from_res(res_any, term: str, alpha: float):
    if not hasattr(res_any, "params"):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if hasattr(res_any.params, "index"):
        if term not in res_any.params.index:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        coef = float(res_any.params.loc[term])
        se = float(res_any.bse.loc[term])
        pval = float(res_any.pvalues.loc[term])
        try:
            ci = res_any.conf_int(alpha=alpha).loc[term].tolist()
            ci_low, ci_high = float(ci[0]), float(ci[1])
        except Exception:
            ci_low, ci_high = np.nan, np.nan
        return coef, se, pval, ci_low, ci_high

    # fallback (rare)
    names = list(res_any.model.exog_names)
    if term not in names:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    idx = names.index(term)
    coef = float(np.asarray(res_any.params)[idx])
    se = float(np.asarray(res_any.bse)[idx])
    pval = float(np.asarray(res_any.pvalues)[idx])
    try:
        ci = np.asarray(res_any.conf_int(alpha=alpha), dtype=float)
        ci_low, ci_high = float(ci[idx, 0]), float(ci[idx, 1])
    except Exception:
        ci_low, ci_high = np.nan, np.nan
    return coef, se, pval, ci_low, ci_high

def term_stats_two_way(res_base, df: pd.DataFrame, term: str, alpha: float):
    # Two-way cluster uses cov_cluster_2groups on base OLS fit
    if not hasattr(res_base, "params"):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if hasattr(res_base.params, "index"):
        if term not in res_base.params.index:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        idx = list(res_base.params.index).index(term)
        coef = float(res_base.params.loc[term])
    else:
        names = list(res_base.model.exog_names)
        if term not in names:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        idx = names.index(term)
        coef = float(np.asarray(res_base.params, dtype=float)[idx])

    V = cov_cluster_2groups(res_base, df["unit"], df["time_cluster"])
    se = float(np.sqrt(max(float(V[idx, idx]), 0.0)))
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

def did_2x2_table(df_model: pd.DataFrame):
    g = df_model.groupby(["treated", "post"])["y"].mean().reset_index()
    pivot = g.pivot(index="treated", columns="post", values="y")

    pivot = pivot.rename(index={0.0: "Control (treated=0)", 1.0: "Treated (treated=1)"})
    pivot = pivot.rename(columns={0.0: "Pre (post=0)", 1.0: "Post (post=1)"})

    did_val = np.nan
    ok = True
    try:
        y_tp = float(pivot.loc["Treated (treated=1)", "Post (post=1)"])
        y_tpre = float(pivot.loc["Treated (treated=1)", "Pre (post=0)"])
        y_cp = float(pivot.loc["Control (treated=0)", "Post (post=1)"])
        y_cpre = float(pivot.loc["Control (treated=0)", "Pre (post=0)"])
        did_val = (y_tp - y_tpre) - (y_cp - y_cpre)
    except Exception:
        ok = False

    return pivot, did_val, ok

def render_interpretation(effect, pval, alpha, label="DiD"):
    sig = "statistically significant" if (np.isfinite(pval) and pval < alpha) else "not statistically significant"
    ptxt = f"{pval:.4g}" if np.isfinite(pval) else "NA"
    etxt = f"{effect:.4f}" if np.isfinite(effect) else "NA"
    return (
        f"**Interpretation ({label}):** The interaction term estimates the **treatment effect on the treated** "
        f"under the parallel trends assumption. Here, the estimated effect is **{etxt}** and it is **{sig}** "
        f"at significance level α = {alpha:.2f} (p = {ptxt})."
    )

def has_all_cells(df, treat_col, post_col):
    c = df.groupby([treat_col, post_col]).size().reset_index(name="n")
    have = set(zip(c[treat_col].astype(float), c[post_col].astype(float)))
    needed = {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
    return (needed - have) == set(), c

def run_placebo_shift_sweep(df_model, cov_cols, se_mode, alpha, times, real_cutoff_time):
    """
    Sweep placebo cutoffs by shifting earlier K=1..max_k and plotting the placebo interaction.
    Returns dataframe with coef + CI by K.
    """
    rows = []
    cutoff_idx = times.index(real_cutoff_time)
    max_k = max(1, cutoff_idx)

    cov_terms = (" + " + " + ".join(cov_cols)) if cov_cols else ""

    for K in range(1, int(max_k) + 1):
        placebo_cutoff_time = times[max(0, cutoff_idx - K)]
        dfA = df_model.copy()
        dfA["placebo_post"] = (dfA["time_key"] >= placebo_cutoff_time).astype(int).astype(float)

        ok_cells, _counts = has_all_cells(dfA, "treated", "placebo_post")
        if not ok_cells:
            continue

        formulaA = f"y ~ treated + placebo_post + treated:placebo_post{cov_terms}"
        res_base = smf.ols(formula=formulaA, data=dfA).fit()

        if se_mode == "Two-way cluster (unit & time)":
            termA = "treated:placebo_post"
            # two-way cluster se for placebo term
            if hasattr(res_base.params, "index") and termA in res_base.params.index:
                idx = list(res_base.params.index).index(termA)
            else:
                names = list(res_base.model.exog_names)
                if termA not in names:
                    continue
                idx = names.index(termA)

            V = cov_cluster_2groups(res_base, dfA["unit"], dfA["time_cluster"])
            se = float(np.sqrt(max(float(V[idx, idx]), 0.0)))
            coef = float(res_base.params[termA]) if hasattr(res_base.params, "index") else float(np.asarray(res_base.params)[idx])
            df_resid = float(getattr(res_base, "df_resid", np.nan))
            tcrit = float(stats.t.ppf(1 - alpha / 2, df=df_resid)) if np.isfinite(df_resid) else np.nan
            ci_low = coef - tcrit * se if (np.isfinite(tcrit) and se > 0) else np.nan
            ci_high = coef + tcrit * se if (np.isfinite(tcrit) and se > 0) else np.nan
        else:
            _, res_rep, _ = fit_with_se(dfA, formulaA, se_mode)
            termA = "treated:placebo_post"
            if res_rep is None:
                continue
            coef, se, _p, ci_low, ci_high = term_stats_from_res(res_rep, termA, alpha)

        rows.append(
            {"K_shift": K, "placebo_cutoff": placebo_cutoff_time, "coef": float(coef), "ci_low": float(ci_low), "ci_high": float(ci_high)}
        )

    return pd.DataFrame(rows)

@dataclass
class PlaceboResult:
    ok: bool
    message: str
    se_label: str = ""
    coef: float = np.nan
    se: float = np.nan
    pval: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    formula: str = ""
    n_obs: int = 0
    term: str = ""

# =============================
# Synthetic data generator
# =============================
def generate_synthetic(seed=7, n_units=200, n_periods=10, treat_share=0.5, effect=2.0, cutoff=6):
    rng = np.random.default_rng(seed)
    units = np.arange(n_units)
    times = np.arange(n_periods)

    treated_units = rng.choice(units, size=int(n_units * treat_share), replace=False)
    treat_flag = np.isin(units, treated_units).astype(int)

    rows = []
    for u in units:
        unit_fe = rng.normal(0, 1)
        for t in times:
            time_fe = 0.2 * t
            treated = int(treat_flag[u])
            post = int(t >= cutoff)

            eps = rng.normal(0, 1)
            y = 5 + unit_fe + time_fe + 0.5 * treated + 0.3 * post
            y += effect * (treated * post)

            x1 = rng.normal(0, 1) + 0.3 * treated + 0.1 * t
            x2 = rng.normal(0, 1)

            rows.append(
                {"unit_id": u, "time": t, "treated": treated, "post": post, "y": y + eps + 0.4 * x1 - 0.2 * x2, "x1": x1, "x2": x2}
            )
    return pd.DataFrame(rows)

def numeric_like_columns(df: pd.DataFrame, min_non_na_share: float = 0.90) -> list:
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= min_non_na_share:
            cols.append(c)
    return cols

# =============================
# UI: Header + Instructions
# =============================
st.title(APP_NAME)
st.caption(APP_SUBTITLE)

with st.expander("Install / Run Instructions", expanded=False):
    st.code(
        "pip install streamlit pandas numpy statsmodels matplotlib scipy\n"
        "streamlit run app.py\n",
        language="bash",
    )

st.write("Upload a CSV or use the built-in synthetic dataset. Then select columns and run DiD + placebo diagnostics.")

# =============================
# Data input
# =============================
uploaded = st.file_uploader("Upload CSV", type=["csv"])
use_synth = st.checkbox("Use synthetic dataset (if no file uploaded)", value=(uploaded is None))

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    if use_synth:
        df_raw = generate_synthetic()
    else:
        st.info("Upload a CSV or enable the synthetic dataset checkbox.")
        st.stop()

if df_raw is None or df_raw.empty:
    st.error("Dataset is empty.")
    st.stop()

cols_all = df_raw.columns
num_candidates = numeric_like_columns(df_raw, min_non_na_share=0.90)

# =============================
# Controls: keep your model setup, but make it bulletproof
# (Always-visible main controls + synced sidebar mirror)
# =============================
def controls_ui():
    st.subheader("Model Setup")

    outcome_col = st.selectbox("Outcome (Y)", cols_all, index=default_index(cols_all, "y"), key="outcome_col")
    unit_col = st.selectbox("Unit ID", cols_all, index=default_index(cols_all, "unit_id"), key="unit_col")
    time_col = st.selectbox("Time", cols_all, index=default_index(cols_all, "time"), key="time_col")
    treated_col = st.selectbox("Treated (0/1)", cols_all, index=default_index(cols_all, "treated"), key="treated_col")
    post_col = st.selectbox("Post (0/1)", cols_all, index=default_index(cols_all, "post"), key="post_col")

    st.divider()

    covariate_options = [c for c in num_candidates if c not in {outcome_col, treated_col, post_col}]
    default_covs = [c for c in ["x1", "x2"] if c in covariate_options]
    covariates = st.multiselect(
        "Optional covariates (numeric only)",
        options=covariate_options,
        default=default_covs,
        key="covariates",
    )

    st.divider()

    se_mode = st.selectbox(
        "Standard errors",
        ["Robust (HC1)", "Cluster by unit", "Cluster by time", "Two-way cluster (unit & time)"],
        index=1,
        key="se_mode",
    )
    alpha = st.selectbox("Significance level α", options=[0.10, 0.05, 0.01], index=1, key="alpha")

    return outcome_col, unit_col, time_col, treated_col, post_col, covariates, se_mode, alpha

# Main-page controls (always visible)
with st.expander("✅ Controls (always visible): choose outcome / treated / post / covariates / SE", expanded=True):
    outcome_col, unit_col, time_col, treated_col, post_col, covariates, se_mode, alpha = controls_ui()

# Sidebar mirror (synced state)
with st.sidebar:
    st.header("Controls (Sidebar)")
    st.caption("These controls are synced with the main controls.")
    controls_ui()

# =============================
# Build modeling dataframe
# =============================
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

warn_msgs = []

dfm["y"] = coerce_numeric(dfm["y"])
dfm["treated"], msgs_t = coerce_binary01(dfm["treated_raw"], "treated")
dfm["post"], msgs_p = coerce_binary01(dfm["post_raw"], "post")
warn_msgs.extend(msgs_t)
warn_msgs.extend(msgs_p)

dfm["time_key"], time_type = try_parse_time_key(dfm["time_raw"])
dfm["time_cluster"] = dfm["time_key"]

clean_covs = []
for c in covariates:
    dfm[c] = coerce_numeric(dfm[c])
    clean_covs.append(c)

before = len(dfm)
essential = ["y", "unit", "time_key", "treated", "post"] + clean_covs
dfm = dfm.dropna(subset=essential)
dropped = before - len(dfm)
if dropped > 0:
    warn_msgs.append(f"Dropped {dropped} rows due to missing values in required columns.")

dfm["treated"] = dfm["treated"].astype(float)
dfm["post"] = dfm["post"].astype(float)

# =============================
# 1) Data Preview
# =============================
st.subheader("1) Data Preview")
st.dataframe(df_raw.head(20), use_container_width=True)

if warn_msgs:
    st.warning("Validation / Warnings:\n- " + "\n- ".join(warn_msgs))

st.divider()

# =============================
# DiD Design Checklist (before estimation)
# =============================
st.subheader("2) DiD Design Checklist")
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
    pivot_counts = cell_counts.pivot(index="treated", columns="post", values="n").reindex(index=[0.0, 1.0], columns=[0.0, 1.0])
    pivot_counts.index = ["Control (treated=0)", "Treated (treated=1)"]
    pivot_counts.columns = ["Pre (post=0)", "Post (post=1)"]
    st.dataframe(pivot_counts, use_container_width=True)

if status == "fail":
    st.info(
        "Fix guidance:\n"
        "- Ensure Treated and Post are binary (0/1) with both values present.\n"
        "- Ensure at least 2 units and 2 time periods.\n"
        "- Ensure you have observations in all four cells: treated/control × pre/post.\n"
        "- Ensure outcome and covariates are numeric (or can be coerced to numeric)."
    )
    st.stop()

st.divider()

# =============================
# 3) Main Estimate (DiD)
# =============================
st.subheader("3) Main Estimate (DiD)")

formula_main = build_formula(clean_covs)
term_main = "treated:post"

model_ok = True
res_base = None
res_rep = None
res_for_table = None
se_label = ""
did_coef = did_se = did_p = did_ciL = did_ciH = np.nan

try:
    res_base, res_rep, se_label = fit_with_se(dfm, formula_main, se_mode)
    if se_mode == "Two-way cluster (unit & time)":
        did_coef, did_se, did_p, did_ciL, did_ciH = term_stats_two_way(res_base, dfm, term_main, alpha)
        res_for_table = res_base
    else:
        did_coef, did_se, did_p, did_ciL, did_ciH = term_stats_from_res(res_rep, term_main, alpha)
        res_for_table = res_rep
except Exception as e:
    model_ok = False
    st.error(f"Model failed to run. Error: {e}")

if not model_ok:
    st.stop()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("DiD (treated×post) coef", f"{did_coef:.4f}" if np.isfinite(did_coef) else "NA")
kpi2.metric("Std. Error", f"{did_se:.4f}" if np.isfinite(did_se) else "NA")
kpi3.metric("p-value", f"{did_p:.4g}" if np.isfinite(did_p) else "NA")
kpi4.metric(f"{int((1-alpha)*100)}% CI", f"[{did_ciL:.4f}, {did_ciH:.4f}]" if np.isfinite(did_ciL) else "NA")

st.caption(f"SE type: {se_label} | Model: {formula_main}")
st.markdown(render_interpretation(did_coef, did_p, alpha, label="Main DiD"))

with st.expander("Compact regression table (main model)", expanded=False):
    try:
        reg_tbl = compact_reg_table(res_for_table, alpha=alpha, top_terms=["treated:post", "treated", "post"])
        # If two-way cluster, patch the treated:post row with our computed SE/p/CI
        if se_mode == "Two-way cluster (unit & time)" and "treated:post" in reg_tbl["term"].values:
            mask = reg_tbl["term"] == "treated:post"
            reg_tbl.loc[mask, "std_err"] = did_se
            reg_tbl.loc[mask, "p_value"] = did_p
            reg_tbl.loc[mask, "ci_low"] = did_ciL
            reg_tbl.loc[mask, "ci_high"] = did_ciH
            reg_tbl.loc[mask, "t"] = did_coef / did_se if (np.isfinite(did_coef) and np.isfinite(did_se) and did_se > 0) else np.nan
        st.dataframe(reg_tbl, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render regression table: {e}")

st.divider()

# =============================
# 4) Sanity Check (2×2 Means)
# =============================
st.subheader("4) Sanity Check (2×2 Means)")

pivot_means, did_val, did_ok = did_2x2_table(dfm)

left, right = st.columns([1.2, 1])
with left:
    st.write("**2×2 mean outcome table**")
    st.dataframe(pivot_means, use_container_width=True)
with right:
    st.write("**Manual DiD**")
    if did_ok and np.isfinite(did_val):
        st.metric("DiD from means", f"{did_val:.4f}")
        st.write(
            "This equals: (Treated Post − Treated Pre) − (Control Post − Control Pre). "
            "It should be close to the regression interaction coefficient when there are no extra covariates "
            "and the sample is balanced."
        )
    else:
        st.error("Could not compute manual DiD because one or more 2×2 cells are missing.")

st.divider()

# =============================
# 5) Visuals
# =============================
st.subheader("5) Visuals")

t_unique = sorted_unique_times(dfm["time_key"], time_type)
post_start = infer_post_start(dfm["time_key"], dfm["post"], time_type)

if post_start is None:
    st.warning("Cannot infer treatment cutoff because 'post' has no 1s.")
else:
    st.caption(f"Inferred treatment cutoff (first post period): {post_start}")

plot1, plot2 = st.columns(2)

with plot1:
    st.write("**Parallel Trends (Average Y over time)**")
    try:
        g = dfm.groupby(["time_key", "treated"])["y"].mean().reset_index()
        pt = g.pivot(index="time_key", columns="treated", values="y").reindex(index=t_unique)

        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        if 0.0 in pt.columns:
            ax.plot(pt.index, pt[0.0], marker="o", linewidth=2, label="Control (treated=0)")
        if 1.0 in pt.columns:
            ax.plot(pt.index, pt[1.0], marker="o", linewidth=2, label="Treated (treated=1)")

        if post_start is not None:
            ax.axvline(post_start, linestyle="--", linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Average outcome (Y)")
        ax.set_title("Average Y by group over time")
        ax.legend()
        ax.grid(True, alpha=0.25)
        safe_show(fig)
    except Exception as e:
        st.error(f"Parallel trends plot failed: {e}")

with plot2:
    st.write("**Outcome Distribution**")
    dist_mode = st.selectbox(
        "Compare distributions",
        options=["Treated vs Control (all periods)", "Pre vs Post (all units)", "4 groups (treated/control × pre/post)"],
        index=0,
        key="dist_mode",
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = dfm["y"].dropna()
    if y.empty:
        st.error("No valid outcome data to plot.")
    else:
        if dist_mode == "Treated vs Control (all periods)":
            y0 = dfm.loc[dfm["treated"] == 0, "y"].dropna()
            y1 = dfm.loc[dfm["treated"] == 1, "y"].dropna()
            ax.hist(y0.values, bins=30, alpha=0.6, label="Control (treated=0)")
            ax.hist(y1.values, bins=30, alpha=0.6, label="Treated (treated=1)")
            ax.set_title("Outcome distribution: Treated vs Control")

        elif dist_mode == "Pre vs Post (all units)":
            ypre = dfm.loc[dfm["post"] == 0, "y"].dropna()
            ypost = dfm.loc[dfm["post"] == 1, "y"].dropna()
            ax.hist(ypre.values, bins=30, alpha=0.6, label="Pre (post=0)")
            ax.hist(ypost.values, bins=30, alpha=0.6, label="Post (post=1)")
            ax.set_title("Outcome distribution: Pre vs Post")

        else:
            labels = [
                ("Control-Pre", (dfm["treated"] == 0) & (dfm["post"] == 0)),
                ("Control-Post", (dfm["treated"] == 0) & (dfm["post"] == 1)),
                ("Treated-Pre", (dfm["treated"] == 1) & (dfm["post"] == 0)),
                ("Treated-Post", (dfm["treated"] == 1) & (dfm["post"] == 1)),
            ]
            for lab, mask in labels:
                yy = dfm.loc[mask, "y"].dropna()
                if len(yy) > 0:
                    ax.hist(yy.values, bins=30, alpha=0.5, label=lab)
            ax.set_title("Outcome distribution: 4 groups")

        ax.set_xlabel("Outcome (Y)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.20)
        safe_show(fig)

st.divider()

# =============================
# 6) Diagnostics (Placebo / Pre-trend checks)
# =============================
st.subheader("6) Diagnostics (Placebo / Pre-trend checks)")

times = t_unique if len(t_unique) > 0 else sorted_unique_times(dfm["time_key"], time_type)

# ---- Placebo A: shift post earlier ----
st.write("### Placebo A — Fake policy date (shift post earlier)")

if post_start is None or len(times) < 3:
    st.warning("Placebo A unavailable: cannot infer post start time or too few time periods.")
    placeboA = PlaceboResult(ok=False, message="Insufficient time periods or no post==1.")
else:
    cutoff_idx = times.index(post_start)
    max_k = max(1, cutoff_idx)

    K = st.slider("Shift earlier by K time steps", min_value=1, max_value=int(max_k), value=1, step=1, key="placeboA_K")
    placebo_cutoff_time = times[max(0, cutoff_idx - K)]

    dfA = dfm.copy()
    dfA["placebo_post"] = (dfA["time_key"] >= placebo_cutoff_time).astype(int).astype(float)

    okA, countsA = has_all_cells(dfA, "treated", "placebo_post")
    st.caption(f"Real cutoff: {post_start} | Placebo cutoff (K={K}): {placebo_cutoff_time}")

    if not okA:
        st.warning("Placebo A has missing cells (treated/control × placebo pre/post).")
        placeboA = PlaceboResult(ok=False, message="Missing placebo A 2x2 cells.")
    else:
        cov_terms = (" + " + " + ".join(clean_covs)) if clean_covs else ""
        formulaA = f"y ~ treated + placebo_post + treated:placebo_post{cov_terms}"
        termA = "treated:placebo_post"
        try:
            res_baseA = smf.ols(formula=formulaA, data=dfA).fit()
            if se_mode == "Two-way cluster (unit & time)":
                coefA, seA, pA, ciLA, ciHA = term_stats_two_way(res_baseA, dfA.assign(time_cluster=dfA["time_cluster"]), termA, alpha)
                se_labelA = "Two-way cluster (unit & time)"
            else:
                _, res_repA, se_labelA = fit_with_se(dfA, formulaA, se_mode)
                coefA, seA, pA, ciLA, ciHA = term_stats_from_res(res_repA, termA, alpha)

            placeboA = PlaceboResult(
                ok=True, message="OK", se_label=se_labelA, coef=coefA, se=seA, pval=pA,
                ci_low=ciLA, ci_high=ciHA, formula=formulaA, n_obs=int(res_baseA.nobs), term=termA
            )
        except Exception as e:
            placeboA = PlaceboResult(ok=False, message=f"Model failed: {e}")

if placeboA.ok:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Placebo A interaction coef", f"{placeboA.coef:.4f}" if np.isfinite(placeboA.coef) else "NA")
    a2.metric("Std. Error", f"{placeboA.se:.4f}" if np.isfinite(placeboA.se) else "NA")
    a3.metric("p-value", f"{placeboA.pval:.4g}" if np.isfinite(placeboA.pval) else "NA")
    a4.metric(f"{int((1-alpha)*100)}% CI", f"[{placeboA.ci_low:.4f}, {placeboA.ci_high:.4f}]" if np.isfinite(placeboA.ci_low) else "NA")
    st.caption(f"SE type: {placeboA.se_label} | Model: {placeboA.formula} | N={placeboA.n_obs}")
    st.markdown(
        "**Interpretation (Placebo A):** This tests for **pre-trends** by pretending the policy started earlier. "
        "If parallel trends holds, the placebo interaction should be close to 0 and typically not significant. "
        + (f"Here it is {'significant' if (np.isfinite(placeboA.pval) and placeboA.pval < alpha) else 'not significant'} at α={alpha:.2f}.")
    )
else:
    st.info(f"Placebo A not run: {placeboA.message}")

st.write("---")

# ---- Placebo B: pre-period only fake cutoff ----
st.write("### Placebo B — Pre-period only (fake cutoff within pre)")
df_pre = dfm.loc[dfm["post"] == 0].copy()

if df_pre.empty or df_pre["treated"].nunique(dropna=True) < 2:
    st.warning("Placebo B unavailable: no pre-period data or no treated/control variation in pre.")
    placeboB = PlaceboResult(ok=False, message="Insufficient pre data.")
else:
    pre_times = sorted_unique_times(df_pre["time_key"], time_type)
    if len(pre_times) < 4:
        st.warning("Placebo B unavailable: too few distinct pre-period time points.")
        placeboB = PlaceboResult(ok=False, message="Too few pre time points.")
    else:
        inner_times = pre_times[1:-1]
        default_idx = len(inner_times) // 2
        fake_cutoff = st.selectbox(
            "Choose a fake cutoff time within the PRE period",
            options=inner_times,
            index=default_idx,
            key="placeboB_cutoff",
            help="Creates a fake 'post' inside pre to test for differential pre-trends.",
        )

        dfB = df_pre.copy()
        dfB["placebo_post_pre"] = (dfB["time_key"] >= fake_cutoff).astype(int).astype(float)

        okB, countsB = has_all_cells(dfB, "treated", "placebo_post_pre")

        if not okB:
            st.warning("Placebo B has missing cells (treated/control × placebo pre/post).")
            placeboB = PlaceboResult(ok=False, message="Missing placebo B 2x2 cells.")
        else:
            cov_terms = (" + " + " + ".join(clean_covs)) if clean_covs else ""
            formulaB = f"y ~ treated + placebo_post_pre + treated:placebo_post_pre{cov_terms}"
            termB = "treated:placebo_post_pre"
            try:
                res_baseB = smf.ols(formula=formulaB, data=dfB).fit()
                if se_mode == "Two-way cluster (unit & time)":
                    coefB, seB, pB, ciLB, ciHB = term_stats_two_way(res_baseB, dfB.assign(time_cluster=dfB["time_cluster"]), termB, alpha)
                    se_labelB = "Two-way cluster (unit & time)"
                else:
                    _, res_repB, se_labelB = fit_with_se(dfB, formulaB, se_mode)
                    coefB, seB, pB, ciLB, ciHB = term_stats_from_res(res_repB, termB, alpha)

                placeboB = PlaceboResult(
                    ok=True, message="OK", se_label=se_labelB, coef=coefB, se=seB, pval=pB,
                    ci_low=ciLB, ci_high=ciHB, formula=formulaB, n_obs=int(res_baseB.nobs), term=termB
                )
            except Exception as e:
                placeboB = PlaceboResult(ok=False, message=f"Model failed: {e}")

if placeboB.ok:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Placebo B interaction coef", f"{placeboB.coef:.4f}" if np.isfinite(placeboB.coef) else "NA")
    b2.metric("Std. Error", f"{placeboB.se:.4f}" if np.isfinite(placeboB.se) else "NA")
    b3.metric("p-value", f"{placeboB.pval:.4g}" if np.isfinite(placeboB.pval) else "NA")
    b4.metric(f"{int((1-alpha)*100)}% CI", f"[{placeboB.ci_low:.4f}, {placeboB.ci_high:.4f}]" if np.isfinite(placeboB.ci_low) else "NA")
    st.caption(f"SE type: {placeboB.se_label} | Model: {placeboB.formula} | N={placeboB.n_obs}")
    st.markdown(
        "**Interpretation (Placebo B):** Uses **only pre-treatment data** and inserts a fake cutoff. "
        "If treated and control already diverge within the pre-period, this can show up as a significant placebo effect. "
        + (f"Here it is {'significant' if (np.isfinite(placeboB.pval) and placeboB.pval < alpha) else 'not significant'} at α={alpha:.2f}.")
    )
else:
    st.info(f"Placebo B not run: {placeboB.message}")

st.divider()

# =============================
# 7) Visual Diagnostic — Main vs Placebos (Sweep)
# =============================
st.subheader("7) Visual Diagnostic — Main DiD vs Placebo Effects")

if post_start is None or len(times) < 3:
    st.info("Visual diagnostic not available (need a real post period and multiple time points).")
else:
    show_curve = st.checkbox("Show placebo shift curve (all K)", value=True, key="show_placebo_curve")
    if show_curve:
        sweep = run_placebo_shift_sweep(
            df_model=dfm,
            cov_cols=clean_covs,
            se_mode=se_mode,
            alpha=alpha,
            times=times,
            real_cutoff_time=post_start,
        )

        if sweep.empty:
            st.info("Could not compute placebo curve (many placebo cutoffs miss some 2×2 cells).")
        else:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.plot(sweep["K_shift"], sweep["coef"], marker="o", label="Placebo A (treated×placebo_post)")
            ax.fill_between(sweep["K_shift"], sweep["ci_low"], sweep["ci_high"], alpha=0.2)

            ax.axhline(did_coef, linestyle="--", linewidth=2, label="Main DiD coef")
            ax.axhline(0, linestyle=":", linewidth=2, label="0 (expected if no pre-trends)")

            ax.set_xlabel("K (placebo starts K periods earlier than real cutoff)")
            ax.set_ylabel("Interaction coefficient estimate")
            ax.set_title("Placebo shift diagnostic (compare against 0 and main DiD)")
            ax.legend()
            ax.grid(True, alpha=0.25)
            safe_show(fig)

        with st.expander("Show placebo curve data table", expanded=False):
            st.dataframe(sweep, use_container_width=True)

st.divider()

# =============================
# 8) Plain-English Summary
# =============================
st.subheader("8) Plain-English Summary")

sig = "Yes" if (np.isfinite(did_p) and did_p < alpha) else "No"
direction = (
    "increase" if (np.isfinite(did_coef) and did_coef > 0)
    else ("decrease" if (np.isfinite(did_coef) and did_coef < 0) else "no change")
)

st.markdown(
    f"""
**What the main DiD estimate means**  
- The DiD interaction term (**treated×post**) compares how the treated group changed from pre→post **relative** to how the control group changed from pre→post.  
- If the **parallel trends** assumption is reasonable, this is interpreted as the causal effect of the intervention on the treated group.

**Your result**  
- **Effect:** {did_coef:.4f}  
- **SE type:** {se_label}  
- **p-value:** {did_p:.4g}  
- **Significant at α={alpha}:** {sig}  
- **Direction:** {direction}

**How to read the placebo tests**  
- **Placebo A (shift post earlier):** Significant placebo effects **before** the real intervention can be a warning sign of pre-trends.  
- **Placebo B (pre-only fake cutoff):** Significant effects within the pre period suggest treated and control were already diverging.

**Quick rule of thumb**  
- Main DiD significant + placebo effects near 0 (not significant) ⇒ more consistent with parallel trends.  
- Placebo significant ⇒ investigate pre-trends, add controls, consider unit/time fixed effects, or reconsider identification.
"""
)

with st.expander("Dataset requirements", expanded=False):
    st.write(
        "- One row per unit-time observation.\n"
        "- `treated` is a group indicator (1 for treated units, 0 for controls).\n"
        "- `post` indicates the post-treatment periods (1 after treatment begins, 0 before).\n"
        "- Outcome Y should be numeric (or convertible to numeric).\n"
        "- Covariates optional; numeric recommended.\n"
    )

















   
  

