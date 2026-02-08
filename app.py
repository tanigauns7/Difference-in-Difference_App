import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

# =============================
# App Identity
# =============================
APP_NAME = "Difference-in-Differences Studio"
APP_SUBTITLE = "DiD estimator with plots + placebo diagnostics + robust / clustered SEs."

st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")

# =============================
# Readability defaults
# =============================
font_scale = 1.12
line_height = 1.60
page_width = 1400

# =============================
# Global Styling (LIGHT)
# =============================
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

:root {{
  --font-scale: {font_scale};
  --line-height: {line_height};
  --max-width: {page_width}px;
}}

.stApp {{
  background: #ffffff;
  color: #111827;
  font-family: "Plus Jakarta Sans", system-ui;
  font-size: calc(16px * var(--font-scale));
  line-height: var(--line-height);
}}

.block-container {{
  max-width: var(--max-width);
  padding-top: 1.2rem;
}}

header, footer {{
  visibility: hidden;
}}

h1, h2, h3 {{
  color: #030712 !important;
  letter-spacing: -0.02em;
}}

p, li {{
  margin-bottom: 0.6rem;
  color: #1f2937;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: #f9fafb;
  border-right: 1px solid #e5e7eb;
}}
section[data-testid="stSidebar"] * {{
  color: #030712 !important;
}}

/* Inputs */
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {{
  background: #ffffff !important;
  border: 1px solid #d1d5db !important;
  color: #030712 !important;
  border-radius: 10px !important;
}}

/* Tabs */
div[data-testid="stTabs"] button {{
  color: #4b5563 !important;
  font-weight: 650;
  font-size: calc(1.05rem * var(--font-scale));
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
  color: #000000 !important;
  font-weight: 800;
  border-bottom: 3px solid #111827;
}}
div[data-testid="stTabs"] button:hover {{
  color: #000000 !important;
}}

/* DataFrames */
div[data-testid="stDataFrame"] {{
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
}}

/* Metrics */
div[data-testid="stMetric"] {{
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 14px;
}}
div[data-testid="stMetricValue"] {{
  color: #000000 !important;
  font-weight: 850;
}}
div[data-testid="stMetricLabel"] {{
  color: #374151 !important;
}}

/* File uploader */
section[data-testid="stFileUploaderDropzone"] {{
  background: #f9fafb !important;
  border: 1px dashed #9ca3af !important;
  border-radius: 14px !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# Helpers
# =============================
def default_index(cols: pd.Index, preferred: str) -> int:
    return int(cols.get_loc(preferred)) if preferred in cols else 0

def coerce_binary01(s: pd.Series) -> pd.Series:
    sn = pd.to_numeric(s, errors="coerce")
    vals = set(sn.dropna().unique().tolist())
    if vals.issubset({0, 1}):
        return sn.astype(float)
    return (sn.fillna(0) > 0).astype(int).astype(float)

def sorted_unique_times(dfm: pd.DataFrame) -> list:
    t = dfm["time"].copy()
    tn = pd.to_numeric(t, errors="coerce")
    if tn.notna().mean() >= 0.9:
        return sorted(pd.unique(tn.dropna()).tolist())
    td = pd.to_datetime(t, errors="coerce")
    if td.notna().mean() >= 0.9:
        return sorted(pd.unique(td.dropna()).tolist())
    return sorted(pd.unique(t.dropna()).tolist(), key=lambda x: str(x))

def infer_post_start(dfm: pd.DataFrame):
    post_rows = dfm.loc[dfm["post"] == 1]
    if post_rows.empty:
        return None
    times = post_rows["time"]
    tn = pd.to_numeric(times, errors="coerce")
    if tn.notna().mean() >= 0.9:
        return tn.min()
    td = pd.to_datetime(times, errors="coerce")
    if td.notna().mean() >= 0.9:
        return td.min()
    return sorted_unique_times(dfm.loc[dfm["post"] == 1])[0]

def has_all_cells(df, treat_col, post_col) -> bool:
    c = df.groupby([treat_col, post_col]).size().reset_index(name="n")
    have = set(zip(c[treat_col].astype(float), c[post_col].astype(float)))
    needed = {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
    return (needed - have) == set()

def term_in_results(res, term: str) -> bool:
    return hasattr(res, "params") and term in res.params.index

def robust_term_stats(res, term: str, alpha: float):
    coef = float(res.params.get(term, np.nan))
    se = float(res.bse.get(term, np.nan))
    pval = float(res.pvalues.get(term, np.nan))
    ci = res.conf_int(alpha=alpha).loc[term].tolist()
    return coef, se, pval, float(ci[0]), float(ci[1])

def twoway_term_stats(res_base, term: str, alpha: float, g1, g2):
    V = cov_cluster_2groups(res_base, g1, g2)
    idx = list(res_base.params.index).index(term)
    coef = float(res_base.params[term])
    se = float(np.sqrt(V[idx, idx]))
    tval = coef / se if se > 0 else np.nan

    df_resid = float(getattr(res_base, "df_resid", np.nan))
    pval = float(2 * stats.t.sf(np.abs(tval), df=df_resid)) if np.isfinite(df_resid) else np.nan
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=df_resid)) if np.isfinite(df_resid) else np.nan
    ci_low = coef - tcrit * se if np.isfinite(tcrit) else np.nan
    ci_high = coef + tcrit * se if np.isfinite(tcrit) else np.nan
    return coef, se, pval, float(ci_low), float(ci_high)

def fit_with_se(df, formula: str, se_mode: str, alpha: float):
    model = smf.ols(formula, data=df)
    res_base = model.fit()

    if se_mode == "Robust (HC1)":
        res_rep = res_base.get_robustcov_results(cov_type="HC1")
        return res_base, res_rep, "Robust (HC1)"

    if se_mode == "Cluster: unit":
        res_rep = res_base.get_robustcov_results(cov_type="cluster", groups=df["unit"])
        return res_base, res_rep, "Cluster-robust (by unit)"

    if se_mode == "Cluster: time":
        res_rep = res_base.get_robustcov_results(cov_type="cluster", groups=df["time"])
        return res_base, res_rep, "Cluster-robust (by time)"

    if se_mode == "Two-way: unit & time":
        return res_base, None, "Two-way cluster (unit & time)"

    res_rep = res_base.get_robustcov_results(cov_type="HC1")
    return res_base, res_rep, "Robust (HC1)"

# =============================
# DiD Design Checklist (NEW)
# =============================
def did_design_checklist(dfm: pd.DataFrame):
    msgs = []
    status = "ok"

    n = len(dfm)
    if n < 30:
        msgs.append(("warn", f"Only {n} usable rows after cleaning (results may be unstable)."))
        status = "warn"

    if dfm["treated"].nunique(dropna=True) < 2:
        msgs.append(("fail", "Treated has no variation (need both 0 and 1)."))
        status = "fail"

    if dfm["post"].nunique(dropna=True) < 2:
        msgs.append(("fail", "Post has no variation (need both 0 and 1)."))
        status = "fail"

    if dfm["unit"].nunique(dropna=True) < 2:
        msgs.append(("fail", "Too few units (need ≥ 2 units)."))
        status = "fail"

    if dfm["time"].nunique(dropna=True) < 2:
        msgs.append(("fail", "Too few time periods (need ≥ 2 time periods)."))
        status = "fail"

    cell_counts = dfm.groupby(["treated", "post"]).size().reset_index(name="n")
    have = set(zip(cell_counts["treated"].astype(float), cell_counts["post"].astype(float)))
    needed = {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
    missing = needed - have

    if missing:
        msgs.append(("fail", f"Missing 2×2 cells: {sorted(list(missing))}. Need treated/control × pre/post."))
        status = "fail"
    else:
        if (cell_counts["n"] < 5).any():
            msgs.append(("warn", "Some 2×2 cells have < 5 observations (SEs can be unreliable)."))
            if status != "fail":
                status = "warn"

    return status, msgs, cell_counts

# =============================
# Synthetic data generator
# =============================
def generate_synthetic(seed=7, n_units=200, n_periods=10, effect=2.0):
    rng = np.random.default_rng(seed)
    cutoff = 6
    rows = []
    for u in range(n_units):
        treated = int(u < n_units / 2)
        for t in range(n_periods):
            post = int(t >= cutoff)
            y = 5 + 0.25 * t + 0.5 * treated + effect * (treated * post) + rng.normal(0, 1)
            rows.append({"unit_id": u, "time": t, "treated": treated, "post": post, "y": y})
    return pd.DataFrame(rows)

# =============================
# Header
# =============================
st.markdown(f"# {APP_NAME}")
st.caption(APP_SUBTITLE)

# =============================
# Upload / Use synthetic
# =============================
st.subheader("Upload data")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
use_synth = st.checkbox("Use synthetic dataset (if no file uploaded)", value=(uploaded is None))

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
else:
    if use_synth:
        df_raw = generate_synthetic()
    else:
        st.info("Upload a CSV or enable the synthetic dataset checkbox.")
        st.stop()

if df_raw.empty:
    st.error("Dataset is empty.")
    st.stop()

# =============================
# Sidebar – Model Setup
# =============================
with st.sidebar:
    st.header("Model setup")
    cols = df_raw.columns

    outcome_col = st.selectbox("Outcome (Y)", cols, index=default_index(cols, "y"))
    unit_col = st.selectbox("Unit ID", cols, index=default_index(cols, "unit_id"))
    time_col = st.selectbox("Time", cols, index=default_index(cols, "time"))
    treated_col = st.selectbox("Treated (0/1)", cols, index=default_index(cols, "treated"))
    post_col = st.selectbox("Post (0/1)", cols, index=default_index(cols, "post"))

    se_mode = st.selectbox(
        "Standard errors",
        [
            "Robust (HC1)",
            "Cluster: unit",
            "Cluster: time",
            "Two-way: unit & time",
        ],
        index=1,
    )

    alpha = st.selectbox("Significance level α", [0.10, 0.05, 0.01], index=1)

# =============================
# Build modeling DF
# =============================
dfm = df_raw[[outcome_col, unit_col, time_col, treated_col, post_col]].copy()
dfm = dfm.dropna()
dfm.columns = ["y", "unit", "time", "treated", "post"]

dfm["y"] = pd.to_numeric(dfm["y"], errors="coerce")
dfm["treated"] = coerce_binary01(dfm["treated"])
dfm["post"] = coerce_binary01(dfm["post"])
dfm = dfm.dropna(subset=["y", "unit", "time", "treated", "post"])

# =============================
# DiD design checklist panel (NEW)
# =============================
st.subheader("DiD design checklist")
status, msgs, cell_counts = did_design_checklist(dfm)

if status == "ok":
    st.success("✅ Dataset looks DiD-ready.")
elif status == "warn":
    st.warning("⚠️ Dataset is usable, but results may be unstable.")
else:
    st.error("❌ Dataset is NOT DiD-ready. Fix the issues below before running the model.")

for level, text in msgs:
    st.markdown(f"- {'❌' if level=='fail' else '⚠️'} {text}")

with st.expander("Show 2×2 cell counts (treated/control × pre/post)", expanded=False):
    st.dataframe(cell_counts, use_container_width=True)

# Stop before model if impossible
if status == "fail":
    st.stop()

# =============================
# Fit main DiD
# =============================
formula_main = "y ~ treated + post + treated:post"
term_main = "treated:post"

try:
    res_base, res_rep, se_label = fit_with_se(dfm, formula_main, se_mode, alpha=alpha)

    if se_mode == "Two-way: unit & time":
        if not term_in_results(res_base, term_main):
            raise ValueError("Key term not found in model.")
        coef, se, pval, ci_low, ci_high = twoway_term_stats(
            res_base, term_main, alpha, dfm["unit"], dfm["time"]
        )
    else:
        coef, se, pval, ci_low, ci_high = robust_term_stats(res_rep, term_main, alpha)

except Exception as e:
    st.error(f"Model failed to run: {e}")
    st.stop()

# =============================
# Tabs
# =============================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Preview", "📌 Estimate", "📈 Plots", "🧪 Placebos", "📝 Summary"]
)

with tab1:
    st.subheader("Data Preview")
    st.dataframe(df_raw.head(25), use_container_width=True)

with tab2:
    st.subheader("Main Estimate (DiD)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DiD coef (treated×post)", f"{coef:.4f}")
    c2.metric("Std. Error", f"{se:.4f}")
    c3.metric("p-value", f"{pval:.4g}")
    c4.metric(f"{int((1-alpha)*100)}% CI", f"[{ci_low:.4f}, {ci_high:.4f}]")
    st.caption(f"SE type: {se_label} | Model: {formula_main}")

with tab3:
    st.subheader("Parallel Trends Plot (Average Y over time)")
    try:
        g = dfm.groupby(["time", "treated"])["y"].mean().reset_index()
        pt = g.pivot(index="time", columns="treated", values="y").sort_index()

        fig, ax = plt.subplots(figsize=(8, 4))
        if 0.0 in pt.columns:
            ax.plot(pt.index, pt[0.0], marker="o", label="Control (treated=0)")
        if 1.0 in pt.columns:
            ax.plot(pt.index, pt[1.0], marker="o", label="Treated (treated=1)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Average outcome (Y)")
        ax.set_title("Average Y by group over time")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"Plot failed: {e}")

with tab4:
    st.subheader("Placebo tests")

    st.markdown(
        """
**How to interpret:**
- Placebo effects should be near **0** and **not significant**.
- If placebos are significant, your result may reflect **pre-trends** (not causal).
"""
    )

    # ---------- Placebo A: shift earlier ----------
    st.markdown("### Placebo A — Fake policy date (shift post earlier)")
    post_start = infer_post_start(dfm)
    times = sorted_unique_times(dfm)

    if post_start is None or len(times) < 3:
        st.info("Placebo A unavailable (need a real post period and multiple time points).")
    else:
        try:
            cutoff_idx = times.index(post_start)
        except ValueError:
            cutoff_idx = max(1, len(times) // 2)

        max_k = max(1, cutoff_idx)
        K = st.slider("Shift earlier by K time steps", min_value=1, max_value=int(max_k), value=1, step=1)
        placebo_cutoff = times[max(0, cutoff_idx - K)]

        dfA = dfm.copy()
        dfA["placebo_post"] = (dfA["time"] >= placebo_cutoff).astype(int).astype(float)

        if not has_all_cells(dfA, "treated", "placebo_post"):
            st.warning("Placebo A missing some 2×2 cells (treated/control × placebo pre/post).")
        else:
            formulaA = "y ~ treated + placebo_post + treated:placebo_post"
            termA = "treated:placebo_post"

            try:
                A_base, A_rep, A_se_label = fit_with_se(dfA, formulaA, se_mode, alpha=alpha)
                if se_mode == "Two-way: unit & time":
                    A_coef, A_se, A_p, A_ciL, A_ciH = twoway_term_stats(A_base, termA, alpha, dfA["unit"], dfA["time"])
                else:
                    A_coef, A_se, A_p, A_ciL, A_ciH = robust_term_stats(A_rep, termA, alpha)

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Placebo A coef", f"{A_coef:.4f}")
                a2.metric("Std. Error", f"{A_se:.4f}")
                a3.metric("p-value", f"{A_p:.4g}")
                a4.metric(f"{int((1-alpha)*100)}% CI", f"[{A_ciL:.4f}, {A_ciH:.4f}]")
                st.caption(f"Real cutoff: {post_start} | Placebo cutoff: {placebo_cutoff} | SE: {A_se_label}")

            except Exception as e:
                st.error(f"Placebo A failed: {e}")

    st.divider()

    # ---------- Placebo B: pre-only fake cutoff ----------
    st.markdown("### Placebo B — Pre-period only fake cutoff (tests differential pre-trends)")
    df_pre = dfm.loc[dfm["post"] == 0].copy()

    if df_pre.empty or df_pre["treated"].nunique() < 2:
        st.info("Placebo B unavailable (need pre data with both treated and control).")
    else:
        pre_times = sorted_unique_times(df_pre)
        if len(pre_times) < 4:
            st.info("Placebo B unavailable (need ≥ 4 distinct pre time points).")
        else:
            inner = pre_times[1:-1]
            fake_cutoff = st.selectbox("Choose fake cutoff within PRE", options=inner, index=len(inner)//2)

            dfB = df_pre.copy()
            dfB["placebo_post_pre"] = (dfB["time"] >= fake_cutoff).astype(int).astype(float)

            if not has_all_cells(dfB, "treated", "placebo_post_pre"):
                st.warning("Placebo B missing some 2×2 cells (treated/control × placebo pre/post).")
            else:
                formulaB = "y ~ treated + placebo_post_pre + treated:placebo_post_pre"
                termB = "treated:placebo_post_pre"

                try:
                    B_base, B_rep, B_se_label = fit_with_se(dfB, formulaB, se_mode, alpha=alpha)
                    if se_mode == "Two-way: unit & time":
                        B_coef, B_se, B_p, B_ciL, B_ciH = twoway_term_stats(B_base, termB, alpha, dfB["unit"], dfB["time"])
                    else:
                        B_coef, B_se, B_p, B_ciL, B_ciH = robust_term_stats(B_rep, termB, alpha)

                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Placebo B coef", f"{B_coef:.4f}")
                    b2.metric("Std. Error", f"{B_se:.4f}")
                    b3.metric("p-value", f"{B_p:.4g}")
                    b4.metric(f"{int((1-alpha)*100)}% CI", f"[{B_ciL:.4f}, {B_ciH:.4f}]")
                    st.caption(f"Fake cutoff (within pre): {fake_cutoff} | SE: {B_se_label}")

                except Exception as e:
                    st.error(f"Placebo B failed: {e}")

with tab5:
    st.subheader("Plain-English Summary")

    sig = "Yes" if (np.isfinite(pval) and pval < 0.05) else "No"
    direction = "increases" if coef > 0 else ("decreases" if coef < 0 else "does not change")

    st.markdown(
        f"""
- **Estimated effect (treated×post):** **{coef:.4f}**
- **p-value:** **{pval:.4g}**
- **Statistically significant (α=0.05):** **{sig}**
- **SE type:** **{se_label}**

**Meaning:**  
After the intervention starts, the treated group’s outcome **{direction}** relative to the control group (net of general pre→post changes).

**When it’s causal:**  
Only if treated and control would have followed **parallel trends** without the intervention.

**Red flags:**  
If placebo tests are significant, your estimate may be capturing **pre-trends** rather than the treatment effect.
"""
    )













   
  

