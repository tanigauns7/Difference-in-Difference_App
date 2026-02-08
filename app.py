import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# =============================
# App Identity
# =============================
APP_NAME = "Difference-in-Differences Studio"
APP_SUBTITLE = "Difference-in-Differences estimator with diagnostics, plots, and placebo checks."

st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")

# =============================
# Readability defaults (NO UI CONTROLS)
# =============================
font_scale = 1.15
line_height = 1.60
page_width = 1350

# =============================
# Global Styling (LIGHT THEME)
# =============================
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

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
  padding-top: 1.4rem;
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
  font-weight: 600;
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
  font-weight: 800;
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
# Synthetic data generator
# =============================
def generate_synthetic(seed=7, n_units=200, n_periods=10, effect=2.0):
    rng = np.random.default_rng(seed)
    rows = []
    cutoff = 6

    for u in range(n_units):
        treated = int(u < n_units / 2)
        for t in range(n_periods):
            post = int(t >= cutoff)
            y = 5 + 0.3 * t + 0.5 * treated + effect * (treated * post)
            y += rng.normal(0, 1)
            rows.append(
                {
                    "unit_id": u,
                    "time": t,
                    "treated": treated,
                    "post": post,
                    "y": y,
                }
            )
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
    df = pd.read_csv(uploaded)
else:
    if use_synth:
        df = generate_synthetic()
    else:
        st.info("Upload a CSV or enable the synthetic dataset checkbox.")
        st.stop()

if df.empty:
    st.error("Dataset is empty.")
    st.stop()

# =============================
# Robust default selector
# =============================
def default_index(cols: pd.Index, preferred: str) -> int:
    return int(cols.get_loc(preferred)) if preferred in cols else 0

# =============================
# Sidebar – Model Setup
# =============================
with st.sidebar:
    st.header("Model setup")
    cols = df.columns

    outcome = st.selectbox("Outcome (Y)", cols, index=default_index(cols, "y"))
    unit = st.selectbox("Unit ID", cols, index=default_index(cols, "unit_id"))
    time = st.selectbox("Time", cols, index=default_index(cols, "time"))
    treated = st.selectbox("Treated (0/1)", cols, index=default_index(cols, "treated"))
    post = st.selectbox("Post (0/1)", cols, index=default_index(cols, "post"))

    cluster = st.checkbox("Cluster SE by unit", value=True)

# =============================
# Prepare modeling data
# =============================
dfm = df[[outcome, unit, time, treated, post]].copy()
dfm = dfm.dropna()
dfm.columns = ["y", "unit", "time", "treated", "post"]

# coerce to numeric for safety
dfm["y"] = pd.to_numeric(dfm["y"], errors="coerce")
dfm["treated"] = pd.to_numeric(dfm["treated"], errors="coerce")
dfm["post"] = pd.to_numeric(dfm["post"], errors="coerce")
dfm = dfm.dropna(subset=["y", "treated", "post", "unit", "time"])

if len(dfm) < 10:
    st.warning("Very few observations after cleaning. Results may be unstable.")

# =============================
# Fit DiD
# =============================
try:
    model = smf.ols("y ~ treated + post + treated:post", data=dfm)
    if cluster:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": dfm["unit"]})
        se_type = "Cluster-robust (by unit)"
    else:
        res = model.fit(cov_type="HC1")
        se_type = "Robust (HC1)"

    coef = float(res.params.get("treated:post", np.nan))
    pval = float(res.pvalues.get("treated:post", np.nan))
    ci = res.conf_int().loc["treated:post"].tolist()
    ci_low, ci_high = float(ci[0]), float(ci[1])
except Exception as e:
    st.error(f"Model failed to run: {e}")
    st.stop()

# =============================
# Tabs
# =============================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Preview", "📌 Estimate", "📈 Plots", "🧪 Diagnostics", "📝 Summary"]
)

with tab1:
    st.subheader("Data Preview")
    st.dataframe(df.head(25), use_container_width=True)

with tab2:
    st.subheader("Main Estimate (DiD)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DiD coef (treated×post)", f"{coef:.3f}")
    c2.metric("p-value", f"{pval:.4g}")
    c3.metric("CI low", f"{ci_low:.3f}")
    c4.metric("CI high", f"{ci_high:.3f}")
    st.caption(f"SE type: {se_type} | Model: y ~ treated + post + treated:post")

with tab3:
    st.subheader("Parallel Trends Plot (Average Y over time)")
    try:
        g = dfm.groupby(["time", "treated"])["y"].mean().reset_index()
        pivot = g.pivot(index="time", columns="treated", values="y").sort_index()

        fig, ax = plt.subplots(figsize=(8, 4))
        if 0 in pivot.columns:
            ax.plot(pivot.index, pivot[0], marker="o", label="Control (treated=0)")
        if 1 in pivot.columns:
            ax.plot(pivot.index, pivot[1], marker="o", label="Treated (treated=1)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Average outcome (Y)")
        ax.set_title("Average Y by group over time")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"Plot failed: {e}")

with tab4:
    st.subheader("Diagnostics (Basic)")
    st.markdown(
        """
- **Parallel trends:** In the pre-period, treated and control should move similarly.  
- **If pre-trends differ:** your DiD estimate may reflect trend differences (not causal).
"""
    )

with tab5:
    st.subheader("Plain-English Summary")
    st.markdown(
        f"""
- **Estimated effect (treated×post):** **{coef:.3f}**  
- **p-value:** **{pval:.4g}**  
- **Statistically significant (α = 0.05):** **{"Yes" if (np.isfinite(pval) and pval < 0.05) else "No"}**

This compares how the treated group changed from **before → after** the intervention,
and subtracts how the control group changed over the same period.

For a **causal** interpretation, the treated and control groups must follow **parallel trends**
in the absence of treatment.
"""
    )











   
  

