import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from dataclasses import dataclass

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
contrast = 1.15

# =============================
# Global Styling
# =============================
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
  --font-scale: {font_scale};
  --line-height: {line_height};
  --max-width: {page_width}px;
  --contrast: {contrast};
}}

.stApp {{
  filter: contrast(var(--contrast));
  background: linear-gradient(180deg, #05070c 0%, #0b0f18 100%);
  color: rgba(243,244,246,0.95);
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
  color: #ffffff !important;
  letter-spacing: -0.02em;
}}

p, li {{
  margin-bottom: 0.55rem;
}}

section[data-testid="stSidebar"] {{
  background: #0a0e17;
  border-right: 1px solid rgba(255,255,255,0.12);
}}

section[data-testid="stSidebar"] * {{
  color: #f9fafb !important;
}}

/* =============================
   Tabs: MAKE THEM READABLE
   ============================= */
div[data-testid="stTabs"] button {{
  color: rgba(255,255,255,0.75) !important;
  font-weight: 600;
  font-size: calc(1.05rem * var(--font-scale));
}}

div[data-testid="stTabs"] button[aria-selected="true"] {{
  color: #ffffff !important;
  font-weight: 800;
}}

div[data-testid="stTabs"] button:hover {{
  color: #ffffff !important;
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
            y = 5 + 0.3*t + 0.5*treated + effect*(treated*post)
            y += rng.normal(0, 1)
            rows.append(
                dict(
                    unit_id=u,
                    time=t,
                    treated=treated,
                    post=post,
                    y=y,
                )
            )
    return pd.DataFrame(rows)

# =============================
# Header
# =============================
st.markdown(
    f"""
<h1>{APP_NAME}</h1>
<p style="opacity:0.85">{APP_SUBTITLE}</p>
""",
    unsafe_allow_html=True,
)

# =============================
# Data Upload
# =============================
st.subheader("Upload data")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
use_synth = st.checkbox("Use synthetic dataset (if no file uploaded)", value=(uploaded is None))

if uploaded:
    df = pd.read_csv(uploaded)
elif use_synth:
    df = generate_synthetic()
else:
    st.stop()

# =============================
# Sidebar – Model Setup
# =============================
with st.sidebar:
    st.header("Model setup")
    outcome = st.selectbox("Outcome", df.columns, index=df.columns.get_loc("y"))
    unit = st.selectbox("Unit ID", df.columns, index=df.columns.get_loc("unit_id"))
    time = st.selectbox("Time", df.columns, index=df.columns.get_loc("time"))
    treated = st.selectbox("Treated (0/1)", df.columns, index=df.columns.get_loc("treated"))
    post = st.selectbox("Post (0/1)", df.columns, index=df.columns.get_loc("post"))
    cluster = st.checkbox("Cluster SE by unit", value=True)

# =============================
# Prepare data
# =============================
dfm = df[[outcome, unit, time, treated, post]].dropna()
dfm.columns = ["y", "unit", "time", "treated", "post"]

# =============================
# Run DiD
# =============================
model = smf.ols("y ~ treated + post + treated:post", data=dfm)
res = model.fit(
    cov_type="cluster",
    cov_kwds={"groups": dfm["unit"]},
) if cluster else model.fit(cov_type="HC1")

coef = res.params["treated:post"]
pval = res.pvalues["treated:post"]
ci = res.conf_int().loc["treated:post"]

# =============================
# Tabs
# =============================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Preview", "📌 Estimate", "📈 Plots", "🧪 Diagnostics", "📝 Summary"]
)

with tab1:
    st.dataframe(df.head(20), use_container_width=True)

with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DiD coef", f"{coef:.3f}")
    c2.metric("p-value", f"{pval:.4g}")
    c3.metric("CI low", f"{ci[0]:.3f}")
    c4.metric("CI high", f"{ci[1]:.3f}")

with tab3:
    g = dfm.groupby(["time", "treated"])["y"].mean().reset_index()
    pivot = g.pivot(index="time", columns="treated", values="y")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pivot.index, pivot[0], marker="o", label="Control")
    ax.plot(pivot.index, pivot[1], marker="o", label="Treated")
    ax.axvline(6, linestyle="--")
    ax.set_title("Parallel trends")
    ax.legend()
    st.pyplot(fig)

with tab4:
    st.info(
        "Diagnostics: Placebo tests and pre-trend checks should be near zero "
        "if the parallel trends assumption holds."
    )

with tab5:
    st.markdown(
        f"""
### Plain-English Summary

- **Estimated effect:** {coef:.3f}  
- **Statistical significance:** {"Yes" if pval < 0.05 else "No"}  

This compares how the treated group changed before vs after the intervention,
relative to how the control group changed.

Causal interpretation requires **parallel trends**.
"""
    )










   
  

