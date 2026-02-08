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
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
  --font-scale: {font_scale};
  --line-height: {line_height};
  --max-width: {page_width}px;
}}

/* =============================
   App base – LIGHT THEME
   ============================= */
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

/* =============================
   Headings
   ============================= */
h1, h2, h3 {{
  color: #030712 !important;
  letter-spacing: -0.02em;
}}

/* Paragraph spacing */
p, li {{
  margin-bottom: 0.6rem;
  color: #1f2937;
}}

/* =============================
   Sidebar
   ============================= */
section[data-testid="stSidebar"] {{
  background: #f9fafb;
  border-right: 1px solid #e5e7eb;
}}

section[data-testid="stSidebar"] * {{
  color: #030712 !important;
}}

/* =============================
   Inputs
   ============================= */
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {{
  background: #ffffff !important;
  border: 1px solid #d1d5db !important;
  color: #030712 !important;
  border-radius: 10px !important;
}}

/* =============================
   Tabs – readable on white
   ============================= */
div[data-testid="stTabs"] button {{
  color: #4b5563 !important;   /* inactive tabs */
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

/* =============================
   Tables / DataFrames
   ============================= */
div[data-testid="stDataFrame"] {{
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
}}

/* =============================
   Metrics
   ============================= */
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

/* =============================
   File uploader
   ============================= */
section[data-testid="stFileUploaderDropzone"] {{
  background: #f9fafb !important;
  border: 1px dashed #9ca3af !important;
  border-radius: 14px !important;
}}

</style>
""",
    unsafe_allow_html=True,
)

### Plain-English Summary

- **Estimated effect:** {coef:.3f}  
- **Statistical significance:** {"Yes" if pval < 0.05 else "No"}  

This compares how the treated group changed before vs after the intervention,
relative to how the control group changed.

Causal interpretation requires **parallel trends**.
"""
    )










   
  

