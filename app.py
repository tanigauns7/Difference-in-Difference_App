# app.py
import io
import base64
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.formula.api as smf


# =========================================================
# App Config + PURE WHITE UI
# =========================================================
APP_NAME = "Difference in Differenece  Analysis App"
APP_ICON = "📊"

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide")

st.markdown(
    """
<style>
/* Force pure white background everywhere */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
  background: #FFFFFF !important;
  color: #111827 !important;
}
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { color: #111827 !important; }
p, li, label, span, div { color: #111827 !important; }

/* Sidebar polish (still white) */
section[data-testid="stSidebar"] {
  border-right: 1px solid rgba(17,24,39,0.08);
}

/* Cards / tables */
div[data-testid="stMetric"] {
  background: #FFFFFF;
  border: 1px solid rgba(17,24,39,0.10);
  border-radius: 14px;
  padding: 12px 14px;
}
div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(17,24,39,0.10);
}

/* Buttons subtle */
.stDownloadButton button, .stButton button {
  border-radius: 12px !important;
}

/* Anchor offset for in-page jumps (prevents header overlap) */
.anchor {
  display: block;
  position: relative;
  top: -75px;
  visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Utilities
# =========================================================
def anchor(id_: str):
    st.markdown(f'<span class="anchor" id="{id_}"></span>', unsafe_allow_html=True)


def is_binary01(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    vals = set(s.dropna().unique().tolist())
    return vals.issubset({0, 1})


def coerce_binary01(series: pd.Series, name: str) -> Tuple[pd.Series, List[str]]:
    msgs = []
    s = series.copy()

    if s.dtype == bool:
        return s.astype(int), msgs

    sn = pd.to_numeric(s, errors="coerce")
    if sn.isna().any():
        msgs.append(f"Column '{name}': some values could not be parsed as numeric (became missing).")

    vals = set(sn.dropna().unique().tolist())
    if vals.issubset({0, 1}):
        return sn.astype(int), msgs

    vals_round = set(np.round(sn.dropna()).unique().tolist())
    if vals_round.issubset({0, 1}):
        msgs.append(f"Column '{name}': values were not strictly 0/1; rounded to 0/1.")
        return np.round(sn).astype(int), msgs

    msgs.append(f"Column '{name}': not strictly 0/1; coerced using rule (value > 0 → 1 else 0).")
    return (sn.fillna(0) > 0).astype(int), msgs


def try_parse_time(s: pd.Series) -> Tuple[pd.Series, str]:
    """
    Returns (parsed_series, kind) where kind in {"datetime","numeric","other"}.
    """
    if np.issubdtype(s.dtype, np.datetime64):
        return s, "datetime"

    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        if dt.notna().mean() >= 0.8:
            return dt, "datetime"

    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.8:
        return num, "numeric"

    return s, "other"


def ensure_long_panel(df: pd.DataFrame, unit_col: str, time_col: str) -> Tuple[bool, str]:
    # Long format panel means: multiple rows per unit across time (usually).
    # We'll gently warn if each unit appears only once or time is missing variation.
    if unit_col not in df.columns or time_col not in df.columns:
        return False, "Missing unit_id or time column."

    units = df[unit_col].nunique(dropna=True)
    times = df[time_col].nunique(dropna=True)
    if units <= 1 or times <= 1:
        return False, "Need variation in both unit_id and time (more than 1 unique value each)."

    # If every unit appears exactly once, likely cross-sectional
    counts = df.groupby(unit_col)[time_col].nunique(dropna=True)
    if (counts <= 1).mean() > 0.8:
        return False, "Most units only appear in one time period (may not be panel / long format)."

    return True, "Looks like panel (long format)."


def compute_summary_stats(df: pd.DataFrame, y: str, treated: str, post: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def stats(s: pd.Series) -> Dict[str, float]:
        return {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "min": float(s.min()),
            "max": float(s.max()),
            "N": int(s.shape[0]),
        }

    rows = []

    # Overall
    rows.append({"group": "Overall", **stats(df[y].dropna())})

    # Treated vs control
    for g, label in [(0, "Control (treated=0)"), (1, "Treated (treated=1)")]:
        s = df.loc[df[treated] == g, y].dropna()
        rows.append({"group": label, **stats(s)})

    # Pre vs post
    for g, label in [(0, "Pre (post=0)"), (1, "Post (post=1)")]:
        s = df.loc[df[post] == g, y].dropna()
        rows.append({"group": label, **stats(s)})

    # 2x2 cells
    for tval, plabel in [(0, "Control"), (1, "Treated")]:
        for pval, tlabel in [(0, "Pre"), (1, "Post")]:
            s = df.loc[(df[treated] == tval) & (df[post] == pval), y].dropna()
            rows.append({"group": f"{plabel} × {tlabel}", **stats(s)})

    table = pd.DataFrame(rows)

    # 2x2 mean pivot for quick reading
    means = df.groupby([treated, post])[y].mean().reset_index()
    pivot = means.pivot(index=treated, columns=post, values=y).rename(
        index={0: "Control", 1: "Treated"},
        columns={0: "Pre", 1: "Post"},
    )
    return table, pivot


def did_from_2x2(pivot: pd.DataFrame) -> Optional[float]:
    try:
        did_val = (pivot.loc["Treated", "Post"] - pivot.loc["Treated", "Pre"]) - (
            pivot.loc["Control", "Post"] - pivot.loc["Control", "Pre"]
        )
        return float(did_val)
    except Exception:
        return None


@dataclass
class ModelResult:
    ok: bool
    message: str
    formula: str = ""
    se_type: str = ""
    coef: float = np.nan
    se: float = np.nan
    pval: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    nobs: int = 0
    reg_table: Optional[pd.DataFrame] = None


def compact_reg_table(res, alpha: float) -> pd.DataFrame:
    ci = res.conf_int(alpha=alpha)
    out = pd.DataFrame(
        {
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.bse.values,
            "t": res.tvalues.values,
            "p_value": res.pvalues.values,
            f"CI_{int((1-alpha)*100)}%_low": ci[0].values,
            f"CI_{int((1-alpha)*100)}%_high": ci[1].values,
        }
    )
    return out


def fit_did(
    df: pd.DataFrame,
    y: str,
    treated: str,
    post: str,
    unit_id: str,
    time_key: str,
    covariates: List[str],
    alpha: float,
    cluster: bool,
    twfe: bool,
    interaction_term: str = None,
) -> ModelResult:
    """
    Main DiD:
      y ~ treated + post + treated:post (+ covariates) (+ unit FE + time FE)
    """
    try:
        cov_terms = " + " + " + ".join(covariates) if covariates else ""
        base = f"{treated} + {post} + {treated}:{post}{cov_terms}"

        if twfe:
            formula = f"{y} ~ {base} + C({unit_id}) + C({time_key})"
            label = "Two-way fixed effects (unit FE + time FE)"
        else:
            formula = f"{y} ~ {base}"
            label = "Simple DiD (no fixed effects)"

        model = smf.ols(formula=formula, data=df)

        if cluster:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": df[unit_id]})
            se_type = "Cluster-robust SE (clustered by unit_id)"
        else:
            res = model.fit(cov_type="HC1")
            se_type = "Robust SE (HC1)"

        term = interaction_term if interaction_term else f"{treated}:{post}"
        if term not in res.params.index:
            return ModelResult(False, f"Could not find interaction term '{term}' in regression output.", formula=formula)

        coef = float(res.params[term])
        se = float(res.bse[term])
        pval = float(res.pvalues[term])
        ci_low, ci_high = res.conf_int(alpha=alpha).loc[term].tolist()

        regtab = compact_reg_table(res, alpha=alpha)

        return ModelResult(
            ok=True,
            message=label,
            formula=formula,
            se_type=se_type,
            coef=coef,
            se=se,
            pval=pval,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            nobs=int(res.nobs),
            reg_table=regtab,
        )
    except Exception as e:
        return ModelResult(False, f"Model failed: {e}")


def render_interpretation_beta3(beta3: float, alpha: float, pval: float) -> str:
    sig = "statistically significant" if (pval < alpha) else "not statistically significant"
    return (
        f"**Plain-English interpretation:** The DiD interaction (**treated × post**) estimates how much the **treated group** "
        f"changed after the intervention **relative** to how the control group changed. "
        f"Here, **β₃ = {beta3:.4f}**, which is **{sig}** at α = {alpha:.2f} (p = {pval:.4g})."
    )


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    return buf.getvalue()


def make_time_key(df: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Create a clean time key column for FE and plotting:
      - If datetime: use datetime + also a string key for C()
      - If numeric: use numeric + also string key
    """
    parsed, kind = try_parse_time(df[time_col])
    out = df.copy()
    out["_time_parsed"] = parsed

    if kind == "datetime":
        out["_time_key"] = pd.to_datetime(out["_time_parsed"]).dt.strftime("%Y-%m-%d")
    elif kind == "numeric":
        out["_time_key"] = pd.to_numeric(out["_time_parsed"], errors="coerce").astype("Int64").astype(str)
    else:
        # fallback
        out["_time_key"] = out[time_col].astype(str)

    return out, kind, "_time_key"


def sorted_time_for_plot(df: pd.DataFrame, time_kind: str) -> List:
    t = df["_time_parsed"].copy()
    if time_kind == "datetime":
        return sorted(pd.to_datetime(t.dropna()).unique().tolist())
    if time_kind == "numeric":
        return sorted(pd.to_numeric(t.dropna(), errors="coerce").dropna().unique().tolist())
    return sorted(t.dropna().astype(str).unique().tolist())


def infer_intervention_point(df: pd.DataFrame, post_col: str, time_kind: str):
    post_times = df.loc[df[post_col] == 1, "_time_parsed"]
    if post_times.dropna().empty:
        return None
    if time_kind == "datetime":
        return pd.to_datetime(post_times).min()
    if time_kind == "numeric":
        return pd.to_numeric(post_times, errors="coerce").min()
    # fallback string-ish
    return post_times.dropna().astype(str).min()


def event_study(
    df: pd.DataFrame,
    y: str,
    treated: str,
    unit_id: str,
    time_key: str,
    time_kind: str,
    intervention_time,
    covariates: List[str],
    alpha: float,
    cluster: bool,
    max_leads_lags: int = 6,
) -> Tuple[bool, str, Optional[pd.DataFrame], Optional[plt.Figure]]:
    """
    Single treatment date event study (assumes treated group adopts at intervention_time; control never treated).
    Uses relative time bins and TWFE:
      y ~ sum_{k != -1} 1[rel_time=k]*treated + C(unit) + C(time) (+ covariates)
    Baseline omitted: k = -1 (one period before intervention).
    """
    try:
        # Build an ordered time index
        times = sorted_time_for_plot(df, time_kind)
        if intervention_time not in times:
            # try match for string keys edge cases
            return False, "Cannot build event study: intervention time not found in sorted time values.", None, None

        t_to_idx = {t: i for i, t in enumerate(times)}
        cutoff_idx = t_to_idx[intervention_time]

        df_es = df.copy()
        df_es["_t_idx"] = df_es["_time_parsed"].map(t_to_idx)
        df_es["_rel"] = df_es["_t_idx"] - cutoff_idx

        # Need at least 3 periods total to be meaningful
        if len(times) < 3:
            return False, "Event study needs at least 3 time periods.", None, None

        # Cap rel time into bins (so plots don’t explode)
        df_es["_rel_cap"] = df_es["_rel"].clip(-max_leads_lags, max_leads_lags)

        # Check we have some pre and post
        if (df_es["_rel"] < 0).sum() == 0 or (df_es["_rel"] > 0).sum() == 0:
            return False, "Event study needs both pre and post periods.", None, None

        # Make categorical rel bins with baseline -1 omitted
        rel_vals = sorted(df_es["_rel_cap"].dropna().unique().tolist())
        if -1 not in rel_vals:
            return False, "Event study expects at least one period with rel_time = -1 (baseline).", None, None

        # Build dummy terms for each k except -1
        rel_terms = []
        for k in rel_vals:
            if k == -1:
                continue
            col = f"rel_{int(k)}"
            df_es[col] = (df_es["_rel_cap"] == k).astype(int)
            rel_terms.append(f"{col}:{treated}")

        # Need at least one pre coefficient (k <= -2) to diagnose pre-trends
        if not any(k <= -2 for k in rel_vals):
            return False, "Event study needs at least 2 pre-periods to estimate pre-trend coefficients.", None, None

        cov_terms = " + " + " + ".join(covariates) if covariates else ""
        rhs = " + ".join(rel_terms) + cov_terms
        formula = f"{y} ~ {rhs} + C({unit_id}) + C({time_key})"

        model = smf.ols(formula=formula, data=df_es)
        if cluster:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": df_es[unit_id]})
        else:
            res = model.fit(cov_type="HC1")

        # Extract coefficients and CIs in rel order
        rows = []
        for k in rel_vals:
            if k == -1:
                continue
            term = f"rel_{int(k)}:{treated}"
            if term not in res.params.index:
                continue
            ci_low, ci_high = res.conf_int(alpha=alpha).loc[term].tolist()
            rows.append(
                {
                    "rel_time": int(k),
                    "coef": float(res.params[term]),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "p_value": float(res.pvalues[term]),
                }
            )

        est = pd.DataFrame(rows).sort_values("rel_time")

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axhline(0, linestyle=":", linewidth=1)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.plot(est["rel_time"], est["coef"], marker="o")
        ax.fill_between(est["rel_time"], est["ci_low"], est["ci_high"], alpha=0.2)
        ax.set_xlabel("Relative time (0 = intervention period)")
        ax.set_ylabel("Estimated effect (vs baseline rel_time = -1)")
        ax.set_title("Event Study: Dynamic Treatment Effects (with 95% CI)")

        return True, "OK", est, fig
    except Exception as e:
        return False, f"Event study failed: {e}", None, None


def build_simple_html_report(
    title: str,
    notes: List[str],
    main_res: ModelResult,
    placebo_res: Optional[ModelResult],
    summary_table: pd.DataFrame,
    pivot_2x2: pd.DataFrame,
) -> str:
    def df_to_html(df: pd.DataFrame) -> str:
        return df.to_html(index=False, border=0)

    pivot_html = pivot_2x2.reset_index().to_html(index=False, border=0)

    notes_html = "".join([f"<li>{st.escape(n)}</li>" for n in notes])

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; color:#111827; }}
  h1,h2 {{ margin-bottom: 0.2rem; }}
  .muted {{ color: #6B7280; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem 0; }}
  th, td {{ border: 1px solid rgba(17,24,39,0.12); padding: 8px; font-size: 14px; text-align: left; }}
  .card {{ border: 1px solid rgba(17,24,39,0.10); border-radius: 12px; padding: 12px 14px; margin: 10px 0; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <p class="muted">Generated by the DiD Analysis App.</p>

  <h2>Key Notes</h2>
  <ul>{notes_html}</ul>

  <h2>Summary Statistics</h2>
  <div class="card">
    <h3 style="margin-top:0;">Summary table</h3>
    {df_to_html(summary_table)}
    <h3>2×2 means (treated/control × pre/post)</h3>
    {pivot_html}
  </div>

  <h2>Main DiD Result</h2>
  <div class="card">
    <p><b>Model:</b> {st.escape(main_res.message)}</p>
    <p><b>SE:</b> {st.escape(main_res.se_type)}</p>
    <p><b>β₃ (treated×post):</b> {main_res.coef:.6f}</p>
    <p><b>95% CI:</b> [{main_res.ci_low:.6f}, {main_res.ci_high:.6f}]</p>
    <p><b>p-value:</b> {main_res.pval:.6g}</p>
  </div>

  <h2>Placebo Result</h2>
  <div class="card">
    {"<p><b>Not run.</b></p>" if (placebo_res is None or not placebo_res.ok) else f"""
      <p><b>β₃_placebo:</b> {placebo_res.coef:.6f}</p>
      <p><b>95% CI:</b> [{placebo_res.ci_low:.6f}, {placebo_res.ci_high:.6f}]</p>
      <p><b>p-value:</b> {placebo_res.pval:.6g}</p>
    """}
  </div>
</body>
</html>
"""
    return html


# =========================================================
# Sidebar Navigation (Clickable TOC)
# =========================================================
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown(
        """
- [Welcome](#welcome)
- [Dataset Requirements](#dataset-requirements)
- [Upload & Mapping](#upload-mapping)
- [Summary Statistics](#summary-statistics)
- [DiD Estimate](#did-estimate)
- [Plots](#plots)
- [Placebo Test](#placebo-test)
- [Results & Export](#results-export)
""",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Model Options")
    twfe = st.checkbox("Use Two-way Fixed Effects (recommended)", value=True, help="Adds unit FE + time FE.")
    cluster = st.checkbox("Cluster standard errors by unit_id (recommended)", value=True)
    alpha = st.selectbox("Significance level (α)", options=[0.10, 0.05, 0.01], index=1)
    st.markdown("---")
    st.markdown(
        "<small>Tip: Click items above to jump to sections.</small>",
        unsafe_allow_html=True,
    )

# =========================================================
# 1) Welcome + What is DiD?
# =========================================================
anchor("welcome")
st.title("Welcome to the DiD Analysis App!")

st.write(
    """
**Difference-in-Differences (DiD)** is a simple way to estimate a causal effect when:
- you have a **treated group** (exposed to a policy/program) and a **control group** (not exposed), and
- you observe outcomes **before** and **after** the intervention.

**Causal question DiD answers:**  
> “How much did the outcome change *because of* the intervention, beyond what would have happened anyway?”

**Key assumption (parallel trends) in plain words:**  
Before the intervention, the treated and control groups should be moving in **similar directions over time**.
If that’s true, then the control group can act like the treated group’s “what would have happened without treatment.”
"""
)

st.write("**How to use this app:**")
st.markdown(
    """
- Upload a **CSV** (or use the sample dataset).
- Map your columns (outcome, treated, post, unit_id, time, optional covariates).
- Review summary stats and plots to check if the design looks reasonable.
- Read the DiD estimate and run the placebo test.
- Export results, plots, and a simple report.
"""
)

st.divider()

# =========================================================
# 2) Dataset Requirements (must appear before upload)
# =========================================================
anchor("dataset-requirements")
st.header("Dataset Requirements")

st.markdown(
    """
Your dataset **must** include these columns (names can differ — you will map them after upload):

✅ **Required**
- Outcome variable (numeric): **y**
- Treatment group indicator (0/1): **treated** (1 = treated group, 0 = control group)
- Time indicator (0/1): **post** (1 = after intervention, 0 = before)
- A unit identifier: **unit_id** (e.g., person, firm, county)
- A time identifier: **time** (date or period number)

➕ **Optional (supported)**
- Covariates: **x1, x2, …** (numeric recommended)

⚠️ **Treatment timing note**
- This app assumes a **single treatment date** shared by the treated group (i.e., not staggered adoption).
  If your treatment is staggered, this app will still run but the interpretation may be incorrect.

📁 **File format**
- **CSV only**

⚠️ **Missing data guidance**
- If key columns have missing values (**y, treated, post, unit_id, time**), those rows may be dropped.
  Too much missingness can make results unreliable.

✅ **Recommended structure**
- **Long format panel data**: one row per **unit × time**.
"""
)

st.divider()

# =========================================================
# 3) Upload & Column Mapping
# =========================================================
anchor("upload-mapping")
st.header("Upload & Column Mapping")

def generate_synthetic(seed=7, n_units=200, n_periods=10, treat_share=0.5, effect=2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    units = np.arange(n_units)
    times = np.arange(n_periods)
    treated_units = rng.choice(units, size=int(n_units * treat_share), replace=False)
    treat_flag = np.isin(units, treated_units).astype(int)
    cutoff = 6  # post=1 for t>=6
    rows = []
    for u in units:
        unit_fe = rng.normal(0, 1)
        for t in times:
            time_fe = 0.2 * t
            treated = int(treat_flag[u])
            post = int(t >= cutoff)
            eps = rng.normal(0, 1)
            y = 5 + unit_fe + time_fe + 0.5 * treated + 0.3 * post + effect * (treated * post) + eps
            x1 = rng.normal(0, 1) + 0.3 * treated + 0.1 * t
            x2 = rng.normal(0, 1)
            rows.append({"unit_id": u, "time": t, "treated": treated, "post": post, "y": y + 0.4 * x1 - 0.2 * x2, "x1": x1, "x2": x2})
    return pd.DataFrame(rows)

colA, colB = st.columns([1.2, 1])
with colA:
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
with colB:
    use_synth = st.checkbox("Use sample dataset instead", value=(uploaded is None))

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    if use_synth:
        df_raw = generate_synthetic()
        st.info("Using the built-in sample dataset.")
    else:
        st.warning("Please upload a CSV (or enable the sample dataset).")
        st.stop()

if df_raw.empty:
    st.error("Dataset is empty.")
    st.stop()

st.subheader("Quick Preview")
st.dataframe(df_raw.head(15), use_container_width=True)

cols = df_raw.columns.tolist()
numeric_candidates = [c for c in cols if pd.api.types.is_numeric_dtype(df_raw[c])]

st.subheader("Column Mapping")
m1, m2, m3 = st.columns(3)
with m1:
    outcome_col = st.selectbox("Outcome (numeric)", options=cols, index=cols.index("y") if "y" in cols else 0)
    treated_col = st.selectbox("Treated (0/1)", options=cols, index=cols.index("treated") if "treated" in cols else 0)
with m2:
    post_col = st.selectbox("Post (0/1)", options=cols, index=cols.index("post") if "post" in cols else 0)
    unit_col = st.selectbox("Unit ID", options=cols, index=cols.index("unit_id") if "unit_id" in cols else 0)
with m3:
    time_col = st.selectbox("Time", options=cols, index=cols.index("time") if "time" in cols else 0)
    covariates = st.multiselect("Optional covariates", options=numeric_candidates, default=[c for c in ["x1", "x2"] if c in numeric_candidates])

# Build modeling df
df = df_raw[[outcome_col, treated_col, post_col, unit_col, time_col] + covariates].copy()
df = df.rename(
    columns={
        outcome_col: "y",
        treated_col: "treated",
        post_col: "post",
        unit_col: "unit_id",
        time_col: "time",
    }
)

# Parse time + FE time key
df, time_kind, time_key_col = make_time_key(df, "time")

# Validation
validation_msgs = []

# Outcome numeric
df["y"] = pd.to_numeric(df["y"], errors="coerce")
if df["y"].isna().mean() > 0:
    validation_msgs.append("Outcome 'y' is not fully numeric; some rows became missing after conversion.")

# treated/post binary
df["treated"], msgs_t = coerce_binary01(df["treated"], "treated")
df["post"], msgs_p = coerce_binary01(df["post"], "post")
validation_msgs.extend(msgs_t + msgs_p)

# Drop missing on required
before = len(df)
df = df.dropna(subset=["y", "treated", "post", "unit_id", "_time_parsed", time_key_col])
dropped = before - len(df)
if dropped > 0:
    validation_msgs.append(f"Dropped {dropped} rows due to missing values in key columns after cleaning.")

# Covariates numeric
for c in covariates:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# At least 2 time periods and both pre/post exist
n_times = df[time_key_col].nunique(dropna=True)
if n_times < 2:
    validation_msgs.append("Need at least 2 distinct time periods.")
if df["post"].nunique(dropna=True) < 2:
    validation_msgs.append("Need both pre (post=0) and post (post=1) observations.")

# Treated and control groups exist
if df["treated"].nunique(dropna=True) < 2:
    validation_msgs.append("Need both treated (treated=1) and control (treated=0) units.")

# Panel-ish check
panel_ok, panel_msg = ensure_long_panel(df, "unit_id", "_time_parsed")
if not panel_ok:
    validation_msgs.append(f"Panel warning: {panel_msg} (The app will still run, but DiD may be invalid.)")

# 2x2 cells exist
cell_counts = df.groupby(["treated", "post"]).size().reset_index(name="N")
needed = {(0, 0), (0, 1), (1, 0), (1, 1)}
have = set(zip(cell_counts["treated"].tolist(), cell_counts["post"].tolist()))
missing_cells = needed - have
if missing_cells:
    validation_msgs.append(f"Missing 2×2 cells: {sorted(list(missing_cells))} (results may not be estimable).")

if validation_msgs:
    st.warning("Validation checks:\n- " + "\n- ".join(validation_msgs))
else:
    st.success("Column mapping and basic validation look good.")

st.divider()

# =========================================================
# 4) Summary Statistics (Table + Interpretation)
# =========================================================
anchor("summary-statistics")
st.header("Summary Statistics")

summary_table, pivot_2x2 = compute_summary_stats(df, "y", "treated", "post")
st.subheader("Table")
st.dataframe(summary_table, use_container_width=True)

c1, c2 = st.columns([1.2, 1])
with c1:
    st.write("**2×2 mean outcome table (treated/control × pre/post)**")
    st.dataframe(pivot_2x2, use_container_width=True)
with c2:
    did_means = did_from_2x2(pivot_2x2)
    if did_means is not None and np.isfinite(did_means):
        st.metric("Raw DiD from means", f"{did_means:.4f}")
        st.caption("This is a quick sanity check. Regression results may differ if you add covariates or fixed effects.")
    else:
        st.info("Raw DiD from means not available (likely missing a cell).")

st.subheader("Interpretation (read this before trusting the estimate)")
baseline_note = ""
try:
    baseline_diff = float(pivot_2x2.loc["Treated", "Pre"] - pivot_2x2.loc["Control", "Pre"])
    baseline_note = f"- Baseline (pre) difference treated vs control: **{baseline_diff:.4f}**."
except Exception:
    baseline_note = "- Baseline (pre) difference treated vs control: (not computable due to missing cell)."

prepost_note = ""
try:
    treated_change = float(pivot_2x2.loc["Treated", "Post"] - pivot_2x2.loc["Treated", "Pre"])
    control_change = float(pivot_2x2.loc["Control", "Post"] - pivot_2x2.loc["Control", "Pre"])
    prepost_note = f"- Raw change: treated **{treated_change:.4f}**, control **{control_change:.4f}**."
except Exception:
    prepost_note = "- Raw change: (not computable due to missing cell)."

st.markdown(
    f"""
{baseline_note}  
{prepost_note}  
- **Important:** raw differences are **not automatically causal**. DiD becomes causal only if **parallel trends** is reasonable (treated and control would have evolved similarly without the intervention).
"""
)

st.divider()

# =========================================================
# 5) Core DiD Estimate (Model + Results)
# =========================================================
anchor("did-estimate")
st.header("Core DiD Estimate")

st.markdown(
    """
We estimate the standard DiD regression:

**y = α + β₁·treated + β₂·post + β₃·(treated×post) + ε**

- **β₃ is the DiD estimate** (the “extra” post-period change for treated units beyond the control trend).
- If you provided `unit_id` and `time`, we can also run **two-way fixed effects** (recommended).
"""
)

# Fit main model
main_res = fit_did(
    df=df,
    y="y",
    treated="treated",
    post="post",
    unit_id="unit_id",
    time_key=time_key_col,
    covariates=covariates,
    alpha=alpha,
    cluster=cluster,
    twfe=twfe,
)

if not main_res.ok:
    st.error(main_res.message)
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("β₃ (treated×post)", f"{main_res.coef:.4f}")
k2.metric("Std. Error", f"{main_res.se:.4f}")
k3.metric("p-value", f"{main_res.pval:.4g}")
k4.metric(f"{int((1-alpha)*100)}% CI", f"[{main_res.ci_low:.4f}, {main_res.ci_high:.4f}]")

st.caption(f"Model: {main_res.message} | SE: {main_res.se_type} | N={main_res.nobs}")
st.markdown(render_interpretation_beta3(main_res.coef, alpha, main_res.pval))

with st.expander("See regression table"):
    st.dataframe(main_res.reg_table, use_container_width=True)

st.divider()

# =========================================================
# 6) Essential DiD Plots
# =========================================================
anchor("plots")
st.header("Plots")

# Infer intervention time (first post period)
intervention_time = infer_intervention_point(df, "post", time_kind)

# A) Parallel trends
st.subheader("A) Parallel Trends / Group Means Over Time")
if intervention_time is None:
    st.warning("Cannot infer the intervention point because there are no rows with post = 1.")
else:
    g = (
        df.groupby(["_time_parsed", "treated"])["y"]
        .mean()
        .reset_index()
        .sort_values("_time_parsed")
    )

    figA = plt.figure()
    ax = figA.add_subplot(111)

    for tval, label in [(0, "Control (treated=0)"), (1, "Treated (treated=1)")]:
        gg = g[g["treated"] == tval]
        ax.plot(gg["_time_parsed"], gg["y"], marker="o", label=label)

    ax.axvline(intervention_time, linestyle="--", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean outcome (y)")
    ax.set_title("Mean outcome over time by group")
    ax.legend()
    st.pyplot(figA, clear_figure=True)

    st.markdown(
        """
**How to read this:**  
Before the dashed line (the intervention), the treated and control lines should look roughly **parallel** (similar direction and slope).
If they diverge strongly before treatment, that’s a warning sign for the parallel trends assumption.
"""
    )

# B) Event study (if feasible)
st.subheader("B) Event Study (dynamic effects)")
if intervention_time is None:
    st.info("Event study disabled (no post=1 period detected).")
else:
    # Event study needs multiple periods; if only 2 periods, disable
    if df[time_key_col].nunique() <= 2:
        st.info("Event study disabled: only 2 time periods detected. (Event study needs multiple pre and post periods.)")
    else:
        ok_es, msg_es, est_es, fig_es = event_study(
            df=df,
            y="y",
            treated="treated",
            unit_id="unit_id",
            time_key=time_key_col,
            time_kind=time_kind,
            intervention_time=intervention_time,
            covariates=covariates,
            alpha=alpha,
            cluster=cluster,
            max_leads_lags=6,
        )
        if not ok_es:
            st.info(f"Event study not available: {msg_es}")
        else:
            st.pyplot(fig_es, clear_figure=True)
            st.markdown(
                """
**How to read this:**  
- The vertical line at **0** is the intervention period.  
- Coefficients **before** 0 (negative relative time) should be near **0** if pre-trends are parallel.  
- Coefficients **after** 0 show how the effect evolves over time.
"""
            )
            with st.expander("Event study coefficient table"):
                st.dataframe(est_es, use_container_width=True)

# C) Boxplots (pre only and post only)
st.subheader("C) Outcome Distribution (Boxplots)")
cA, cB = st.columns(2)

with cA:
    st.write("**Pre period only: treated vs control**")
    df_pre = df[df["post"] == 0].copy()
    if df_pre.empty or df_pre["treated"].nunique() < 2:
        st.info("Not enough pre data to show this plot.")
        figC1 = None
    else:
        figC1 = plt.figure()
        ax = figC1.add_subplot(111)
        data = [
            df_pre.loc[df_pre["treated"] == 0, "y"].dropna().values,
            df_pre.loc[df_pre["treated"] == 1, "y"].dropna().values,
        ]
        ax.boxplot(data, labels=["Control", "Treated"])
        ax.set_ylabel("Outcome (y)")
        ax.set_title("Pre: Outcome distribution by group")
        st.pyplot(figC1, clear_figure=True)

with cB:
    st.write("**Post period only: treated vs control**")
    df_post = df[df["post"] == 1].copy()
    if df_post.empty or df_post["treated"].nunique() < 2:
        st.info("Not enough post data to show this plot.")
        figC2 = None
    else:
        figC2 = plt.figure()
        ax = figC2.add_subplot(111)
        data = [
            df_post.loc[df_post["treated"] == 0, "y"].dropna().values,
            df_post.loc[df_post["treated"] == 1, "y"].dropna().values,
        ]
        ax.boxplot(data, labels=["Control", "Treated"])
        ax.set_ylabel("Outcome (y)")
        ax.set_title("Post: Outcome distribution by group")
        st.pyplot(figC2, clear_figure=True)

st.divider()

# =========================================================
# 7) Placebo Test (must run)
# =========================================================
anchor("placebo-test")
st.header("Placebo Test")

st.markdown(
    """
We run a **fake treatment date** inside the pre-period (preferred placebo).

**Goal:** If the design is valid, we should not find a “treatment effect” *before* the real intervention.
So the placebo interaction should be close to **0** and usually **not significant**.
"""
)

placebo_res = None
placebo_fig = None

if intervention_time is None:
    st.warning("Placebo test cannot run because we cannot infer a real intervention point (no post=1 detected).")
else:
    # Choose placebo cutoff in the PRE period
    df_pre_all = df[df["post"] == 0].copy()

    if df_pre_all.empty:
        st.warning("No pre-period rows found (post=0). Placebo test cannot run.")
    else:
        # Need at least a few distinct pre times
        pre_times = sorted_time_for_plot(df_pre_all, time_kind)
        if len(pre_times) < 3:
            st.warning("Too few distinct pre-period time points for a meaningful placebo cutoff.")
        else:
            # Avoid endpoints to reduce degenerate splits
            selectable = pre_times[1:-1] if len(pre_times) > 3 else pre_times
            default_idx = len(selectable) // 2
            placebo_cutoff = st.selectbox(
                "Choose a placebo cutoff (must be within the PRE period)",
                options=selectable,
                index=default_idx,
                help="We create a placebo_post = 1 if time >= placebo_cutoff (still inside pre).",
            )

            df_pl = df.copy()
            df_pl["placebo_post"] = (df_pl["_time_parsed"] >= placebo_cutoff).astype(int)

            # Validate 2x2 exists for placebo
            cell_pl = df_pl.groupby(["treated", "placebo_post"]).size().reset_index(name="N")
            have_pl = set(zip(cell_pl["treated"].tolist(), cell_pl["placebo_post"].tolist()))
            missing_pl = {(0, 0), (0, 1), (1, 0), (1, 1)} - have_pl

            if missing_pl:
                st.warning(f"Placebo 2×2 cells missing: {sorted(list(missing_pl))}. Try a different cutoff.")
            else:
                placebo_res = fit_did(
                    df=df_pl.rename(columns={"placebo_post": "post"}),  # reuse function (treat placebo as post)
                    y="y",
                    treated="treated",
                    post="post",
                    unit_id="unit_id",
                    time_key=time_key_col,
                    covariates=covariates,
                    alpha=alpha,
                    cluster=cluster,
                    twfe=twfe,
                )

                if not placebo_res.ok:
                    st.error(placebo_res.message)
                else:
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("β₃_placebo", f"{placebo_res.coef:.4f}")
                    p2.metric("Std. Error", f"{placebo_res.se:.4f}")
                    p3.metric("p-value", f"{placebo_res.pval:.4g}")
                    p4.metric(f"{int((1-alpha)*100)}% CI", f"[{placebo_res.ci_low:.4f}, {placebo_res.ci_high:.4f}]")

                    st.caption(f"Placebo cutoff: {placebo_cutoff} | Model: {placebo_res.message}")

                    st.markdown(
                        f"""
**Interpretation:**  
A good sign is **β₃_placebo ≈ 0** and not statistically significant.  
Here, the placebo effect is **{placebo_res.coef:.4f}** (p = {placebo_res.pval:.4g}).
"""
                    )

                    # Placebo plot (coefficient + CI)
                    placebo_fig = plt.figure()
                    ax = placebo_fig.add_subplot(111)
                    ax.axhline(0, linestyle=":", linewidth=1)
                    ax.errorbar(
                        [0],
                        [placebo_res.coef],
                        yerr=[[placebo_res.coef - placebo_res.ci_low], [placebo_res.ci_high - placebo_res.coef]],
                        fmt="o",
                        capsize=6,
                    )
                    ax.set_xticks([0])
                    ax.set_xticklabels(["Placebo β₃"])
                    ax.set_ylabel("Coefficient")
                    ax.set_title("Placebo DiD effect (with 95% CI)")
                    st.pyplot(placebo_fig, clear_figure=True)

                    with st.expander("Placebo regression table"):
                        st.dataframe(placebo_res.reg_table, use_container_width=True)

st.divider()

# =========================================================
# 8) Results Summary + Export
# =========================================================
anchor("results-export")
st.header("Results Summary + Export")

warnings_list = []
# pre-trend warning heuristic: if parallel trends plot looks off we can’t detect automatically;
# but we can warn if placebo significant
if placebo_res is not None and placebo_res.ok and placebo_res.pval < alpha:
    warnings_list.append("Placebo test is statistically significant → possible pre-trends / design issues.")

if not panel_ok:
    warnings_list.append("Data may not be panel / long format → DiD assumptions may not hold.")

if df[time_key_col].nunique() <= 2:
    warnings_list.append("Only 2 time periods → event study disabled; diagnostics are limited.")

takeaways = [
    f"Estimated effect (β₃): **{main_res.coef:.4f}** (95% CI: [{main_res.ci_low:.4f}, {main_res.ci_high:.4f}]).",
    f"Statistical significance: **{'Significant' if main_res.pval < alpha else 'Not significant'}** at α={alpha:.2f} (p={main_res.pval:.4g}).",
]

if placebo_res is None or not placebo_res.ok:
    takeaways.append("Placebo test: **Not run / not available**.")
else:
    takeaways.append(
        f"Placebo test: β₃_placebo = **{placebo_res.coef:.4f}** (p={placebo_res.pval:.4g}) → "
        f"{'supports credibility (near zero)' if placebo_res.pval >= alpha else 'warning sign (not near zero)'}."
    )

if warnings_list:
    takeaways.append("Warnings: " + " ".join([f"**{w}**" for w in warnings_list]))

st.subheader("Key Takeaways")
st.markdown("\n".join([f"- {t}" for t in takeaways]))

# -------- Export: Results table CSV --------
st.subheader("Export")
export_cols = ["coef", "std_err", "p_value", f"CI_{int((1-alpha)*100)}%_low", f"CI_{int((1-alpha)*100)}%_high"]
results_rows = [
    {
        "model": "main",
        "coef": main_res.coef,
        "std_err": main_res.se,
        "p_value": main_res.pval,
        f"CI_{int((1-alpha)*100)}%_low": main_res.ci_low,
        f"CI_{int((1-alpha)*100)}%_high": main_res.ci_high,
    }
]
if placebo_res is not None and placebo_res.ok:
    results_rows.append(
        {
            "model": "placebo",
            "coef": placebo_res.coef,
            "std_err": placebo_res.se,
            "p_value": placebo_res.pval,
            f"CI_{int((1-alpha)*100)}%_low": placebo_res.ci_low,
            f"CI_{int((1-alpha)*100)}%_high": placebo_res.ci_high,
        }
    )
results_df = pd.DataFrame(results_rows)

st.download_button(
    "Download results table (CSV)",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="did_results.csv",
    mime="text/csv",
)

# -------- Export: plots as PNG --------
plot_bytes = {}

# Parallel trends figA
if "figA" in globals() and figA is not None:
    plot_bytes["parallel_trends.png"] = fig_to_png_bytes(figA)
# Event study
if "fig_es" in globals() and fig_es is not None:
    plot_bytes["event_study.png"] = fig_to_png_bytes(fig_es)
# Boxplots
if "figC1" in globals() and figC1 is not None:
    plot_bytes["boxplot_pre.png"] = fig_to_png_bytes(figC1)
if "figC2" in globals() and figC2 is not None:
    plot_bytes["boxplot_post.png"] = fig_to_png_bytes(figC2)
# Placebo
if placebo_fig is not None:
    plot_bytes["placebo_effect.png"] = fig_to_png_bytes(placebo_fig)

if plot_bytes:
    # Zip-less approach: let user download each plot
    st.write("Download plots (PNG):")
    for fname, b in plot_bytes.items():
        st.download_button(
            f"Download {fname}",
            data=b,
            file_name=fname,
            mime="image/png",
            key=f"dl_{fname}",
        )
else:
    st.info("No plots available to export yet.")

# -------- Export: simple HTML report --------
notes = [
    f"Model type: {main_res.message}",
    f"SE type: {main_res.se_type}",
    "Interpretation relies on the parallel trends assumption.",
]
if warnings_list:
    notes.extend(warnings_list)

html_report = build_simple_html_report(
    title="DiD Analysis Report",
    notes=notes,
    main_res=main_res,
    placebo_res=placebo_res if (placebo_res is not None and placebo_res.ok) else None,
    summary_table=summary_table,
    pivot_2x2=pivot_2x2,
)

st.download_button(
    "Download simple report (HTML)",
    data=html_report.encode("utf-8"),
    file_name="did_report.html",
    mime="text/html",
)

st.caption("That’s it! Use the sidebar navigation to jump back to any section instantly.")
















   
  

