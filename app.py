# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from dataclasses import dataclass

# =============================
# App Identity + Dark Styling
# =============================
APP_NAME = "DiD Insight Studio"
APP_SUBTITLE = "Difference-in-Differences estimator with diagnostics, plots, and placebo checks."

st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")

# Strong dark UI even without .streamlit/config.toml
st.markdown(
    """
<style>
/* Global background */
.stApp { background-color: #0B0F14; color: #E5E7EB; }

/* Remove some default whitespace feel */
.block-container { padding-top: 1.2rem; }

/* Make metric cards feel like dashboard tiles */
div[data-testid="stMetric"] {
  background-color: #111827;
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.08);
}

/* Dataframe border polish */
div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.08);
}

/* Code blocks */
pre { border-radius: 12px !important; }

/* Section headers spacing */
h1, h2, h3 { letter-spacing: 0.2px; }

/* Subtle divider */
hr { border-top: 1px solid rgba(255,255,255,0.10); }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def try_parse_time(s: pd.Series) -> pd.Series:
    """
    Try to parse a time column that may be numeric-like or date-like.
    Returns a series that is either datetime64[ns] or numeric.
    """
    if np.issubdtype(s.dtype, np.datetime64):
        return s

    if s.dtype == object or pd.api.types.is_string_dtype(s):
        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        if dt.notna().mean() >= 0.8:
            return dt

    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.8:
        return num

    return s


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


def make_design_df(df: pd.DataFrame, outcome, unit_id, time_col, treated, post, covariates):
    """
    Build a clean modeling dataframe with safe column names.
    """
    cols = [outcome, unit_id, time_col, treated, post] + (covariates or [])
    d = df[cols].copy()

    rename_map = {
        outcome: "y",
        unit_id: "unit",
        time_col: "time_raw",
        treated: "treated",
        post: "post",
    }
    if covariates:
        for i, c in enumerate(covariates, start=1):
            rename_map[c] = f"x{i}"

    d = d.rename(columns=rename_map)
    d["time"] = try_parse_time(d["time_raw"])

    before = len(d)
    d = d.dropna(subset=["y", "unit", "time", "treated", "post"])
    dropped = before - len(d)

    return d, dropped


def fit_did_model(df_model: pd.DataFrame, covariate_cols, cluster: bool, alpha: float):
    """
    Fit DiD regression:
      y ~ treated + post + treated:post (+ covariates)
    """
    cov_terms = ""
    if covariate_cols:
        cov_terms = " + " + " + ".join(covariate_cols)

    formula = f"y ~ treated + post + treated:post{cov_terms}"
    model = smf.ols(formula=formula, data=df_model)

    if cluster:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df_model["unit"]})
        se_type = "Cluster-robust (by unit)"
    else:
        res = model.fit(cov_type="HC1")
        se_type = "Robust (HC1)"

    term = "treated:post"
    coef = float(res.params.get(term, np.nan))
    se = float(res.bse.get(term, np.nan))
    pval = float(res.pvalues.get(term, np.nan))
    ci_low, ci_high = res.conf_int(alpha=alpha).loc[term].tolist()

    return res, se_type, term, coef, se, pval, float(ci_low), float(ci_high), formula


def compact_reg_table(res, alpha: float):
    params = res.params
    bse = res.bse
    tvals = res.tvalues
    pvals = res.pvalues
    ci = res.conf_int(alpha=alpha)
    out = pd.DataFrame(
        {
            "coef": params,
            "std_err": bse,
            "t": tvals,
            "p_value": pvals,
            f"CI_{int((1-alpha)*100)}%_low": ci[0],
            f"CI_{int((1-alpha)*100)}%_high": ci[1],
        }
    )
    out.index.name = "term"
    return out.reset_index()


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


def check_2x2_cells(df_model: pd.DataFrame):
    counts = df_model.groupby(["treated", "post"]).size().reset_index(name="n")
    cell_set = set(zip(counts["treated"].tolist(), counts["post"].tolist()))
    needed = {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
    missing = needed - cell_set
    return counts, missing


def infer_post_start_time(df_model: pd.DataFrame):
    post_times = df_model.loc[df_model["post"] == 1, "time"]
    if len(post_times) == 0:
        return None
    return post_times.min()


def sorted_unique_times(df_model: pd.DataFrame):
    t = df_model["time"].copy()
    try:
        uniq = pd.Series(pd.unique(t)).dropna().sort_values()
        return uniq.tolist()
    except Exception:
        uniq = sorted([str(x) for x in pd.unique(t.dropna())])
        return uniq


def render_interpretation(effect, pval, alpha, label="DiD"):
    sig = "statistically significant" if (pval < alpha) else "not statistically significant"
    return (
        f"**Interpretation ({label}):** The interaction term estimates the **treatment effect on the treated** "
        f"under the parallel trends assumption. Here, the estimated effect is **{effect:.4f}** and it is **{sig}** "
        f"at significance level α = {alpha:.2f} (p = {pval:.4g})."
    )


def has_all_cells(df, treat_col, post_col):
    c = df.groupby([treat_col, post_col]).size().reset_index(name="n")
    have = set(zip(c[treat_col].astype(float), c[post_col].astype(float)))
    needed = {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
    return (needed - have) == set(), c


def run_placebo_shift_sweep(df_model, cov_cols, cluster, alpha, times, real_cutoff_time):
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
        dfA["placebo_post"] = (dfA["time"] >= placebo_cutoff_time).astype(int).astype(float)

        ok_cells, _counts = has_all_cells(dfA, "treated", "placebo_post")
        if not ok_cells:
            continue

        formulaA = f"y ~ treated + placebo_post + treated:placebo_post{cov_terms}"
        modelA = smf.ols(formula=formulaA, data=dfA)

        if cluster:
            resA = modelA.fit(cov_type="cluster", cov_kwds={"groups": dfA["unit"]})
        else:
            resA = modelA.fit(cov_type="HC1")

        termA = "treated:placebo_post"
        coefA = float(resA.params.get(termA, np.nan))
        ciLA, ciHA = resA.conf_int(alpha=alpha).loc[termA].tolist()

        rows.append(
            {
                "K_shift": K,
                "placebo_cutoff": placebo_cutoff_time,
                "coef": coefA,
                "ci_low": float(ciLA),
                "ci_high": float(ciHA),
            }
        )

    return pd.DataFrame(rows)


@dataclass
class PlaceboResult:
    ok: bool
    message: str
    res: object = None
    se_type: str = ""
    term: str = ""
    coef: float = np.nan
    se: float = np.nan
    pval: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    formula: str = ""
    n_obs: int = 0


# -----------------------------
# Synthetic data generator
# -----------------------------
def generate_synthetic(seed=7, n_units=200, n_periods=10, treat_share=0.5, effect=2.0):
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
            y = 5 + unit_fe + time_fe + 0.5 * treated + 0.3 * post
            y += effect * (treated * post)

            x1 = rng.normal(0, 1) + 0.3 * treated + 0.1 * t
            x2 = rng.normal(0, 1)

            rows.append(
                {
                    "unit_id": u,
                    "time": t,
                    "treated": treated,
                    "post": post,
                    "y": y + eps + 0.4 * x1 - 0.2 * x2,
                    "x1": x1,
                    "x2": x2,
                }
            )
    return pd.DataFrame(rows)


# =============================
# UI
# =============================
st.title(APP_NAME)
st.caption(APP_SUBTITLE)

with st.expander("Install / Run Instructions", expanded=False):
    st.code(
        "pip install streamlit pandas numpy statsmodels matplotlib\n"
        "streamlit run app.py\n\n"
        "# Optional (recommended): create .streamlit/config.toml for theme\n"
        "# [theme]\n"
        "# base='dark'\n"
        "# backgroundColor='#0B0F14'\n"
        "# secondaryBackgroundColor='#111827'\n"
        "# textColor='#E5E7EB'\n",
        language="bash",
    )

st.write("Upload a CSV or use the built-in synthetic dataset. Then select columns and run DiD + placebo diagnostics.")

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

if df_raw.empty:
    st.error("Dataset is empty.")
    st.stop()

cols = df_raw.columns.tolist()

# Sidebar “app-like” controls
with st.sidebar:
    st.header("Model Setup")
    outcome_col = st.selectbox("Outcome (Y)", options=cols, index=cols.index("y") if "y" in cols else 0)
    unit_col = st.selectbox("Unit ID", options=cols, index=cols.index("unit_id") if "unit_id" in cols else 0)
    time_col = st.selectbox("Time", options=cols, index=cols.index("time") if "time" in cols else 0)
    treated_col = st.selectbox("Treated (0/1)", options=cols, index=cols.index("treated") if "treated" in cols else 0)
    post_col = st.selectbox("Post (0/1)", options=cols, index=cols.index("post") if "post" in cols else 0)

    numeric_candidates = [c for c in cols if pd.api.types.is_numeric_dtype(df_raw[c])]
    default_covs = [c for c in ["x1", "x2"] if c in cols]
    covariates = st.multiselect(
        "Optional covariates",
        options=numeric_candidates,
        default=default_covs if set(default_covs).issubset(set(numeric_candidates)) else [],
    )

    cluster = st.checkbox("Cluster SE by unit", value=True)
    alpha = st.selectbox("Significance level α", options=[0.10, 0.05, 0.01], index=1)

st.subheader("1) Data Preview")
st.dataframe(df_raw.head(20), use_container_width=True)

st.divider()

# -----------------------------
# Build model df and validate
# -----------------------------
df_model, dropped = make_design_df(df_raw, outcome_col, unit_col, time_col, treated_col, post_col, covariates)

warn_msgs = []
if dropped > 0:
    warn_msgs.append(f"Dropped {dropped} rows due to missing values in required columns.")

df_model["treated"], msgs_t = coerce_binary01(df_model["treated"], "treated")
df_model["post"], msgs_p = coerce_binary01(df_model["post"], "post")
warn_msgs.extend(msgs_t)
warn_msgs.extend(msgs_p)

df_model["y"] = pd.to_numeric(df_model["y"], errors="coerce")
na_y = df_model["y"].isna().sum()
if na_y > 0:
    warn_msgs.append(f"Outcome 'y' has {na_y} missing values after numeric coercion; those rows were dropped.")
    df_model = df_model.dropna(subset=["y"])

cov_cols = []
if covariates:
    for i, _orig in enumerate(covariates, start=1):
        col = f"x{i}"
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
        cov_cols.append(col)

if len(df_model) < 30:
    warn_msgs.append("Very few observations (<30). Estimates and placebo tests may be unstable.")

cell_counts, missing_cells = check_2x2_cells(df_model)
if missing_cells:
    warn_msgs.append(
        f"Real DiD 2x2 cells missing: {sorted(list(missing_cells))}. "
        "You may not have both treated/control and pre/post observations."
    )

if warn_msgs:
    st.warning("Validation / Warnings:\n- " + "\n- ".join(warn_msgs))

# -----------------------------
# Main Model
# -----------------------------
st.subheader("2) Main Estimate (DiD)")

try:
    res_main, se_type_main, term_main, coef_main, se_main, p_main, ciL_main, ciH_main, formula_main = fit_did_model(
        df_model, cov_cols, cluster=cluster, alpha=alpha
    )
except Exception as e:
    st.error(f"Model failed to run. Error: {e}")
    st.stop()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("DiD (treated×post) coef", f"{coef_main:.4f}")
kpi2.metric("Std. Error", f"{se_main:.4f}")
kpi3.metric("p-value", f"{p_main:.4g}")
kpi4.metric(f"{int((1-alpha)*100)}% CI", f"[{ciL_main:.4f}, {ciH_main:.4f}]")

st.caption(f"SE type: {se_type_main} | Model: {formula_main}")
st.markdown(render_interpretation(coef_main, p_main, alpha, label="Main DiD"))

with st.expander("Compact regression table (main model)", expanded=False):
    st.dataframe(compact_reg_table(res_main, alpha=alpha), use_container_width=True)

st.divider()

# -----------------------------
# Manual 2x2 DiD
# -----------------------------
st.subheader("3) Sanity Check (2×2 Means)")

pivot, did_val, did_ok = did_2x2_table(df_model)
left, right = st.columns([1.2, 1])
with left:
    st.write("**2×2 mean outcome table**")
    st.dataframe(pivot, use_container_width=True)
with right:
    st.write("**Manual DiD**")
    if did_ok and np.isfinite(did_val):
        st.metric("DiD from means", f"{did_val:.4f}")
        st.write(
            "This equals: (Treated Post − Treated Pre) − (Control Post − Control Pre). "
            "It should be close to the regression interaction coefficient when the model has no extra covariates "
            "and the sample is balanced."
        )
    else:
        st.error("Could not compute manual DiD because one or more 2×2 cells are missing.")

with st.expander("2×2 cell counts (treated/control × pre/post)", expanded=False):
    st.dataframe(cell_counts, use_container_width=True)

st.divider()

# -----------------------------
# Plots
# -----------------------------
st.subheader("4) Visuals")

post_start = infer_post_start_time(df_model)
if post_start is None:
    st.warning("Cannot infer treatment cutoff because 'post' has no 1s.")
else:
    st.caption(f"Inferred treatment cutoff (first post period): {post_start}")

plot1, plot2 = st.columns(2)

with plot1:
    st.write("**Parallel Trends (Average Y over time)**")
    try:
        g = df_model.groupby(["time", "treated"])["y"].mean().reset_index().sort_values("time")
        pt = g.pivot(index="time", columns="treated", values="y").sort_index()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if 0.0 in pt.columns:
            ax.plot(pt.index, pt[0.0].values, marker="o", label="Control (treated=0)")
        if 1.0 in pt.columns:
            ax.plot(pt.index, pt[1.0].values, marker="o", label="Treated (treated=1)")

        if post_start is not None:
            ax.axvline(post_start, linestyle="--", linewidth=1)

        ax.set_xlabel("Time")
        ax.set_ylabel("Average outcome (Y)")
        ax.legend()
        ax.set_title("Average Y by group over time")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"Parallel trends plot failed: {e}")

with plot2:
    st.write("**Outcome Distribution**")
    dist_mode = st.selectbox(
        "Compare distributions",
        options=[
            "Treated vs Control (all periods)",
            "Pre vs Post (all units)",
            "4 groups (treated/control × pre/post)",
        ],
        index=0,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    y = df_model["y"].dropna()
    if y.empty:
        st.error("No valid outcome data to plot.")
    else:
        if dist_mode == "Treated vs Control (all periods)":
            y0 = df_model.loc[df_model["treated"] == 0, "y"].dropna()
            y1 = df_model.loc[df_model["treated"] == 1, "y"].dropna()
            ax.hist(y0.values, bins=30, alpha=0.6, label="Control (treated=0)")
            ax.hist(y1.values, bins=30, alpha=0.6, label="Treated (treated=1)")
            ax.set_title("Outcome distribution: Treated vs Control")

        elif dist_mode == "Pre vs Post (all units)":
            ypre = df_model.loc[df_model["post"] == 0, "y"].dropna()
            ypost = df_model.loc[df_model["post"] == 1, "y"].dropna()
            ax.hist(ypre.values, bins=30, alpha=0.6, label="Pre (post=0)")
            ax.hist(ypost.values, bins=30, alpha=0.6, label="Post (post=1)")
            ax.set_title("Outcome distribution: Pre vs Post")

        else:
            labels = [
                ("Control-Pre", (df_model["treated"] == 0) & (df_model["post"] == 0)),
                ("Control-Post", (df_model["treated"] == 0) & (df_model["post"] == 1)),
                ("Treated-Pre", (df_model["treated"] == 1) & (df_model["post"] == 0)),
                ("Treated-Post", (df_model["treated"] == 1) & (df_model["post"] == 1)),
            ]
            for lab, mask in labels:
                yy = df_model.loc[mask, "y"].dropna()
                if len(yy) > 0:
                    ax.hist(yy.values, bins=30, alpha=0.5, label=lab)
            ax.set_title("Outcome distribution: 4 groups")

        ax.set_xlabel("Outcome (Y)")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

st.divider()

# -----------------------------
# Placebo Tests + Visual
# -----------------------------
st.subheader("5) Diagnostics (Placebo / Pre-trend checks)")

st.write("### Placebo A — Fake policy date (shift post earlier)")
times = sorted_unique_times(df_model)

if post_start is None or len(times) < 3:
    st.warning("Placebo A unavailable: cannot infer post start time or too few time periods.")
    placeboA = PlaceboResult(ok=False, message="Insufficient time periods or no post==1.")
else:
    cutoff_idx = times.index(post_start)
    max_k = max(1, cutoff_idx)

    K = st.slider("Shift earlier by K time steps", min_value=1, max_value=int(max_k), value=1, step=1)
    placebo_cutoff_time = times[max(0, cutoff_idx - K)]

    dfA = df_model.copy()
    dfA["placebo_post"] = (dfA["time"] >= placebo_cutoff_time).astype(int).astype(float)

    okA, countsA = has_all_cells(dfA, "treated", "placebo_post")

    st.caption(f"Real cutoff: {post_start} | Placebo cutoff (K={K}): {placebo_cutoff_time}")

    if not okA:
        st.warning("Placebo A has missing cells (treated/control × placebo pre/post).")
        placeboA = PlaceboResult(ok=False, message="Missing placebo A 2x2 cells.")
    else:
        try:
            cov_terms = (" + " + " + ".join(cov_cols)) if cov_cols else ""
            formulaA = f"y ~ treated + placebo_post + treated:placebo_post{cov_terms}"
            modelA = smf.ols(formula=formulaA, data=dfA)

            if cluster:
                resA = modelA.fit(cov_type="cluster", cov_kwds={"groups": dfA["unit"]})
                se_typeA = "Cluster-robust (by unit)"
            else:
                resA = modelA.fit(cov_type="HC1")
                se_typeA = "Robust (HC1)"

            termA = "treated:placebo_post"
            coefA = float(resA.params.get(termA, np.nan))
            seA = float(resA.bse.get(termA, np.nan))
            pA = float(resA.pvalues.get(termA, np.nan))
            ciLA, ciHA = resA.conf_int(alpha=alpha).loc[termA].tolist()

            placeboA = PlaceboResult(
                ok=True,
                message="OK",
                res=resA,
                se_type=se_typeA,
                term=termA,
                coef=coefA,
                se=seA,
                pval=pA,
                ci_low=float(ciLA),
                ci_high=float(ciHA),
                formula=formulaA,
                n_obs=int(resA.nobs),
            )
        except Exception as e:
            placeboA = PlaceboResult(ok=False, message=f"Model failed: {e}")

if placeboA.ok:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Placebo A interaction coef", f"{placeboA.coef:.4f}")
    a2.metric("Std. Error", f"{placeboA.se:.4f}")
    a3.metric("p-value", f"{placeboA.pval:.4g}")
    a4.metric(f"{int((1-alpha)*100)}% CI", f"[{placeboA.ci_low:.4f}, {placeboA.ci_high:.4f}]")
    st.caption(f"SE type: {placeboA.se_type} | Model: {placeboA.formula} | N={placeboA.n_obs}")
    st.markdown(
        "**Interpretation (Placebo A):** This tests for **pre-trends** by pretending the policy started earlier. "
        "If parallel trends holds, the placebo interaction should be close to 0 and typically not significant. "
        + (f"Here it is {'significant' if placeboA.pval < alpha else 'not significant'} at α={alpha:.2f}.")
    )
    with st.expander("Compact regression table (Placebo A)", expanded=False):
        st.dataframe(compact_reg_table(placeboA.res, alpha=alpha), use_container_width=True)
else:
    st.info(f"Placebo A not run: {placeboA.message}")

st.write("---")

st.write("### Placebo B — Pre-period only (fake cutoff within pre)")
df_pre = df_model.loc[df_model["post"] == 0].copy()

if df_pre.empty or df_pre["treated"].nunique(dropna=True) < 2:
    st.warning("Placebo B unavailable: no pre-period data or no treated/control variation in pre.")
    placeboB = PlaceboResult(ok=False, message="Insufficient pre data.")
else:
    pre_times = sorted_unique_times(df_pre)
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
            help="Creates a fake 'post' inside pre to test for differential pre-trends.",
        )

        dfB = df_pre.copy()
        dfB["placebo_post_pre"] = (dfB["time"] >= fake_cutoff).astype(int).astype(float)
        okB, countsB = has_all_cells(dfB, "treated", "placebo_post_pre")

        if not okB:
            st.warning("Placebo B has missing cells (treated/control × placebo pre/post).")
            placeboB = PlaceboResult(ok=False, message="Missing placebo B 2x2 cells.")
        else:
            try:
                cov_terms = (" + " + " + ".join(cov_cols)) if cov_cols else ""
                formulaB = f"y ~ treated + placebo_post_pre + treated:placebo_post_pre{cov_terms}"
                modelB = smf.ols(formula=formulaB, data=dfB)

                if cluster:
                    resB = modelB.fit(cov_type="cluster", cov_kwds={"groups": dfB["unit"]})
                    se_typeB = "Cluster-robust (by unit)"
                else:
                    resB = modelB.fit(cov_type="HC1")
                    se_typeB = "Robust (HC1)"

                termB = "treated:placebo_post_pre"
                coefB = float(resB.params.get(termB, np.nan))
                seB = float(resB.bse.get(termB, np.nan))
                pB = float(resB.pvalues.get(termB, np.nan))
                ciLB, ciHB = resB.conf_int(alpha=alpha).loc[termB].tolist()

                placeboB = PlaceboResult(
                    ok=True,
                    message="OK",
                    res=resB,
                    se_type=se_typeB,
                    term=termB,
                    coef=coefB,
                    se=seB,
                    pval=pB,
                    ci_low=float(ciLB),
                    ci_high=float(ciHB),
                    formula=formulaB,
                    n_obs=int(resB.nobs),
                )
            except Exception as e:
                placeboB = PlaceboResult(ok=False, message=f"Model failed: {e}")

if placeboB.ok:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Placebo B interaction coef", f"{placeboB.coef:.4f}")
    b2.metric("Std. Error", f"{placeboB.se:.4f}")
    b3.metric("p-value", f"{placeboB.pval:.4g}")
    b4.metric(f"{int((1-alpha)*100)}% CI", f"[{placeboB.ci_low:.4f}, {placeboB.ci_high:.4f}]")
    st.caption(f"SE type: {placeboB.se_type} | Model: {placeboB.formula} | N={placeboB.n_obs}")
    st.markdown(
        "**Interpretation (Placebo B):** Uses **only pre-treatment data** and inserts a fake cutoff. "
        "If treated and control already diverge within the pre-period, this can show up as a significant placebo effect. "
        + (f"Here it is {'significant' if placeboB.pval < alpha else 'not significant'} at α={alpha:.2f}.")
    )
    with st.expander("Compact regression table (Placebo B)", expanded=False):
        st.dataframe(compact_reg_table(placeboB.res, alpha=alpha), use_container_width=True)
else:
    st.info(f"Placebo B not run: {placeboB.message}")

st.write("---")

# Visual requested: show placebo effects together with main DiD
st.subheader("6) Visual Diagnostic — Main DiD vs Placebo Effects")

if post_start is None or len(times) < 3:
    st.info("Visual diagnostic not available (need a real post period and multiple time points).")
else:
    show_curve = st.checkbox("Show placebo shift curve (all K)", value=True)

    if show_curve:
        sweep = run_placebo_shift_sweep(
            df_model=df_model,
            cov_cols=cov_cols,
            cluster=cluster,
            alpha=alpha,
            times=times,
            real_cutoff_time=post_start,
        )

        if sweep.empty:
            st.info("Could not compute placebo curve (many placebo cutoffs miss some 2×2 cells).")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(sweep["K_shift"], sweep["coef"], marker="o", label="Placebo A (treated×placebo_post)")
            ax.fill_between(sweep["K_shift"], sweep["ci_low"], sweep["ci_high"], alpha=0.2)

            ax.axhline(coef_main, linestyle="--", linewidth=1, label="Main DiD coef")
            ax.axhline(0, linestyle=":", linewidth=1, label="0 (expected if no pre-trends)")

            ax.set_xlabel("K (placebo starts K periods earlier than real cutoff)")
            ax.set_ylabel("Interaction coefficient estimate")
            ax.set_title("Placebo shift diagnostic (compare against 0 and main DiD)")
            ax.legend()

            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        with st.expander("Show placebo curve data table", expanded=False):
            st.dataframe(sweep, use_container_width=True)

st.divider()

# -----------------------------
# Plain-English Summary
# -----------------------------
st.subheader("7) Plain-English Summary")

st.markdown(
    f"""
**What the main DiD estimate means**  
- The DiD interaction term (**treated×post**) compares how the treated group changed from pre→post **relative** to how the control group changed from pre→post.  
- If the **parallel trends** assumption is reasonable, this is interpreted as the causal effect of the intervention on the treated group.

**How to read the placebo tests**  
- **Placebo A (shift post earlier):** If you see a big / significant placebo effect **before** the real intervention, that can be a warning sign of pre-trends.  
- **Placebo B (pre-only fake cutoff):** If treated and control already diverge within the pre period, the placebo interaction may be significant—again suggesting possible pre-trends.

**Quick rule of thumb**  
- Main DiD significant + placebo effects near 0 (not significant) ⇒ more consistent with parallel trends.  
- Placebo significant ⇒ investigate pre-trends, add controls, consider unit/time fixed effects (not implemented here), or reconsider identification.
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






   
  

