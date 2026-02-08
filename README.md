# Difference-in-Differences App 📊

An interactive Streamlit dashboard for causal inference using
Difference-in-Differences (DiD).

## 🚀 Live App
👉 https://difference-in-differences-app.streamlit.app

## Features
- DiD regression with robust or clustered standard errors
- Manual 2×2 DiD sanity check
- Parallel trends visualization
- Placebo tests (shifted cutoff + pre-period fake cutoff)
- Visual diagnostics comparing main DiD vs placebo effects

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
