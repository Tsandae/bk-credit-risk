"""BK Credit-Risk Monitoring Dashboard (Final Project).

Run locally:   streamlit run dashboard/monitoring_app.py
Deploy:        Streamlit Community Cloud as a second app, main file path = dashboard/monitoring_app.py
Reads dashboard/data/monitoring.json, produced by:  python -m src.final.monitor
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.set_page_config(page_title="BK Credit-Risk Monitoring", page_icon="📈", layout="wide")
st.markdown("""<style>
:root {color-scheme: light;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:#ffffff !important;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {color:#0b2545 !important;}
.note {background:#eef3f8; border-left:4px solid #2a78d6; padding:.7rem 1rem; border-radius:4px; margin:.5rem 0 1rem 0;}
.warn {background:#fff4ec; border-left:4px solid #eb6834; padding:.7rem 1rem; border-radius:4px; margin:.5rem 0 1rem 0;}
.note, .note *, .warn, .warn * {color:#0b2545 !important;}
</style>""", unsafe_allow_html=True)

import monitoring_view  # noqa: E402

monitoring_view.render()
