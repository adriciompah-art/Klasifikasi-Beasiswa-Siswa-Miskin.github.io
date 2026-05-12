# =========================================================
# INSTALL:  pip install streamlit openpyxl scikit-learn
#           pandas numpy seaborn matplotlib
# RUN    :  streamlit run klasifikasi_bsm.py
# =========================================================

import io
import base64

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Klasifikasi BSM – Naive Bayes",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS — Vibrant Education Theme, Fixed Layout
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── COLOUR TOKENS ──────────────────────────────────── */
    :root {
        --bg:           #F0F4FF;
        --surface:      #FFFFFF;
        --surface2:     #F5F7FF;
        --surface3:     #E8EEFF;
        --border:       #C5D0F0;
        --border2:      #9AAEE0;

        --navy:         #1648C8;
        --navy-mid:     #1E56E0;
        --navy-light:   #4D7EFF;
        --navy-glow:    rgba(22,72,200,0.25);

        --teal:         #00B896;
        --teal-light:   #00D9B0;
        --teal-pale:    #C0FFF2;
        --teal-glow:    rgba(0,184,150,0.30);

        --amber:        #FF8C00;
        --amber-light:  #FFAD33;
        --amber-pale:   #FFF3DB;
        --amber-glow:   rgba(255,140,0,0.30);

        --rose:         #FF2D6E;
        --rose-light:   #FF5C8D;
        --rose-pale:    #FFE0EB;
        --rose-glow:    rgba(255,45,110,0.25);

        --violet:       #7C3AED;
        --violet-light: #9F67FF;
        --violet-pale:  #EDE9FE;
        --violet-glow:  rgba(124,58,237,0.25);

        --cyan:         #00BFFF;
        --cyan-pale:    #DAFAFF;

        --text:         #080F1E;
        --text-body:    #1A2540;
        --text-dim:     #3D5070;
        --text-muted:   #6B80A8;

        --sidebar-w:    300px;
        --header-h:     0px;

        --shadow-sm:    0 2px 8px rgba(22,72,200,0.10), 0 1px 3px rgba(22,72,200,0.06);
        --shadow-md:    0 6px 24px rgba(22,72,200,0.14), 0 2px 8px rgba(22,72,200,0.08);
        --shadow-lg:    0 12px 48px rgba(22,72,200,0.18), 0 4px 16px rgba(22,72,200,0.10);
        --shadow-glow:  0 0 0 3px rgba(22,72,200,0.15);

        --radius:       12px;
        --radius-lg:    18px;
        --radius-xl:    24px;
    }

    /* ── RESET BASE ─────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text-body);
        font-size: 14px;
        line-height: 1.65;
    }

    /* ── SIDEBAR — Fixed, full height ──────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(170deg, #060E24 0%, #0D1E48 40%, #0A1835 100%) !important;
        border-right: 1px solid rgba(78,120,255,0.18) !important;
        box-shadow: 4px 0 32px rgba(6,14,36,0.35) !important;
        width: var(--sidebar-w) !important;
        min-width: var(--sidebar-w) !important;
        max-width: var(--sidebar-w) !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0 !important;
        background: transparent !important;
    }
    section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.82) !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

    /* Sidebar file uploader */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(78,120,255,0.08) !important;
        border: 2px dashed rgba(0,217,176,0.45) !important;
        border-radius: 12px !important;
        transition: all 0.25s !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(0,217,176,0.80) !important;
        background: rgba(0,184,150,0.12) !important;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #00B896, #00D9B0) !important;
        color: #060E24 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        font-size: 0.82rem !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 4px 16px rgba(0,184,150,0.40) !important;
        transition: all 0.22s !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #00D9B0, #00FFD0) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0,184,150,0.50) !important;
    }

    /* ── SIDEBAR LOGO AREA ──────────────────────────────── */
    .sb-logo {
        padding: 1.6rem 1.4rem 1.2rem;
        background: linear-gradient(135deg, rgba(22,72,200,0.25), rgba(0,184,150,0.12));
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.5rem;
    }
    .sb-logo-row { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.6rem; }
    .sb-logo-icon {
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #1648C8, #00B896);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,184,150,0.35);
        flex-shrink: 0;
    }
    .sb-logo-title { font-family: 'Fraunces', serif; font-size: 1.3rem; color: #FFFFFF !important; line-height: 1.1; }
    .sb-logo-title span { color: #00D9B0 !important; }
    .sb-logo-desc { font-size: 0.68rem !important; color: rgba(255,255,255,0.38) !important; letter-spacing: 0.3px; }

    /* ── SIDEBAR SECTION LABEL ──────────────────────────── */
    .sb-sec {
        font-size: 0.62rem !important;
        font-weight: 800 !important;
        letter-spacing: 2.2px !important;
        text-transform: uppercase !important;
        color: #00D9B0 !important;
        margin: 1.4rem 0 0.6rem !important;
        display: block;
        padding-left: 0.1rem;
    }

    /* ── PIPELINE STEPS ─────────────────────────────────── */
    .pipeline { display: flex; flex-direction: column; gap: 0.3rem; }
    .pipeline-step {
        display: flex; align-items: center; gap: 0.65rem;
        padding: 0.55rem 0.9rem;
        border-radius: 10px;
        border: 1px solid transparent;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: rgba(255,255,255,0.28) !important;
        transition: all 0.2s;
    }
    .pipeline-step.done {
        background: rgba(0,184,150,0.18) !important;
        border-color: rgba(0,217,176,0.40) !important;
        color: #00D9B0 !important;
    }
    .pipeline-step.active {
        background: rgba(78,120,255,0.15) !important;
        border-color: rgba(78,120,255,0.35) !important;
        color: rgba(255,255,255,0.85) !important;
    }
    .step-num {
        width: 22px; height: 22px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.68rem; font-weight: 800;
        flex-shrink: 0;
        background: rgba(255,255,255,0.08);
        color: rgba(255,255,255,0.30) !important;
    }
    .step-num.done { background: #00B896; color: #fff !important; box-shadow: 0 0 10px rgba(0,184,150,0.50); }
    .step-num.active { background: rgba(78,120,255,0.55); color: #fff !important; }

    /* ── MAIN CONTENT — no fixed header, normal flow ──── */
    .main .block-container {
    padding-top: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 100% !important;
}

    /* ── INLINE PAGE TITLE BAR ──────────────────────────── */
/* ── INLINE PAGE TITLE BAR ──────────────────────────── */
.page-titlebar {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: linear-gradient(110deg, #080F1E 0%, #0F1E42 55%, #091E45 100%);
    border-radius: var(--radius-lg);
    padding: 1rem 1.6rem;
    margin-bottom: 1.4rem;
    border: 1px solid rgba(0,184,150,0.22);
    box-shadow: 0 4px 24px rgba(8,15,30,0.18), inset 0 1px 0 rgba(255,255,255,0.05);
    position: relative; overflow: hidden;
}
    .page-titlebar::before {
        content: '';
        position: absolute; inset: 0;
        background: radial-gradient(ellipse 50% 100% at 90% 50%, rgba(0,184,150,0.10) 0%, transparent 65%);
        pointer-events: none;
    }
    .ptb-badge {
        width: 42px; height: 42px; flex-shrink: 0;
        background: linear-gradient(135deg, #1648C8, #00B896);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
        box-shadow: 0 3px 14px rgba(0,184,150,0.38);
        position: relative; z-index: 1;
    }
    .ptb-text { position: relative; z-index: 1; flex: 1; }
    .ptb-eyebrow {
        font-size: 0.58rem; font-weight: 800; letter-spacing: 2.5px;
        text-transform: uppercase; color: #00D9B0; margin-bottom: 0.15rem;
    }
    .ptb-title {
        font-family: 'Fraunces', serif; font-size: 1.22rem; font-weight: 700;
        color: #FFFFFF; line-height: 1.1; letter-spacing: -0.2px;
    }
    .ptb-title .hl { color: #FFD166; font-style: italic; }
    .ptb-pills {
        margin-left: auto; display: flex; gap: 0.4rem; align-items: center;
        position: relative; z-index: 1; flex-wrap: wrap;
    }
    .hpill {
        border-radius: 999px; padding: 4px 12px;
        font-size: 0.68rem; font-weight: 800; letter-spacing: 0.3px;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .hpill-teal  { background: rgba(0,184,150,0.22);  color: #00FFD0 !important; }
    .hpill-blue  { background: rgba(22,72,200,0.35);   color: #7AADFF !important; }
    .hpill-amber { background: rgba(255,140,0,0.25);   color: #FFD166 !important; }

    /* ── NOTIFICATION BANNERS ───────────────────────────── */
    .notif-box {
        border-radius: var(--radius);
        padding: 1rem 1.3rem;
        margin: 0.8rem 0 1.2rem;
        display: flex;
        align-items: flex-start;
        gap: 0.9rem;
        border: 1px solid;
        animation: slideIn 0.35s ease;
    }
    @keyframes slideIn {
        from { opacity:0; transform: translateY(-8px); }
        to   { opacity:1; transform: translateY(0); }
    }
    .notif-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 0.1rem; }
    .notif-content { flex: 1; }
    .notif-title { font-weight: 800; font-size: 0.88rem; margin-bottom: 0.2rem; }
    .notif-body { font-size: 0.8rem; line-height: 1.6; }
    .notif-stats {
        display: flex; gap: 0.6rem; margin-top: 0.6rem; flex-wrap: wrap;
    }
    .notif-stat {
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 0.73rem;
        font-weight: 700;
        display: inline-flex; align-items: center; gap: 5px;
    }

    /* Notif variants */
    .notif-upload {
        background: linear-gradient(135deg, #EEF3FF, #E8F8FF);
        border-color: rgba(22,72,200,0.25);
    }
    .notif-upload .notif-title { color: #1648C8; }
    .notif-upload .notif-stat { background: rgba(22,72,200,0.10); color: #1648C8; }

    .notif-prep {
        background: linear-gradient(135deg, #ECFDF9, #E0FFF7);
        border-color: rgba(0,184,150,0.30);
    }
    .notif-prep .notif-title { color: #00896F; }
    .notif-prep .notif-stat { background: rgba(0,184,150,0.12); color: #00896F; }

    .notif-nb {
        background: linear-gradient(135deg, #F5F0FF, #EDE9FE);
        border-color: rgba(124,58,237,0.25);
    }
    .notif-nb .notif-title { color: #6D28D9; }
    .notif-nb .notif-stat { background: rgba(124,58,237,0.10); color: #6D28D9; }

    .notif-warn {
        background: linear-gradient(135deg, #FFFBEE, #FFF6DC);
        border-color: rgba(255,140,0,0.30);
    }
    .notif-warn .notif-title { color: #C26000; }
    .notif-warn .notif-stat { background: rgba(255,140,0,0.12); color: #C26000; }

    /* ── METRIC CARDS ───────────────────────────────────── */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 140px;
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: 1.3rem 1.4rem 1.1rem;
        position: relative; overflow: hidden;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border);
        transition: transform 0.22s, box-shadow 0.22s;
    }
    .metric-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-md); }
    .metric-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0;
        height: 4px;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }
    .metric-card::after {
        content: '';
        position: absolute; bottom: -20px; right: -20px;
        width: 80px; height: 80px; border-radius: 50%;
        opacity: 0.06;
    }
    .mc-navy::before   { background: linear-gradient(90deg, #1648C8, #4D7EFF); }
    .mc-navy::after    { background: #1648C8; }
    .mc-teal::before   { background: linear-gradient(90deg, #00B896, #00FFD0); }
    .mc-teal::after    { background: #00B896; }
    .mc-amber::before  { background: linear-gradient(90deg, #FF8C00, #FFD166); }
    .mc-amber::after   { background: #FF8C00; }
    .mc-violet::before { background: linear-gradient(90deg, #7C3AED, #9F67FF); }
    .mc-violet::after  { background: #7C3AED; }
    .mc-rose::before   { background: linear-gradient(90deg, #FF2D6E, #FF5C8D); }
    .mc-rose::after    { background: #FF2D6E; }

    .mc-icon { font-size: 1.3rem; margin-bottom: 0.55rem; display: block; }
    .mc-label { font-size: 0.64rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700; margin-bottom: 0.3rem; }
    .mc-value { font-family: 'Fraunces', serif; font-size: 2.3rem; line-height: 1; }
    .cv-navy   { color: #1648C8; }
    .cv-teal   { color: #00B896; }
    .cv-amber  { color: #FF8C00; }
    .cv-violet { color: #7C3AED; }
    .cv-rose   { color: #FF2D6E; }

    /* ── SECTION HEADER ─────────────────────────────────── */
    .section-header {
        font-family: 'Fraunces', serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text);
        padding: 0.4rem 0 0.4rem 1rem;
        border-left: 4px solid var(--teal);
        margin-bottom: 1.2rem;
        line-height: 1.2;
        background: linear-gradient(90deg, rgba(0,184,150,0.06), transparent);
        border-radius: 0 8px 8px 0;
    }

    /* ── INFO BOX ───────────────────────────────────────── */
    .info-box {
        background: linear-gradient(135deg, #EEF4FF, #E8FFF9);
        border: 1px solid rgba(22,72,200,0.16);
        border-left: 3px solid #4D7EFF;
        border-radius: var(--radius);
        padding: 0.9rem 1.2rem;
        margin-bottom: 1.1rem;
        font-size: 0.83rem;
        color: #1A2540;
        line-height: 1.7;
    }

    /* ── FORMULA BOX ────────────────────────────────────── */
    .formula-box {
        background: linear-gradient(135deg, #FFFBEE, #FFF8F0);
        border: 1px solid rgba(255,140,0,0.20);
        border-left: 3px solid #FF8C00;
        border-radius: var(--radius);
        padding: 1rem 1.3rem;
        margin: 0.9rem 0;
        font-size: 0.82rem;
        color: var(--text-body);
        line-height: 1.9;
    }
    .formula-box code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        background: rgba(255,140,0,0.12);
        border-radius: 5px;
        padding: 0.12rem 0.45rem;
        color: #C05A00;
    }
    .formula-title {
        font-weight: 800;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        color: #FF8C00;
        margin-bottom: 0.6rem;
    }

    /* ── SPLIT CONFIG BOX ───────────────────────────────── */
    .split-config-box {
        background: linear-gradient(135deg, #0F1E42, #1A2E58);
        border: 1px solid rgba(0,217,176,0.25);
        border-radius: var(--radius-lg);
        padding: 1.4rem 1.6rem;
        margin: 1rem 0 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .split-config-box::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #1648C8, #00B896, #FF8C00);
    }
    .split-config-title {
        font-family: 'Fraunces', serif;
        font-size: 1rem;
        color: #FFFFFF;
        margin-bottom: 0.3rem;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .split-config-desc {
        font-size: 0.74rem;
        color: rgba(255,255,255,0.45);
        margin-bottom: 1.1rem;
    }
    .split-pills-row {
        display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1rem;
    }
    .split-pill {
        border-radius: 999px;
        padding: 6px 18px;
        font-size: 0.76rem;
        font-weight: 800;
        border: 2px solid rgba(255,255,255,0.12);
        color: rgba(255,255,255,0.45);
        background: rgba(255,255,255,0.04);
        cursor: pointer;
        transition: all 0.2s;
        letter-spacing: 0.3px;
    }
    .split-pill.active {
        border-color: #00D9B0;
        color: #00FFD0;
        background: rgba(0,184,150,0.18);
        box-shadow: 0 0 12px rgba(0,184,150,0.25);
    }
    .split-bar-wrap {
        background: rgba(255,255,255,0.06);
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }
    .split-bar-train {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }
    .split-bar-labels {
        display: flex; justify-content: space-between;
        font-size: 0.68rem; font-weight: 700;
    }
    .split-label-train { color: #00D9B0; }
    .split-label-test  { color: #FF5C8D; }

    /* ── TABS ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border-bottom: 2px solid var(--border);
        gap: 0; padding: 0 0.5rem;
        border-radius: var(--radius) var(--radius) 0 0;
        box-shadow: var(--shadow-sm);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-muted);
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 0.76rem;
        padding: 0.9rem 1.1rem;
        border: none;
        border-bottom: 3px solid transparent;
        margin-bottom: -2px;
        letter-spacing: 0.2px;
        transition: color 0.2s, border-color 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #1648C8; }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #00B896 !important;
        border-bottom: 3px solid #00B896 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 var(--radius) var(--radius);
        padding: 1.8rem 2rem;
    }

    /* ── DATAFRAME ──────────────────────────────────────── */
    .stDataFrame { border-radius: var(--radius) !important; overflow: hidden !important; border: 1px solid var(--border) !important; box-shadow: var(--shadow-sm) !important; }
    .stDataFrame thead th { background: linear-gradient(90deg, #0F1E42, #1648C8) !important; color: #FFFFFF !important; font-weight: 700 !important; font-size: 0.73rem !important; letter-spacing: 0.5px !important; }
    .stDataFrame tbody tr:nth-child(even) td { background: rgba(22,72,200,0.03) !important; }
    .stDataFrame tbody tr:hover td { background: rgba(0,184,150,0.06) !important; }

    /* ── MAIN BUTTONS ───────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #1648C8, #4D7EFF);
        color: #fff;
        border: none;
        border-radius: var(--radius);
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 0.84rem;
        padding: 0.65rem 1.5rem;
        cursor: pointer;
        transition: all 0.22s;
        width: 100%;
        box-shadow: 0 4px 16px rgba(22,72,200,0.35);
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4D7EFF, #7C3AED);
        box-shadow: 0 8px 28px rgba(77,126,255,0.45);
        transform: translateY(-2px);
    }

    /* ── PRED RESULT ─────────────────────────────────────── */
    .pred-result-box {
        border-radius: var(--radius-lg);
        padding: 1.8rem 2rem;
        margin-top: 1rem;
        text-align: center;
        border: 2px solid;
        position: relative; overflow: hidden;
    }
    .pred-result-box::before {
        content: '';
        position: absolute; inset: 0;
        background-image: radial-gradient(rgba(255,255,255,0.6) 1px, transparent 1px);
        background-size: 20px 20px;
        pointer-events: none;
        opacity: 0.3;
    }
    .pred-layak { background: linear-gradient(135deg, #E8FFF9, #D0FFE8); border-color: rgba(0,184,150,0.40); }
    .pred-tidak { background: linear-gradient(135deg, #FFF0F5, #FFE0EB); border-color: rgba(255,45,110,0.35); }
    .pred-label { font-family: 'Fraunces', serif; font-size: 2rem; font-weight: 700; position: relative; }
    .pred-sub { font-size: 0.79rem; color: var(--text-dim); margin-top: 0.5rem; position: relative; }

    /* ── DOWNLOAD LINK ──────────────────────────────────── */
    .dl-btn {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: var(--surface3);
        color: #1648C8 !important;
        text-decoration: none;
        border: 1.5px solid var(--border2);
        border-radius: var(--radius);
        padding: 0.48rem 1.2rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 0.78rem;
        transition: all 0.2s;
        margin-top: 0.4rem;
    }
    .dl-btn:hover { border-color: #00B896; background: #C0FFF2; color: #007A62 !important; }

    /* ── MISC ───────────────────────────────────────────── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--surface2); }
    ::-webkit-scrollbar-thumb { background: #A0B0D0; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #1648C8; }
    [data-testid="stAlert"] { border-radius: var(--radius) !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }
    footer { display: none !important; }
    #MainMenu { visibility: hidden; }

    /* ── SELECT/NUMBER INPUTS ───────────────────────────── */
    .stSelectbox label, .stNumberInput label, .stTextInput label, .stRadio label {
        color: var(--text-body) !important; font-weight: 700 !important; font-size: 0.81rem !important;
    }
    .stTextInput input {
        background: var(--surface) !important;
        border: 1.5px solid var(--border2) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.875rem !important;
    }
    .stTextInput input:focus { border-color: #1648C8 !important; box-shadow: 0 0 0 3px rgba(22,72,200,0.14) !important; }

    /* Slider styling */
    .stSlider > div > div > div { background: linear-gradient(90deg, #00B896, #1648C8) !important; }
    .stSlider label { color: var(--text-body) !important; font-weight: 700 !important; font-size: 0.81rem !important; }

    /* Sembunyikan tombol collapse sidebar */
    button[kind="header"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    button[data-testid="baseButton-header"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SPLASH SCREEN (WELCOME PAGE) — Enhanced CSS & Animations
# =========================================================

if "page_mode" not in st.session_state:
    st.session_state.page_mode = "welcome"

if st.session_state.page_mode == "welcome":
    # CSS khusus untuk tampilan welcome yang lebih menarik
    st.markdown(
        """
        <style>
        /* Sembunyikan sidebar dan elemen default */
        section[data-testid="stSidebar"] { display: none !important; }
        header[data-testid="stHeader"] { display: none !important; }
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 1200px !important;
        }
        /* Background gradient yang halus */
        .stApp {
            background: linear-gradient(135deg, #F0F7FF 0%, #E9F0FA 100%);
        }
        /* Container utama dashboard */
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 85vh;
            text-align: center;
            padding: 2rem;
            position: relative;
            z-index: 1;
        }
        /* Efek ornamen lingkaran latar belakang */
        .bg-ornament {
            position: fixed;
            top: -20%;
            right: -10%;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(0,184,150,0.08) 0%, rgba(0,184,150,0) 70%);
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
        }
        .bg-ornament2 {
            position: fixed;
            bottom: -15%;
            left: -10%;
            width: 450px;
            height: 450px;
            background: radial-gradient(circle, rgba(22,72,200,0.06) 0%, rgba(22,72,200,0) 70%);
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
        }
        /* Kartu utama dengan efek glassmorphism */
        .hero-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            border-radius: 48px;
            padding: 2.5rem 3rem;
            box-shadow: 0 25px 45px -12px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.6);
            border: 1px solid rgba(255,255,255,0.4);
            max-width: 850px;
            width: 100%;
            transition: transform 0.3s ease;
            z-index: 2;
        }
        .hero-card:hover {
            transform: translateY(-5px);
        }
        /* Ikon animasi */
        .icon-pulse {
            background: linear-gradient(135deg, #1648C8, #00B896);
            width: 100px;
            height: 100px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.5rem;
            box-shadow: 0 15px 35px rgba(22,72,200,0.3);
            animation: float 3s ease-in-out infinite;
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-12px); }
            100% { transform: translateY(0px); }
        }
        .icon-pulse span {
            font-size: 3.5rem;
            filter: drop-shadow(2px 4px 6px rgba(0,0,0,0.2));
        }
        /* Judul utama dengan gradien */
        .hero-title {
            font-family: 'Fraunces', serif;
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #0F1E42, #1648C8, #00B896);
            background-size: 200% auto;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 0.5rem;
            animation: shine 6s linear infinite;
        }
        @keyframes shine {
            0% { background-position: 0% 50%; }
            100% { background-position: 200% 50%; }
        }
        .hero-sub {
            font-family: 'Fraunces', serif;
            font-size: 1.9rem;
            font-weight: 600;
            color: #1A2540;
            margin-bottom: 1rem;
        }
        .hero-desc {
            font-size: 1.05rem;
            color: #3D5070;
            max-width: 600px;
            margin: 0 auto 1.8rem;
            line-height: 1.6;
        }
        /* Pills / tags */
        .pill-group {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            justify-content: center;
            margin-bottom: 2rem;
        }
        .pill {
            background: rgba(22,72,200,0.1);
            backdrop-filter: blur(4px);
            padding: 0.5rem 1.4rem;
            border-radius: 60px;
            font-weight: 600;
            color: #1648C8;
            border: 1px solid rgba(22,72,200,0.2);
            transition: all 0.2s;
            font-size: 0.85rem;
        }
        .pill:hover {
            background: rgba(0,184,150,0.15);
            color: #00896F;
            border-color: rgba(0,184,150,0.4);
            transform: translateY(-2px);
        }
        /* Tombol utama */
        .btn-start {
            background: linear-gradient(105deg, #1648C8, #00B896);
            border: none;
            padding: 0.85rem 2.2rem;
            font-size: 1.1rem;
            font-weight: 700;
            border-radius: 48px;
            color: white;
            cursor: pointer;
            transition: all 0.25s ease;
            box-shadow: 0 8px 20px rgba(22,72,200,0.3);
            width: 100%;
            letter-spacing: 0.5px;
        }
        .btn-start:hover {
            transform: scale(1.02);
            box-shadow: 0 12px 28px rgba(22,72,200,0.4);
            background: linear-gradient(105deg, #1E56E0, #00D9B0);
        }
        /* footer kecil */
        .welcome-footer {
            margin-top: 2rem;
            font-size: 0.75rem;
            color: #8EA0C0;
            text-align: center;
        }
        </style>
        <div class="bg-ornament"></div>
        <div class="bg-ornament2"></div>
        <div class="welcome-container">
            <div class="hero-card">
                <div class="icon-pulse">
                    <span>🎓</span>
                </div>
                <div class="hero-title">Sistem Pendukung Keputusan</div>
                <div class="hero-sub">Klasifikasi Beasiswa Siswa Miskin</div>
                <div class="hero-desc">
                    Menggunakan algoritma <strong>Gaussian Naive Bayes</strong> untuk menentukan kelayakan penerima BSM secara objektif dan akurat berdasarkan data ekonomi keluarga.
                </div>
                <div class="pill-group">
                    <div class="pill">📊 Akurasi Tinggi</div>
                    <div class="pill">🤖 Machine Learning</div>
                    <div class="pill">🎯 Tepat Sasaran</div>
                    <div class="pill">📈 Evaluasi Otomatis</div>
                    <div class="pill">⚡ Real-time</div>
                </div>
                <div style="margin-top: 0.5rem;">
        """,
        unsafe_allow_html=True,
    )

    # Tombol dengan Streamlit (tetap menggunakan st.button agar responsif)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if st.button("🚀 Mulai Klasifikasi", use_container_width=True, key="start_btn"):
            st.session_state.page_mode = "main"
            st.rerun()

    st.markdown(
        """
                </div>
                <div class="welcome-footer">
                    © 2025 · BSM Classifier · Gaussian Naive Bayes
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

    st.markdown(
    """
    <style>
    /* Hilangkan scroll horizontal di seluruh halaman */
    html, body, .stApp, .main, .block-container, .element-container, .row-widget, .stMarkdown {
        overflow-x: hidden !important;
        max-width: 100% !important;
    }
    /* Pastikan sidebar tidak menyebabkan scroll */
    section[data-testid="stSidebar"] {
        overflow-x: hidden !important;
        width: 300px !important;
    }
    /* Header menjadi fixed secara horizontal */
    .page-titlebar {
        position: relative !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        margin: 0 !important;
        transform: none !important;
        box-sizing: border-box !important;
        /* Jika ingin sticky vertikal, tambahkan baris di bawah */
        position: sticky !important;
        top: 0 !important;
        z-index: 1000 !important;
    }
    /* Konten utama tidak melebihi lebar */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        overflow-x: hidden !important;
    }
    /* Tabel dan dataframe bisa di-scroll internal */
    .stDataFrame, .stTable, [data-testid="stImage"] {
        overflow-x: auto !important;
        max-width: 100% !important;
        display: block !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# FIXED HEADER
# =========================================================

st.markdown(
    """
    <div class="page-titlebar">
        <div class="ptb-badge">🎓</div>
        <div class="ptb-text">
            <div class="ptb-eyebrow">Sistem Pendukung Keputusan · Gaussian Naive Bayes</div>
            <div class="ptb-title">Klasifikasi <span class="hl">Beasiswa Siswa Miskin</span></div>
        </div>
        <div class="ptb-pills">
            <div class="hpill hpill-teal">🤖 ML</div>
            <div class="hpill hpill-blue">📊 Gaussian NB</div>
            <div class="hpill hpill-amber">🎒 BSM</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE
# =========================================================

for key in ["raw", "df_clean", "nb_results", "data_tambahan", "notif_upload", "notif_prep", "notif_nb", "split_ratio"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state["data_tambahan"] is None:
    st.session_state["data_tambahan"] = pd.DataFrame(
        columns=["NAMA SISWA","KELAS","NIS","PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL (PREDIKSI)"]
    )

# Default split ratio 80/20
if st.session_state["split_ratio"] is None:
    st.session_state["split_ratio"] = 80


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def make_excel_download(df: pd.DataFrame, filename: str, label: str, judul: str = "") -> str:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        if judul:
            title_df = pd.DataFrame([[judul]], columns=[""])
            title_df.to_excel(w, index=False, startrow=0)
            df.to_excel(w, index=False, startrow=2)
        else:
            df.to_excel(w, index=False)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<a class="dl-btn" href="data:application/vnd.openxmlformats-officedocument'
        f'.spreadsheetml.sheet;base64,{b64}" download="{filename}">{label}</a>'
    )


def make_png_download(fig, filename: str, label: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<a class="dl-btn" href="data:image/png;base64,{b64}" download="{filename}">{label}</a>'
    )


def notif_html(variant: str, icon: str, title: str, body: str, stats: list = None) -> str:
    stats_html = ""
    if stats:
        stats_html = '<div class="notif-stats">'
        for s in stats:
            stats_html += f'<span class="notif-stat">{s}</span>'
        stats_html += '</div>'
    return (
        f'<div class="notif-box notif-{variant}">'
        f'<div class="notif-icon">{icon}</div>'
        f'<div class="notif-content">'
        f'<div class="notif-title">{title}</div>'
        f'<div class="notif-body">{body}</div>'
        f'{stats_html}'
        f'</div></div>'
    )


# =========================================================
# NORMALISASI
# =========================================================

def kategori_pendapatan(x):
    x = str(x).lower().replace(".", "").replace("rp", "").strip()
    angka = "".join(filter(str.isdigit, x))
    if angka == "": return np.nan
    angka = int(angka)
    if 500_000 <= angka <= 1_000_000: return 0
    if 3_000_000 <= angka <= 4_000_000: return 1
    if 5_000_000 <= angka <= 7_000_000: return 2
    return np.nan

def kategori_pekerjaan(x):
    mapping = {"petani": 0, "pns": 1, "polisi": 2, "wiraswasta": 3}
    return mapping.get(str(x).lower().strip(), np.nan)

def kategori_rumah(x):
    x = str(x).lower().strip()
    if x == "milik sendiri": return 1
    if x in ["kontrak", "sewa", "kontrak/sewa"]: return 0
    return np.nan

def kategori_label(x):
    x = str(x).lower().strip()
    if x == "ya": return 1
    if x == "tidak": return 0
    return np.nan


# =========================================================
# LOAD DATA
# =========================================================

def load_data(file) -> tuple:
    try:
        if file.name.endswith(".csv"):
            raw = pd.read_csv(file, header=None)
        elif file.name.endswith(".xlsx"):
            raw = pd.read_excel(file, header=None)
        else:
            return None, "❌ Format file tidak didukung (gunakan .csv atau .xlsx)", {}
        info = {
            "rows": len(raw),
            "cols": len(raw.columns),
            "data_rows": max(0, len(raw) - 3),
            "filename": file.name,
            "size_kb": round(file.size / 1024, 1) if hasattr(file, 'size') else "?",
        }
        return raw, "ok", info
    except Exception as e:
        return None, f"❌ Error membaca file: {e}", {}


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_data(raw: pd.DataFrame) -> tuple:
    try:
        before_count = max(0, len(raw) - 3)
        header = raw.iloc[2]
        df = raw.iloc[3:].copy()
        df.columns = header
        df.reset_index(drop=True, inplace=True)
        df.columns = df.columns.astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()

        col_map = {}
        for col in df.columns:
            if "PENDAPATAN" in col: col_map["pendapatan"] = col
            if "PEKERJAAN"  in col: col_map["pekerjaan"]  = col
            if "TANGGUNGAN" in col: col_map["tanggungan"]  = col
            if "RUMAH"      in col: col_map["rumah"]       = col
            if "LABEL"      in col: col_map["label"]       = col

        before_na = len(df)
        df[col_map["pendapatan"]] = df[col_map["pendapatan"]].apply(kategori_pendapatan)
        df[col_map["pekerjaan"]]  = df[col_map["pekerjaan"]].apply(kategori_pekerjaan)
        df[col_map["tanggungan"]] = pd.to_numeric(df[col_map["tanggungan"]], errors="coerce")
        df[col_map["rumah"]]      = df[col_map["rumah"]].apply(kategori_rumah)
        df[col_map["label"]]      = df[col_map["label"]].apply(kategori_label)

        df.rename(columns={
            col_map["pendapatan"]: "PENDAPATAN ORANG TUA",
            col_map["pekerjaan"]:  "PEKERJAAN ORANG TUA",
            col_map["tanggungan"]: "JUMLAH TANGGUNGAN",
            col_map["rumah"]:      "STATUS RUMAH",
            col_map["label"]:      "LABEL",
        }, inplace=True)

        df.dropna(subset=["PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL"], inplace=True)
        after_count  = len(df)
        dropped      = before_na - after_count
        layak_count  = int((df["LABEL"] == 1).sum())
        tidak_count  = int((df["LABEL"] == 0).sum())

        info = {
            "before": before_count,
            "after": after_count,
            "dropped": dropped,
            "layak": layak_count,
            "tidak": tidak_count,
        }
        return df, "ok", info
    except Exception as e:
        return None, f"❌ Error preprocessing: {e}", {}


# =========================================================
# GAUSSIAN PROBABILITY & NAIVE BAYES
# =========================================================

def gaussian_probability(x, mean, std):
    if std == 0 or pd.isna(std): std = 0.0001
    exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


def naive_bayes_process(df: pd.DataFrame, train_pct: int = 80) -> dict:
    """
    train_pct : int, persentase data training (0–100).
                Jika 100, seluruh data dipakai untuk training & testing (full-train).
    """
    fitur  = ["PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH"]
    target = "LABEL"

    X, y = df[fitur], df[target]

    # ── Pembagian data ───────────────────────────────────
    if train_pct >= 100:
        # 100% training, evaluasi pada data training itu sendiri
        X_train, X_test = X.copy(), X.copy()
        y_train, y_test = y.copy(), y.copy()
        split_mode = "full_train"
    else:
        test_size = (100 - train_pct) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        split_mode = "split"

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    classes = np.unique(y_train)
    prior, mean, std = {}, {}, {}
    for c in classes:
        prior[c] = len(y_train[y_train == c]) / len(y_train)
        data_c   = X_train[y_train == c]
        mean[c]  = data_c.mean()
        std[c]   = data_c.std()

    tabel_prior = pd.DataFrame({"Kelas": list(prior.keys()), "Prior Probability": list(prior.values())})

    hasil_manual = []
    for i in range(len(X_test)):
        row = X_test.iloc[i]
        probs = {}
        for c in classes:
            probs[c] = prior[c]
            for f in fitur:
                probs[c] *= gaussian_probability(row[f], mean[c][f], std[c][f])
        prediksi = max(probs, key=probs.get)
        hasil_manual.append({
            "Data Ke": i + 1,
            "P(Tidak Layak=0)": probs.get(0, 0),
            "P(Layak=1)":       probs.get(1, 0),
            "Prediksi":         prediksi,
            "Aktual":           y_test.iloc[i],
            "Benar?":           "✅" if prediksi == y_test.iloc[i] else "❌",
        })

    hasil_manual_df = pd.DataFrame(hasil_manual)
    cm       = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).transpose()

    # ── Chart theme ─────────────────────────────────────
    BG     = "#F5F7FF"
    NAVY   = "#0F1E42"
    TEAL   = "#00B896"
    TEAL_L = "#00D9B0"
    ROSE   = "#FF2D6E"
    AMBER  = "#FF8C00"
    MUTED  = "#6B80A8"
    BORDER = "#C5D0F0"
    TEXT   = "#080F1E"

    # Confusion Matrix chart
    fig_cm, ax = plt.subplots(figsize=(7, 5.5))
    fig_cm.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor(BG)

    n_rows, n_cols = cm.shape
    cell_colors = [[TEAL, ROSE], [ROSE, TEAL]]
    cell_labels  = [["TN", "FP"], ["FN", "TP"]]

    for i in range(n_rows):
        for j in range(n_cols):
            val      = cm[i, j]
            bg_color = cell_colors[i][j]
            rect = plt.Rectangle((j+0.06, i+0.06), 0.88, 0.88, color=bg_color, alpha=0.10, zorder=0)
            ax.add_patch(rect)
            border_rect = plt.Rectangle((j+0.06, i+0.06), 0.88, 0.88, fill=False, edgecolor=bg_color, alpha=0.45, linewidth=2.2, zorder=1)
            ax.add_patch(border_rect)
            ax.text(j+0.5, i+0.4,  str(val),           ha="center", va="center", fontsize=52, fontweight="bold", color=bg_color,        zorder=2, fontfamily="monospace")
            ax.text(j+0.5, i+0.74, cell_labels[i][j], ha="center", va="center", fontsize=12, fontweight="700",  color=bg_color, alpha=0.70, zorder=2)

    for k in range(1, n_rows): ax.axhline(k, color=BORDER, linewidth=1.5, zorder=3)
    for k in range(1, n_cols): ax.axvline(k, color=BORDER, linewidth=1.5, zorder=3)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER); spine.set_linewidth(1.5)

    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows); ax.invert_yaxis()
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Tidak Layak (0)", "Layak (1)"], color=TEXT, fontsize=11, fontweight="600")
    ax.set_yticklabels(["Tidak Layak (0)", "Layak (1)"], color=TEXT, fontsize=11, fontweight="600", rotation=0, va="center")
    ax.set_xlabel("Prediksi", color=MUTED, labelpad=14, fontsize=12)
    ax.set_ylabel("Aktual",   color=MUTED, labelpad=14, fontsize=12)
    split_label = f"Training {train_pct}% / Testing {100-train_pct}%"
    ax.set_title(f"Confusion Matrix — Hasil Evaluasi Model\n({split_label})", color=NAVY, fontsize=13, fontweight="bold", pad=22, fontfamily="DejaVu Serif")
    ax.tick_params(colors=BORDER, length=0)
    legend_items = [
        mpatches.Patch(color=TEAL, label="Prediksi Benar  (TN / TP)"),
        mpatches.Patch(color=ROSE, label="Prediksi Salah  (FP / FN)"),
    ]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              facecolor="#FFFFFF", edgecolor=BORDER, labelcolor=TEXT, fontsize=10)
    plt.tight_layout()

    benar = int((hasil_manual_df["Benar?"] == "✅").sum())
    salah = int((hasil_manual_df["Benar?"] == "❌").sum())

    return {
        "prior":      tabel_prior,
        "manual":     hasil_manual_df,
        "report":     report_df,
        "accuracy":   accuracy,
        "fig_cm":     fig_cm,
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "train_pct":  train_pct,
        "test_pct":   100 - train_pct,
        "split_mode": split_mode,
        "classes":    classes.tolist(),
        "model":      model,
        "cm":         cm,
        "mean":       mean,
        "std":        std,
        "prior_dict": prior,
        "benar":      benar,
        "salah":      salah,
    }


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sb-logo">
            <div class="sb-logo-row">
                <div class="sb-logo-icon">🎓</div>
                <div>
                    <div class="sb-logo-title">BSM <span>Classifier</span></div>
                    <div class="sb-logo-desc">Gaussian Naive Bayes</div>
                </div>
            </div>
            <div class="sb-logo-desc">Sistem Klasifikasi Kelayakan Beasiswa Siswa Miskin</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<span class="sb-sec">📂 Unggah Dataset</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Pilih file CSV atau Excel", type=["csv","xlsx"], label_visibility="collapsed", key="uploader_main")

    if uploaded:
        if st.button("⬆ Upload Dataset", key="btn_upload"):
            raw, msg, info = load_data(uploaded)
            if raw is not None:
                st.session_state["raw"]        = raw
                st.session_state["df_clean"]   = None
                st.session_state["nb_results"] = None
                st.session_state["notif_upload"] = info
                st.session_state["notif_prep"]   = None
                st.session_state["notif_nb"]     = None
            else:
                st.error(msg)

    st.markdown('<span class="sb-sec">⚙ Preprocessing Data</span>', unsafe_allow_html=True)

    if st.session_state["raw"] is not None:
        if st.button("🔧 Jalankan Preprocessing", key="btn_preprocess"):
            df, msg, info = preprocess_data(st.session_state["raw"])
            if df is not None:
                st.session_state["df_clean"]   = df
                st.session_state["nb_results"] = None
                st.session_state["notif_prep"] = info
                st.session_state["notif_nb"]   = None
            else:
                st.error(msg)

    # Pipeline status (indentation fixed)
    st.markdown('<span class="sb-sec">📋 Status Pipeline</span>', unsafe_allow_html=True)
    raw_ok   = st.session_state["raw"]        is not None
    clean_ok = st.session_state["df_clean"]   is not None
    nb_ok    = st.session_state["nb_results"] is not None

    steps = [
        ("1", "Upload Data",   raw_ok,   raw_ok and not clean_ok),
        ("2", "Preprocessing", clean_ok, clean_ok and not nb_ok),
        ("3", "Naive Bayes",   nb_ok,    False),
        ("4", "Evaluasi",      nb_ok,    False),
    ]

    pipe_html = '<div class="pipeline">'
    for num, label_s, done, active in steps:
        if done:
            cls = "done"; num_cls = "done"; symbol = "✓"
        elif active:
            cls = "active"; num_cls = "active"; symbol = num
        else:
            cls = ""; num_cls = ""; symbol = num
        pipe_html += (
            f'<div class="pipeline-step {cls}">'
            f'<span class="step-num {num_cls}">{symbol}</span>'
            f'{label_s}'
            f'</div>'
        )
    pipe_html += '</div>'
    st.markdown(pipe_html, unsafe_allow_html=True)

    # Mini stats sidebar (hanya jika nb_results ada)
    if nb_ok:
        nb_side = st.session_state["nb_results"]
        st.markdown('<span class="sb-sec">📊 Ringkasan Model</span>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:rgba(0,184,150,0.12);border:1px solid rgba(0,217,176,0.30);
                        border-radius:12px;padding:1rem 1.1rem;margin-top:0.2rem;">
                <div style="font-size:0.65rem;color:rgba(255,255,255,0.40);text-transform:uppercase;letter-spacing:1.8px;font-weight:700;margin-bottom:0.7rem;">Performa</div>
                <div style="font-family:'Fraunces',serif;font-size:2.6rem;color:#00FFD0;line-height:1;text-align:center;margin-bottom:0.3rem;">{nb_side['accuracy']*100:.1f}%</div>
                <div style="text-align:center;font-size:0.68rem;color:rgba(255,255,255,0.38);margin-bottom:0.8rem;">Akurasi Model</div>
                <div style="text-align:center;font-size:0.63rem;color:rgba(255,255,255,0.30);margin-bottom:0.8rem;">Split: {nb_side['train_pct']}% Train / {nb_side['test_pct']}% Test</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
                    <div style="background:rgba(0,184,150,0.15);border-radius:8px;padding:0.5rem;text-align:center;">
                        <div style="font-size:1.1rem;font-weight:800;color:#00FFD0;">{nb_side['train_size']}</div>
                        <div style="font-size:0.58rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:1px;">Train</div>
                    </div>
                    <div style="background:rgba(255,45,110,0.12);border-radius:8px;padding:0.5rem;text-align:center;">
                        <div style="font-size:1.1rem;font-weight:800;color:#FF5C8D;">{nb_side['test_size']}</div>
                        <div style="font-size:0.58rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:1px;">Test</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# METRIC CARDS
# =========================================================

raw      = st.session_state["raw"]
df_clean = st.session_state["df_clean"]
nb       = st.session_state["nb_results"]

total_raw   = max(0, len(raw) - 3) if raw is not None else "–"
total_clean = len(df_clean) if df_clean is not None else "–"
akurasi_val = f"{nb['accuracy']*100:.1f}%" if nb else "–"
train_size  = nb["train_size"] if nb else "–"
test_size   = nb["test_size"]  if nb else "–"

# ── GLOBAL NOTIFICATIONS ──────────────────────────────────
notif_upload = st.session_state.get("notif_upload")
notif_prep   = st.session_state.get("notif_prep")
notif_nb     = st.session_state.get("notif_nb")

if notif_upload:
    st.markdown(
        notif_html(
            "upload", "📂",
            "✅ Dataset Berhasil Diunggah!",
            f"File <strong>{notif_upload['filename']}</strong> berhasil dibaca dan siap diproses.",
            [
                f"📋 Total Baris: {notif_upload['rows']}",
                f"📊 Kolom: {notif_upload['cols']}",
                f"🗃 Data Siswa: {notif_upload['data_rows']} baris",
                f"💾 Ukuran: {notif_upload['size_kb']} KB",
            ]
        ),
        unsafe_allow_html=True,
    )

if notif_prep:
    st.markdown(
        notif_html(
            "prep", "🔧",
            "✅ Preprocessing Selesai!",
            f"Data berhasil dinormalisasi dan di-encode. <strong>{notif_prep['dropped']}</strong> baris dihapus karena nilai tidak valid.",
            [
                f"📥 Input: {notif_prep['before']} baris",
                f"✅ Output: {notif_prep['after']} baris bersih",
                f"🎓 Layak BSM: {notif_prep['layak']} siswa",
                f"📋 Tidak Layak: {notif_prep['tidak']} siswa",
                f"🗑 Dihapus: {notif_prep['dropped']} baris",
            ]
        ),
        unsafe_allow_html=True,
    )

if notif_nb:
    st.markdown(
        notif_html(
            "nb", "🤖",
            "✅ Model Naive Bayes Selesai Dilatih!",
            f"Model Gaussian Naive Bayes telah dilatih dengan split <strong>{notif_nb['train_pct']}%/{notif_nb['test_pct']}%</strong>. Akurasi mencapai <strong>{notif_nb['acc']*100:.2f}%</strong>.",
            [
                f"🏋 Data Training: {notif_nb['train']} data ({notif_nb['train_pct']}%)",
                f"🧪 Data Testing: {notif_nb['test']} data ({notif_nb['test_pct']}%)",
                f"✅ Prediksi Benar: {notif_nb['benar']}",
                f"❌ Prediksi Salah: {notif_nb['salah']}",
                f"🏅 Akurasi: {notif_nb['acc']*100:.2f}%",
            ]
        ),
        unsafe_allow_html=True,
    )

# Tampilkan split ratio aktif jika nb sudah dijalankan
if nb:
    split_badge_train = f"{nb['train_pct']}%"
    split_badge_test  = f"{nb['test_pct']}%"
else:
    split_ratio_now = st.session_state.get("split_ratio", 80)
    split_badge_train = f"{split_ratio_now}%"
    split_badge_test  = f"{100 - split_ratio_now}%"

st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-card mc-navy">
            <span class="mc-icon">📋</span>
            <div class="mc-label">Total Data Awal</div>
            <div class="mc-value cv-navy">{total_raw}</div>
        </div>
        <div class="metric-card mc-teal">
            <span class="mc-icon">✅</span>
            <div class="mc-label">Setelah Preprocessing</div>
            <div class="mc-value cv-teal">{total_clean}</div>
        </div>
        <div class="metric-card mc-violet">
            <span class="mc-icon">🏋</span>
            <div class="mc-label">Data Training ({split_badge_train})</div>
            <div class="mc-value cv-violet">{train_size}</div>
        </div>
        <div class="metric-card mc-rose">
            <span class="mc-icon">🧪</span>
            <div class="mc-label">Data Testing ({split_badge_test})</div>
            <div class="mc-value cv-rose">{test_size}</div>
        </div>
        <div class="metric-card mc-amber">
            <span class="mc-icon">🏅</span>
            <div class="mc-label">Akurasi Model</div>
            <div class="mc-value cv-amber">{akurasi_val}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📁 Data Awal",
    "🔧 Normalisasi",
    "📊 Prior & Manual",
    "🎯 Hasil Prediksi",
    "🔲 Confusion Matrix",
    "📈 Laporan Klasifikasi",
    "➕ Input Data Baru",
])

# ─────────────────────────────────────────────────────────
# TAB 1 — Dataset Awal
# ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📁 Dataset Mentah</div>', unsafe_allow_html=True)
    if raw is not None:
        data_rows = max(0, len(raw) - 3)
        st.markdown(
            notif_html(
                "upload", "💡",
                f"Dataset dimuat: {data_rows} baris data siswa",
                "Header tabel dimulai dari baris ke-3 (index 2) pada file sumber. Data di bawah adalah tampilan mentah sebelum preprocessing.",
                [f"📋 Total baris file: {len(raw)}", f"🗃 Estimasi data: {data_rows}", f"📊 Kolom: {len(raw.columns)}"]
            ),
            unsafe_allow_html=True,
        )
        st.dataframe(raw, use_container_width=True, height=420)

        st.markdown("---")
        col_judul1, col_dl1 = st.columns([2, 1])
        with col_judul1:
            judul_raw = st.text_input("Judul untuk file unduhan", value="Dataset Awal BSM", key="judul_raw")
        with col_dl1:
            st.markdown("<br>", unsafe_allow_html=True)
            # Nama file dinamis sesuai judul yang diinput
            filename_raw = f"{judul_raw.replace(' ', '_')}.xlsx"
            st.markdown(make_excel_download(raw, filename_raw, f"⬇ Unduh {judul_raw} (.xlsx)", judul_raw), unsafe_allow_html=True)
    else:
        st.info("👈 Unggah dataset terlebih dahulu melalui panel kiri untuk memulai.")

# ─────────────────────────────────────────────────────────
# TAB 2 — Normalisasi + KONFIGURASI SPLIT + TOMBOL PROSES NB
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🔧 Hasil Normalisasi & Encoding</div>', unsafe_allow_html=True)
    if df_clean is not None:
        if notif_prep:
            st.markdown(
                notif_html(
                    "prep", "🔧",
                    "Hasil Preprocessing Dataset",
                    f"Berhasil memproses <strong>{notif_prep['after']}</strong> data bersih dari <strong>{notif_prep['before']}</strong> data awal.",
                    [
                        f"✅ Data bersih: {notif_prep['after']}",
                        f"🗑 Dihapus (nilai kosong/invalid): {notif_prep['dropped']}",
                        f"🎓 Label Layak: {notif_prep['layak']}",
                        f"📋 Label Tidak Layak: {notif_prep['tidak']}",
                    ]
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            """<div class="info-box">
            💡 Setiap atribut diubah ke bentuk numerik agar dapat diproses model.<br>
            <strong>Pendapatan</strong> → 0 (Rendah: Rp500rb–1jt) / 1 (Menengah: Rp3jt–4jt) / 2 (Tinggi: Rp5jt–7jt) &nbsp;|&nbsp;
            <strong>Pekerjaan</strong> → Petani=0, PNS=1, Polisi=2, Wiraswasta=3 &nbsp;|&nbsp;
            <strong>Rumah</strong> → Kontrak=0, Milik Sendiri=1 &nbsp;|&nbsp;
            <strong>Label</strong> → Tidak=0, Ya=1
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="formula-box">
                <div class="formula-title">📐 Catatan Rumus Encoding & Normalisasi</div>
                <strong>1. Pendapatan Orang Tua</strong> — Kategorisasi rentang nilai rupiah:<br>
                &nbsp;&nbsp;&nbsp;• Rp500.000 – Rp1.000.000 → <code>0 (Rendah)</code><br>
                &nbsp;&nbsp;&nbsp;• Rp3.000.000 – Rp4.000.000 → <code>1 (Menengah)</code><br>
                &nbsp;&nbsp;&nbsp;• Rp5.000.000 – Rp7.000.000 → <code>2 (Tinggi)</code><br>
                <strong>2. Pekerjaan Orang Tua</strong> — Label Encoding: Petani=<code>0</code>, PNS=<code>1</code>, Polisi=<code>2</code>, Wiraswasta=<code>3</code><br>
                <strong>3. Status Rumah</strong> — Binary Encoding: Kontrak/Sewa=<code>0</code>, Milik Sendiri=<code>1</code><br>
                <strong>4. Label Target</strong> — Binary Encoding: Tidak Layak=<code>0</code>, Layak=<code>1</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Tabel penuh lebar ────────────────────────────────
        st.dataframe(df_clean, use_container_width=True, height=320)

        st.markdown("---")

        # ════════════════════════════════════════════════════
        # BLOK KONFIGURASI SPLIT RATIO 
        # ════════════════════════════════════════════════════
        st.markdown('<div class="section-header">⚖️ Konfigurasi Pembagian Data (Train/Test Split)</div>', unsafe_allow_html=True)

        total_data = len(df_clean)

        # Preset pilihan
        PRESET_OPTIONS = {
            "90% / 10%": 90,
            "80% / 20%": 80,
            "70% / 30%": 70,
            "60% / 40%": 60,
            "50% / 50%": 50,
            "100% / 0%": 100,
        }

        current_ratio = st.session_state.get("split_ratio", 80)

        col_preset, col_slider = st.columns([1.2, 1.8])

        with col_preset:
            st.markdown(
                """
                <p style="font-size:.72rem;color:#6B80A8;text-transform:uppercase;
                letter-spacing:1.2px;font-weight:700;margin-bottom:.6rem;">
                🎛 Pilih Preset Rasio
                </p>
                """,
                unsafe_allow_html=True,
            )
            preset_labels = list(PRESET_OPTIONS.keys())
            default_preset_idx = 1  # default 80/20
            for idx, (label, val) in enumerate(PRESET_OPTIONS.items()):
                if val == current_ratio:
                    default_preset_idx = idx
                    break

            selected_preset = st.radio(
                "Pilih preset rasio",
                options=preset_labels,
                index=default_preset_idx,
                key="preset_split",
                label_visibility="collapsed",
            )
            preset_val = PRESET_OPTIONS[selected_preset]
            st.session_state["split_ratio"] = preset_val

        with col_slider:
            st.markdown(
                """
                <p style="font-size:.72rem;color:#6B80A8;text-transform:uppercase;
                letter-spacing:1.2px;font-weight:700;margin-bottom:.6rem;">
                🔢 Atau Atur Manual dengan Slider (% Data Training)
                </p>
                """,
                unsafe_allow_html=True,
            )
            slider_val = st.slider(
                "Persentase data training",
                min_value=50,
                max_value=100,
                value=st.session_state["split_ratio"],
                step=5,
                format="%d%%",
                key="slider_split",
                label_visibility="collapsed",
            )
            if slider_val != st.session_state["split_ratio"]:
                st.session_state["split_ratio"] = slider_val

        final_ratio  = st.session_state["split_ratio"]
        train_count  = round(total_data * final_ratio / 100)
        test_count   = total_data - train_count
        bar_width    = final_ratio

        # Visual bar
        st.markdown(
            f"""
            <div class="split-config-box">
                <div class="split-config-title">
                    ⚖️ Rasio Pembagian Data yang Dipilih
                </div>
                <div class="split-config-desc">
                    Konfigurasi ini akan digunakan saat tombol <strong>🤖 Proses & Klasifikasi Naive Bayes</strong> di bawah ditekan.
                </div>
                <div class="split-bar-wrap">
                    <div class="split-bar-train"
                         style="width:{bar_width}%;background:linear-gradient(90deg,#00B896,#1648C8);"></div>
                </div>
                <div class="split-bar-labels">
                    <span class="split-label-train">🏋 Training — {final_ratio}% ({train_count} data)</span>
                    <span class="split-label-test">🧪 Testing — {100-final_ratio}% ({test_count} data)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Info tambahan jika 100%
        if final_ratio == 100:
            st.markdown(
                notif_html(
                    "warn", "⚠️",
                    "Mode 100% Training — Perhatian!",
                    "Seluruh data digunakan untuk training. Evaluasi dilakukan pada data training itu sendiri (in-sample), sehingga akurasi cenderung lebih tinggi namun tidak mencerminkan performa pada data baru.",
                    ["📌 In-sample evaluation", "⚠️ Risiko overfitting", "💡 Gunakan untuk eksplorasi"]
                ),
                unsafe_allow_html=True,
            )

        # ─────────────────────────────────────────────────────
        # TOMBOL PROSES NAIVE BAYES (LANJUT KE KLASIFIKASI)
        # ─────────────────────────────────────────────────────
        st.markdown("---")
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("🤖 Proses & Klasifikasi Naive Bayes", use_container_width=True):
                with st.spinner("Melatih model dan melakukan klasifikasi..."):
                    nb_res = naive_bayes_process(st.session_state["df_clean"], train_pct=final_ratio)
                    st.session_state["nb_results"] = nb_res
                    st.session_state["notif_nb"] = {
                        "train": nb_res["train_size"],
                        "test":  nb_res["test_size"],
                        "acc":   nb_res["accuracy"],
                        "benar": nb_res["benar"],
                        "salah": nb_res["salah"],
                        "train_pct": nb_res["train_pct"],
                        "test_pct":  nb_res["test_pct"],
                    }
                    st.success(f"✅ Klasifikasi selesai! Akurasi: {nb_res['accuracy']*100:.2f}%")
        with col_btn2:
            if st.button("🗑 Reset Model & Data", use_container_width=True):
                st.session_state["nb_results"] = None
                st.session_state["notif_nb"] = None
                st.session_state["df_clean"] = None
                st.session_state["raw"] = None
                st.session_state["notif_upload"] = None
                st.session_state["notif_prep"] = None
                st.session_state["data_tambahan"] = pd.DataFrame(
                    columns=["NAMA SISWA","KELAS","NIS","PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL (PREDIKSI)"]
                )
                st.rerun()

        # ── Distribusi Label (ringkasan teks) ───────────────
        st.markdown("---")
        label_counts = df_clean["LABEL"].value_counts().rename({0: "Tidak Layak", 1: "Layak"})
        layak_n  = int(df_clean["LABEL"].sum())
        tidak_n  = len(df_clean) - layak_n

        st.markdown(
            f"""
            <div style="display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap;">
                <div style="flex:1;min-width:160px;background:linear-gradient(135deg,#E8FFF9,#D0FFE8);
                            border:1.5px solid rgba(0,184,150,0.30);border-radius:14px;
                            padding:1.1rem 1.3rem;text-align:center;">
                    <div style="font-size:.62rem;color:#00896F;text-transform:uppercase;
                                letter-spacing:1.5px;font-weight:800;margin-bottom:.4rem;">🎓 Layak BSM</div>
                    <div style="font-family:'Fraunces',serif;font-size:2.4rem;
                                color:#00B896;line-height:1;font-weight:700;">{layak_n:,}</div>
                    <div style="font-size:.75rem;color:#5A7A70;margin-top:.3rem;font-weight:600;">
                        {layak_n/len(df_clean)*100:.1f}% dari total data</div>
                </div>
                <div style="flex:1;min-width:160px;background:linear-gradient(135deg,#FFF0F5,#FFE0EB);
                            border:1.5px solid rgba(255,45,110,0.25);border-radius:14px;
                            padding:1.1rem 1.3rem;text-align:center;">
                    <div style="font-size:.62rem;color:#B0003A;text-transform:uppercase;
                                letter-spacing:1.5px;font-weight:800;margin-bottom:.4rem;">📋 Tidak Layak</div>
                    <div style="font-family:'Fraunces',serif;font-size:2.4rem;
                                color:#FF2D6E;line-height:1;font-weight:700;">{tidak_n:,}</div>
                    <div style="font-size:.75rem;color:#8A4A5A;margin-top:.3rem;font-weight:600;">
                        {tidak_n/len(df_clean)*100:.1f}% dari total data</div>
                </div>
                <div style="flex:1;min-width:160px;background:linear-gradient(135deg,#EEF4FF,#E8F0FF);
                            border:1.5px solid rgba(22,72,200,0.20);border-radius:14px;
                            padding:1.1rem 1.3rem;text-align:center;">
                    <div style="font-size:.62rem;color:#1648C8;text-transform:uppercase;
                                letter-spacing:1.5px;font-weight:800;margin-bottom:.4rem;">👥 Total Siswa</div>
                    <div style="font-family:'Fraunces',serif;font-size:2.4rem;
                                color:#1648C8;line-height:1;font-weight:700;">{len(df_clean):,}</div>
                    <div style="font-size:.75rem;color:#3D5070;margin-top:.3rem;font-weight:600;">
                        Data bersih siap diproses</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_dl_wrap, _ = st.columns([1, 2])
        with col_dl_wrap:
            judul_norm = st.text_input("Judul file", value="Hasil Normalisasi BSM", key="judul_norm")
            filename_norm = f"{judul_norm.replace(' ', '_')}.xlsx"
            st.markdown(
                make_excel_download(df_clean, filename_norm, f"Unduh {judul_norm} (.xlsx)", judul_norm),
                unsafe_allow_html=True,
            )
    else:
        st.info("👈 Klik **Jalankan Preprocessing** di panel kiri setelah mengunggah data.")

# ─────────────────────────────────────────────────────────
# TAB 3 — Prior & Manual
# ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📊 Prior Probability & Perhitungan Manual</div>', unsafe_allow_html=True)
    if nb is not None:
        if notif_nb:
            st.markdown(
                notif_html(
                    "nb", "📊",
                    "Model Telah Dilatih — Data Prior & Likelihood Tersedia",
                    f"Dihitung dari <strong>{notif_nb['train']}</strong> data training ({notif_nb['train_pct']}%). Prior probability mencerminkan distribusi kelas pada data.",
                    [f"🏋 Training: {notif_nb['train']} ({notif_nb['train_pct']}%)", f"📊 Kelas: {len(nb['classes'])}"]
                ),
                unsafe_allow_html=True,
            )

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<p style="font-size:.68rem;color:#6B80A8;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.45rem;">Tabel Prior Probability</p>', unsafe_allow_html=True)
            st.dataframe(nb["prior"], use_container_width=True)

            fig_prior, ax_pr = plt.subplots(figsize=(3.5, 2.8))
            fig_prior.patch.set_facecolor("#FFFFFF")
            ax_pr.set_facecolor("#F5F7FF")
            bar_col = ["#FF2D6E", "#00B896"]
            bars = ax_pr.bar(
                ["Tidak Layak (0)", "Layak (1)"],
                nb["prior"]["Prior Probability"],
                color=bar_col, width=0.5, edgecolor="#ffffff", linewidth=1.5,
            )
            ax_pr.set_ylim(0, 1)
            ax_pr.set_ylabel("Probabilitas", color="#6B80A8", fontsize=9)
            ax_pr.tick_params(colors="#6B80A8", labelsize=8)
            for spine in ax_pr.spines.values(): spine.set_edgecolor("#C5D0F0")
            for bar in bars:
                ax_pr.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.015, f"{bar.get_height():.3f}", ha="center", va="bottom", color="#080F1E", fontsize=9, fontweight="bold")
            ax_pr.axhline(0.5, color="#FF8C00", linewidth=1, linestyle="--", alpha=.6)
            ax_pr.set_title("Prior per Kelas", fontsize=10, color="#0F1E42", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_prior)

        with c2:
            st.markdown('<p style="font-size:.68rem;color:#6B80A8;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.45rem;">Perhitungan Manual Naive Bayes (per data uji)</p>', unsafe_allow_html=True)
            st.dataframe(nb["manual"], use_container_width=True, height=380)

        st.markdown(
            """
            <div class="formula-box">
                <div class="formula-title">📐 Rumus Perhitungan Naive Bayes</div>
                <strong>1. Prior Probability:</strong><br>
                &nbsp;&nbsp;&nbsp;<code>P(Kelas) = Jumlah data kelas / Total data training</code><br><br>
                <strong>2. Gaussian Probability (Likelihood):</strong><br>
                &nbsp;&nbsp;&nbsp;<code>P(x|Kelas) = (1 / √(2π × σ²)) × exp(-(x-μ)² / (2σ²))</code><br><br>
                <strong>3. Posterior Probability:</strong><br>
                &nbsp;&nbsp;&nbsp;<code>P(Kelas|X) = P(Kelas) × P(x₁|Kelas) × P(x₂|Kelas) × ... × P(xₙ|Kelas)</code><br><br>
                <strong>4. Keputusan:</strong><br>
                &nbsp;&nbsp;&nbsp;<code>Prediksi = argmax P(Kelas|X)</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        col_judul3, col_dl3 = st.columns([2, 1])
        with col_judul3:
            judul_manual = st.text_input("Judul untuk file unduhan", value="Perhitungan Manual Naive Bayes", key="judul_manual")
        with col_dl3:
            st.markdown("<br>", unsafe_allow_html=True)
            filename_manual = f"{judul_manual.replace(' ', '_')}.xlsx"
            st.markdown(make_excel_download(nb["manual"], filename_manual, f"⬇ Unduh {judul_manual} (.xlsx)", judul_manual), unsafe_allow_html=True)
    else:
        st.info("👈 Setelah preprocessing, atur rasio split di Tab 'Normalisasi' dan klik tombol **Proses & Klasifikasi Naive Bayes**.")

# ─────────────────────────────────────────────────────────
# TAB 4 — Hasil Prediksi
# ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🎯 Hasil Prediksi vs Data Aktual</div>', unsafe_allow_html=True)
    if nb is not None:
        prediksi_df = nb["manual"][["Data Ke","Prediksi","Aktual","Benar?"]]
        benar = nb["benar"]
        salah = nb["salah"]

        st.markdown(
            notif_html(
                "nb", "🎯",
                f"Prediksi Selesai — {benar} dari {benar+salah} data uji diprediksi dengan benar",
                f"Split: <strong>{nb['train_pct']}% Training / {nb['test_pct']}% Testing</strong> | "
                f"Akurasi pada data uji: <strong>{(benar/(benar+salah)*100) if (benar+salah)>0 else 0:.2f}%</strong>.",
                [
                    f"✅ Benar: {benar}",
                    f"❌ Salah: {salah}",
                    f"📊 Total uji: {benar+salah}",
                    f"🏅 Akurasi: {(benar/(benar+salah)*100) if (benar+salah)>0 else 0:.2f}%",
                ]
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="metric-row">
            <div class="metric-card mc-teal"><span class="mc-icon">✅</span><div class="mc-label">Prediksi Benar</div><div class="mc-value cv-teal">{benar}</div></div>
            <div class="metric-card mc-rose"><span class="mc-icon">❌</span><div class="mc-label">Prediksi Salah</div><div class="mc-value cv-rose">{salah}</div></div>
            <div class="metric-card mc-violet"><span class="mc-icon">🧑‍🎓</span><div class="mc-label">Total Data Uji</div><div class="mc-value cv-violet">{benar+salah}</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.dataframe(prediksi_df, use_container_width=True, height=400)

        st.markdown(
            f"""
            <div class="formula-box">
                <div class="formula-title">📐 Rumus Akurasi</div>
                <code>Akurasi = (Jumlah Prediksi Benar / Total Data Uji) × 100%</code><br>
                &nbsp;&nbsp;&nbsp;= ({benar} / {benar+salah}) × 100% = <strong>{(benar/(benar+salah)*100) if (benar+salah)>0 else 0:.2f}%</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        col_judul4, col_dl4 = st.columns([2, 1])
        with col_judul4:
            judul_pred = st.text_input("Judul untuk file unduhan", value="Hasil Prediksi Naive Bayes", key="judul_pred")
        with col_dl4:
            st.markdown("<br>", unsafe_allow_html=True)
            filename_pred = f"{judul_pred.replace(' ', '_')}.xlsx"
            st.markdown(make_excel_download(prediksi_df, filename_pred, f"⬇ Unduh {judul_pred} (.xlsx)", judul_pred), unsafe_allow_html=True)
    else:
        st.info("👈 Setelah preprocessing, atur rasio split di Tab 'Normalisasi' dan klik tombol **Proses & Klasifikasi Naive Bayes**.")

# ─────────────────────────────────────────────────────────
# TAB 5 — Confusion Matrix
# ─────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
    if nb is not None:
        cm_raw = nb["cm"]
        tn = cm_raw[0, 0] if cm_raw.shape[0] > 1 else 0
        fp = cm_raw[0, 1] if cm_raw.shape[1] > 1 else 0
        fn = cm_raw[1, 0] if cm_raw.shape[0] > 1 else 0
        tp = cm_raw[1, 1] if cm_raw.shape[1] > 1 else 0

        st.markdown(
            notif_html(
                "nb", "🔲",
                f"Confusion Matrix — Split {nb['train_pct']}%/{nb['test_pct']}%",
                f"True Positive: <strong>{tp}</strong> | True Negative: <strong>{tn}</strong> | False Positive: <strong>{fp}</strong> | False Negative: <strong>{fn}</strong>",
                [
                    f"✅ TN: {tn}",
                    f"✅ TP: {tp}",
                    f"⚠️ FP: {fp}",
                    f"⚠️ FN: {fn}",
                    f"🏅 Akurasi: {nb['accuracy']*100:.2f}%",
                ]
            ),
            unsafe_allow_html=True,
        )

        col_img, col_exp = st.columns([1, 1])
        with col_img:
            st.pyplot(nb["fig_cm"])
            col_judul5, col_dl5 = st.columns([2, 1])
            with col_judul5:
                judul_cm = st.text_input("Judul file", value="Confusion Matrix BSM", key="judul_cm")
            with col_dl5:
                st.markdown("<br>", unsafe_allow_html=True)
                filename_cm_png = f"{judul_cm.replace(' ', '_')}.png"
                st.markdown(make_png_download(nb["fig_cm"], filename_cm_png, f"⬇ Unduh {judul_cm} (.png)"), unsafe_allow_html=True)

        with col_exp:
            st.markdown(
                """
                <div class="info-box">
                <strong>Cara Membaca Confusion Matrix</strong><br><br>
                <strong style="color:#00B896;">True Negative (TN)</strong> — Prediksi <em>Tidak Layak</em>, Aktual <em>Tidak Layak</em> ✅<br><br>
                <strong style="color:#FF2D6E;">False Positive (FP)</strong> — Prediksi <em>Layak</em>, Aktual <em>Tidak Layak</em> ❌<br><br>
                <strong style="color:#FF2D6E;">False Negative (FN)</strong> — Prediksi <em>Tidak Layak</em>, Aktual <em>Layak</em> ❌<br><br>
                <strong style="color:#00B896;">True Positive (TP)</strong> — Prediksi <em>Layak</em>, Aktual <em>Layak</em> ✅
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-bottom:.9rem">
                    <div style="background:#fff;border:1.5px solid #C5D0F0;border-top:3px solid #00B896;border-radius:12px;padding:.9rem 1rem;text-align:center;box-shadow:0 2px 8px rgba(0,184,150,0.10)">
                        <div style="font-size:.6rem;color:#8FA3BF;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.3rem;font-weight:700;">True Negative</div>
                        <div style="font-family:'Fraunces',serif;font-size:2.4rem;color:#00B896;line-height:1">{tn}</div>
                    </div>
                    <div style="background:#fff;border:1.5px solid #C5D0F0;border-top:3px solid #FF2D6E;border-radius:12px;padding:.9rem 1rem;text-align:center;box-shadow:0 2px 8px rgba(255,45,110,0.10)">
                        <div style="font-size:.6rem;color:#8FA3BF;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.3rem;font-weight:700;">False Positive</div>
                        <div style="font-family:'Fraunces',serif;font-size:2.4rem;color:#FF2D6E;line-height:1">{fp}</div>
                    </div>
                    <div style="background:#fff;border:1.5px solid #C5D0F0;border-top:3px solid #FF2D6E;border-radius:12px;padding:.9rem 1rem;text-align:center;box-shadow:0 2px 8px rgba(255,45,110,0.10)">
                        <div style="font-size:.6rem;color:#8FA3BF;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.3rem;font-weight:700;">False Negative</div>
                        <div style="font-family:'Fraunces',serif;font-size:2.4rem;color:#FF2D6E;line-height:1">{fn}</div>
                    </div>
                    <div style="background:#fff;border:1.5px solid #C5D0F0;border-top:3px solid #00B896;border-radius:12px;padding:.9rem 1rem;text-align:center;box-shadow:0 2px 8px rgba(0,184,150,0.10)">
                        <div style="font-size:.6rem;color:#8FA3BF;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.3rem;font-weight:700;">True Positive</div>
                        <div style="font-family:'Fraunces',serif;font-size:2.4rem;color:#00B896;line-height:1">{tp}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            acc = nb["accuracy"]
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#080F1E,#0F1E42);
                            border-radius:16px;padding:1.4rem 1.6rem;text-align:center;
                            border-bottom:4px solid #00B896;box-shadow:0 10px 36px rgba(8,15,30,0.25);">
                    <div style="font-size:.62rem;color:rgba(255,255,255,0.35);text-transform:uppercase;
                                letter-spacing:2px;margin-bottom:.4rem;font-weight:700;">Akurasi Model</div>
                    <div style="font-family:'Fraunces',serif;font-size:3.8rem;
                                color:#00FFD0;line-height:1">{acc*100:.2f}%</div>
                    <div style="font-size:.78rem;color:rgba(255,255,255,0.35);margin-top:.5rem;">
                        {tn+tp} prediksi benar dari {tn+tp+fn+fp} data uji
                    </div>
                    <div style="font-size:.68rem;color:rgba(0,217,176,0.50);margin-top:.3rem;">
                        Split: {nb['train_pct']}% Train / {nb['test_pct']}% Test
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="formula-box" style="margin-top:.9rem;">
                    <div class="formula-title">📐 Rumus Metrik dari Confusion Matrix</div>
                    <code>Akurasi  = (TP + TN) / (TP + TN + FP + FN)</code><br>
                    <code>Presisi  = TP / (TP + FP)</code><br>
                    <code>Recall   = TP / (TP + FN)</code><br>
                    <code>F1-Score = 2 × (Presisi × Recall) / (Presisi + Recall)</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("👈 Setelah preprocessing, atur rasio split di Tab 'Normalisasi' dan klik tombol **Proses & Klasifikasi Naive Bayes**.")

# ─────────────────────────────────────────────────────────
# TAB 6 — Classification Report
# ─────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">📈 Laporan Klasifikasi Lengkap</div>', unsafe_allow_html=True)
    if nb is not None:
        report         = nb["report"].copy()
        report_display = report.round(4)

        if notif_nb:
            try:
                prec0 = float(report.loc["0", "precision"])
                rec0  = float(report.loc["0", "recall"])
                prec1 = float(report.loc["1", "precision"])
                rec1  = float(report.loc["1", "recall"])
                st.markdown(
                    notif_html(
                        "prep", "📈",
                        f"Laporan Klasifikasi Per Kelas — Split {nb['train_pct']}%/{nb['test_pct']}%",
                        f"Kelas Layak (1): Precision={prec1:.2f}, Recall={rec1:.2f} | Kelas Tidak Layak (0): Precision={prec0:.2f}, Recall={rec0:.2f}",
                        [
                            f"Layak — P: {prec1:.2f}, R: {rec1:.2f}",
                            f"Tidak Layak — P: {prec0:.2f}, R: {rec0:.2f}",
                            f"🏅 Akurasi: {notif_nb['acc']*100:.2f}%",
                        ]
                    ),
                    unsafe_allow_html=True,
                )
            except:
                pass

        st.dataframe(report_display, use_container_width=True)

        metrics_df = report.loc[["0","1"], ["precision","recall","f1-score"]]
        fig_bar, ax_b = plt.subplots(figsize=(6, 3.5))
        fig_bar.patch.set_facecolor("#FFFFFF")
        ax_b.set_facecolor("#F5F7FF")

        x = np.arange(len(metrics_df))
        w = 0.25
        bar_palette = ["#1648C8", "#00B896", "#FF8C00"]
        for idx, (col, color) in enumerate(zip(metrics_df.columns, bar_palette)):
            ax_b.bar(x + idx*w, metrics_df[col], w, label=col.capitalize(), color=color, edgecolor="#ffffff", linewidth=1.5)

        ax_b.set_xticks(x + w)
        ax_b.set_xticklabels(["Kelas 0 — Tidak Layak", "Kelas 1 — Layak"], color="#080F1E", fontsize=10)
        ax_b.set_ylim(0, 1.2)
        ax_b.tick_params(colors="#6B80A8", labelsize=9)
        ax_b.legend(facecolor="#FFFFFF", edgecolor="#C5D0F0", labelcolor="#080F1E", fontsize=9)
        for spine in ax_b.spines.values(): spine.set_edgecolor("#C5D0F0")
        ax_b.axhline(1.0, color="#C5D0F0", linewidth=0.8, linestyle="--")
        ax_b.set_title(f"Perbandingan Metrik Evaluasi per Kelas (Split {nb['train_pct']}%/{nb['test_pct']}%)", color="#0F1E42", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_bar)

        try:
            prec0 = float(report.loc["0", "precision"])
            rec0  = float(report.loc["0", "recall"])
            f1_0  = float(report.loc["0", "f1-score"])
            prec1 = float(report.loc["1", "precision"])
            rec1  = float(report.loc["1", "recall"])
            f1_1  = float(report.loc["1", "f1-score"])

            def interpret(val):
                if val >= 0.90: return ("Sangat Baik", "#00B896")
                elif val >= 0.75: return ("Baik", "#1648C8")
                elif val >= 0.60: return ("Cukup", "#FF8C00")
                else: return ("Perlu Ditingkatkan", "#FF2D6E")

            ip0, cp0 = interpret(prec0)
            ir0, cr0 = interpret(rec0)
            if0_, cf0 = interpret(f1_0)
            ip1, cp1 = interpret(prec1)
            ir1, cr1 = interpret(rec1)
            if1_, cf1 = interpret(f1_1)

            p0_pct  = f"{prec0*100:.1f}%"
            r0_pct  = f"{rec0*100:.1f}%"
            p1_pct  = f"{prec1*100:.1f}%"
            r1_pct  = f"{rec1*100:.1f}%"
            p0_val  = f"{prec0:.2f}"
            r0_val  = f"{rec0:.2f}"
            f0_val  = f"{f1_0:.2f}"
            p1_val  = f"{prec1:.2f}"
            r1_val  = f"{rec1:.2f}"
            f1_val  = f"{f1_1:.2f}"

            html_interp = (
                '<div style="margin-top:1.2rem;">'
                '<p style="font-size:.68rem;color:#6B80A8;text-transform:uppercase;'
                'letter-spacing:1.2px;font-weight:700;margin-bottom:.8rem;">'
                '💬 Interpretasi Diagram — Apa Artinya?</p>'
                '<div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:.8rem;">'

                # Kelas 0
                '<div style="flex:1;min-width:260px;background:#fff;border:1.5px solid #C5D0F0;'
                'border-top:4px solid #FF2D6E;border-radius:14px;padding:1.1rem 1.3rem;'
                'box-shadow:0 2px 10px rgba(255,45,110,0.08);">'
                '<div style="font-size:.7rem;font-weight:800;color:#FF2D6E;text-transform:uppercase;'
                'letter-spacing:1.2px;margin-bottom:.7rem;">📋 Kelas 0 — Tidak Layak BSM</div>'
                '<div style="display:flex;flex-direction:column;gap:.45rem;">'

                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'background:#F5F7FF;border-radius:8px;padding:.45rem .8rem;">'
                '<span style="font-size:.8rem;color:#3D5070;font-weight:600;">'
                '🎯 Precision <span style="font-size:.68rem;color:#9AAEE0;">(Ketepatan prediksi Tidak Layak)</span></span>'
                '<span style="font-size:.88rem;font-weight:800;color:' + cp0 + ';">' + p0_val + ' — ' + ip0 + '</span></div>'

                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'background:#F5F7FF;border-radius:8px;padding:.45rem .8rem;">'
                '<span style="font-size:.8rem;color:#3D5070;font-weight:600;">'
                '🔍 Recall <span style="font-size:.68rem;color:#9AAEE0;">(Cakupan siswa Tidak Layak yang terdeteksi)</span></span>'
                '<span style="font-size:.88rem;font-weight:800;color:' + cr0 + ';">' + r0_val + ' — ' + ir0 + '</span></div>'

                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'background:#F5F7FF;border-radius:8px;padding:.45rem .8rem;">'
                '<span style="font-size:.8rem;color:#3D5070;font-weight:600;">'
                '⚖️ F1-Score <span style="font-size:.68rem;color:#9AAEE0;">(Keseimbangan Precision dan Recall)</span></span>'
                '<span style="font-size:.88rem;font-weight:800;color:' + cf0 + ';">' + f0_val + ' — ' + if0_ + '</span></div>'
                '</div>'

                '<div style="margin-top:.7rem;font-size:.75rem;color:#6B80A8;line-height:1.6;'
                'background:#FFF0F5;border-radius:8px;padding:.6rem .8rem;border-left:3px solid #FF2D6E;">'
                '💡 Dari semua siswa yang diprediksi <strong>Tidak Layak</strong>, '
                '<strong style="color:' + cp0 + ';">' + p0_pct + '</strong> memang benar-benar tidak layak. '
                'Model berhasil mendeteksi <strong style="color:' + cr0 + ';">' + r0_pct + '</strong> '
                'dari seluruh siswa yang seharusnya tidak layak.</div>'
                '</div>'

                # Kelas 1
                '<div style="flex:1;min-width:260px;background:#fff;border:1.5px solid #C5D0F0;'
                'border-top:4px solid #00B896;border-radius:14px;padding:1.1rem 1.3rem;'
                'box-shadow:0 2px 10px rgba(0,184,150,0.08);">'
                '<div style="font-size:.7rem;font-weight:800;color:#00B896;text-transform:uppercase;'
                'letter-spacing:1.2px;margin-bottom:.7rem;">🎓 Kelas 1 — Layak BSM</div>'
                '<div style="display:flex;flex-direction:column;gap:.45rem;">'

                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'background:#F5F7FF;border-radius:8px;padding:.45rem .8rem;">'
                '<span style="font-size:.8rem;color:#3D5070;font-weight:600;">'
                '🎯 Precision <span style="font-size:.68rem;color:#9AAEE0;">(Ketepatan prediksi Layak)</span></span>'
                '<span style="font-size:.88rem;font-weight:800;color:' + cp1 + ';">' + p1_val + ' — ' + ip1 + '</span></div>'

                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'background:#F5F7FF;border-radius:8px;padding:.45rem .8rem;">'
                '<span style="font-size:.8rem;color:#3D5070;font-weight:600;">'
                '🔍 Recall <span style="font-size:.68rem;color:#9AAEE0;">(Cakupan siswa Layak yang terdeteksi)</span></span>'
                '<span style="font-size:.88rem;font-weight:800;color:' + cr1 + ';">' + r1_val + ' — ' + ir1 + '</span></div>'

                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'background:#F5F7FF;border-radius:8px;padding:.45rem .8rem;">'
                '<span style="font-size:.8rem;color:#3D5070;font-weight:600;">'
                '⚖️ F1-Score <span style="font-size:.68rem;color:#9AAEE0;">(Keseimbangan Precision dan Recall)</span></span>'
                '<span style="font-size:.88rem;font-weight:800;color:' + cf1 + ';">' + f1_val + ' — ' + if1_ + '</span></div>'
                '</div>'

                '<div style="margin-top:.7rem;font-size:.75rem;color:#6B80A8;line-height:1.6;'
                'background:#E8FFF9;border-radius:8px;padding:.6rem .8rem;border-left:3px solid #00B896;">'
                '💡 Dari semua siswa yang diprediksi <strong>Layak BSM</strong>, '
                '<strong style="color:' + cp1 + ';">' + p1_pct + '</strong> memang benar-benar layak. '
                'Model berhasil mendeteksi <strong style="color:' + cr1 + ';">' + r1_pct + '</strong> '
                'dari seluruh siswa yang seharusnya layak menerima beasiswa.</div>'
                '</div>'

                '</div></div>'
            )

            st.markdown(html_interp, unsafe_allow_html=True)

        except:
            pass

        st.markdown(
            """
            <div class="formula-box">
                <div class="formula-title">📐 Keterangan Metrik Evaluasi</div>
                <strong>Precision</strong> — Dari semua yang diprediksi positif, berapa yang benar-benar positif:<br>
                &nbsp;&nbsp;&nbsp;<code>Precision = TP / (TP + FP)</code><br><br>
                <strong>Recall (Sensitivity)</strong> — Dari semua data positif aktual, berapa yang berhasil dideteksi:<br>
                &nbsp;&nbsp;&nbsp;<code>Recall = TP / (TP + FN)</code><br><br>
                <strong>F1-Score</strong> — Harmonic mean antara Precision dan Recall:<br>
                &nbsp;&nbsp;&nbsp;<code>F1 = 2 × (Precision × Recall) / (Precision + Recall)</code><br><br>
                <strong>Support</strong> — Jumlah data aktual per kelas dalam data uji
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        col_dl_a, col_dl_b = st.columns(2)
        with col_dl_a:
            judul_rpt = st.text_input("Judul file laporan", value="Laporan Klasifikasi BSM", key="judul_rpt")
            filename_rpt = f"{judul_rpt.replace(' ', '_')}.xlsx"
            st.markdown(make_excel_download(report_display, filename_rpt, f"⬇ Unduh {judul_rpt} (.xlsx)", judul_rpt), unsafe_allow_html=True)
        with col_dl_b:
            st.markdown("<br>", unsafe_allow_html=True)
            filename_metrics_png = f"{judul_rpt.replace(' ', '_')}_chart.png"
            st.markdown(make_png_download(fig_bar, filename_metrics_png, f"⬇ Unduh Grafik {judul_rpt} (.png)"), unsafe_allow_html=True)
    else:
        st.info("👈 Setelah preprocessing, atur rasio split di Tab 'Normalisasi' dan klik tombol **Proses & Klasifikasi Naive Bayes**.")

# ─────────────────────────────────────────────────────────
# TAB 7 — Tambah Data Baru (tidak berubah)
# ─────────────────────────────────────────────────────────
with tab7:
    st.markdown('<div class="section-header">➕ Input & Prediksi Data Siswa Baru</div>', unsafe_allow_html=True)

    if nb is None:
        st.markdown(
            notif_html(
                "warn", "⚠️",
                "Model Belum Dilatih",
                "Harap jalankan proses Naive Bayes terlebih dahulu agar prediksi dapat dijalankan. Ikuti langkah: Upload Data → Preprocessing → Atur Split → Klik **Proses & Klasifikasi Naive Bayes** di Tab Normalisasi.",
                ["1️⃣ Upload Data", "2️⃣ Preprocessing", "3️⃣ Atur Split di Tab Normalisasi", "4️⃣ Klik tombol Proses NB"]
            ),
            unsafe_allow_html=True,
        )
    else:
        mode = st.radio(
            "Pilih metode input data:",
            ["📝 Input Manual (Satu per Satu)", "📂 Upload File Banyak Data (CSV/Excel)"],
            horizontal=True,
            label_visibility="visible",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── MODE 1 — Input Manual ─────────────────────────
        if mode == "📝 Input Manual (Satu per Satu)":
            st.markdown(
                '<div class="info-box">🧑‍🎓 Isi data atribut siswa di bawah. <strong>Label kelayakan akan diprediksi otomatis</strong> oleh model Gaussian Naive Bayes yang telah dilatih.</div>',
                unsafe_allow_html=True,
            )
            with st.form("form_tambah_data", clear_on_submit=True):
                st.markdown('<p style="font-size:.72rem;color:#6B80A8;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.8rem;">📝 Identitas Siswa</p>', unsafe_allow_html=True)
                col_id1, col_id2, col_id3 = st.columns(3)
                with col_id1: inp_nama  = st.text_input("Nama Siswa", placeholder="Contoh: Ahmad Fauzi")
                with col_id2: inp_kelas = st.text_input("Kelas", placeholder="Contoh: X IPA 1")
                with col_id3: inp_nis   = st.text_input("NIS / No. Induk", placeholder="Contoh: 2024001")

                st.markdown('<p style="font-size:.72rem;color:#6B80A8;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin:.8rem 0;">📋 Data Atribut Ekonomi</p>', unsafe_allow_html=True)
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    inp_pendapatan = st.selectbox("Pendapatan Orang Tua", options=[0,1,2],
                        format_func=lambda x: {0:"Rendah — Rp500.000 hingga Rp1.000.000",1:"Menengah — Rp3.000.000 hingga Rp4.000.000",2:"Tinggi — Rp5.000.000 hingga Rp7.000.000"}[x])
                    inp_pekerjaan = st.selectbox("Pekerjaan Orang Tua", options=[0,1,2,3],
                        format_func=lambda x: {0:"Petani",1:"Pegawai Negeri Sipil (PNS)",2:"Polisi / TNI",3:"Wiraswasta"}[x])
                with col_f2:
                    inp_tanggungan = st.number_input("Jumlah Tanggungan Keluarga", min_value=0, max_value=20, value=1, step=1)
                    inp_rumah      = st.selectbox("Status Kepemilikan Rumah", options=[0,1],
                        format_func=lambda x: "Kontrak / Sewa" if x == 0 else "Milik Sendiri")

                submitted = st.form_submit_button("🔍 Prediksi & Tambahkan ke Riwayat", use_container_width=True)

            if submitted:
                if not inp_nama.strip():
                    st.markdown(notif_html("warn","⚠️","Nama Siswa Kosong","Nama siswa tidak boleh kosong. Isi terlebih dahulu.",None), unsafe_allow_html=True)
                else:
                    fitur_input = pd.DataFrame([{
                        "PENDAPATAN ORANG TUA": int(inp_pendapatan),
                        "PEKERJAAN ORANG TUA":  int(inp_pekerjaan),
                        "JUMLAH TANGGUNGAN":    int(inp_tanggungan),
                        "STATUS RUMAH":         int(inp_rumah),
                    }])
                    label_pred = int(nb["model"].predict(fitur_input)[0])
                    label_text = "Layak Menerima Beasiswa" if label_pred == 1 else "Tidak Layak Menerima Beasiswa"
                    css_class  = "pred-layak" if label_pred == 1 else "pred-tidak"
                    icon       = "🎓" if label_pred == 1 else "📋"
                    color      = "#00B896" if label_pred == 1 else "#FF2D6E"

                    st.markdown(
                        notif_html(
                            "prep" if label_pred == 1 else "upload",
                            icon,
                            f"Hasil Prediksi: {inp_nama.strip()}",
                            f"Berdasarkan data yang dimasukkan, siswa <strong>{inp_nama.strip()}</strong> diprediksi: <strong style='color:{color}'>{label_text}</strong>",
                            [
                                f"Pendapatan: {['Rendah','Menengah','Tinggi'][int(inp_pendapatan)]}",
                                f"Pekerjaan: {['Petani','PNS','Polisi','Wiraswasta'][int(inp_pekerjaan)]}",
                                f"Tanggungan: {inp_tanggungan}",
                                f"Rumah: {'Milik Sendiri' if inp_rumah==1 else 'Kontrak/Sewa'}",
                            ]
                        ),
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f'<div class="pred-result-box {css_class}"><div class="pred-label" style="color:{color}">{icon} {label_text}</div>'
                        f'<div class="pred-sub">Diprediksi oleh Gaussian Naive Bayes untuk <strong>{inp_nama.strip()}</strong></div></div>',
                        unsafe_allow_html=True,
                    )

                    new_row = pd.DataFrame([{
                        "NAMA SISWA":           inp_nama.strip(),
                        "KELAS":                inp_kelas.strip(),
                        "NIS":                  inp_nis.strip(),
                        "PENDAPATAN ORANG TUA": int(inp_pendapatan),
                        "PEKERJAAN ORANG TUA":  int(inp_pekerjaan),
                        "JUMLAH TANGGUNGAN":    int(inp_tanggungan),
                        "STATUS RUMAH":         int(inp_rumah),
                        "LABEL (PREDIKSI)":     label_pred,
                    }])
                    st.session_state["data_tambahan"] = pd.concat(
                        [st.session_state["data_tambahan"], new_row], ignore_index=True
                    )

        # ── MODE 2 — Upload File (sama seperti semula) ─────
        else:
            st.markdown(
                """<div class="info-box">
                📂 Upload file CSV atau Excel berisi data siswa. File harus memiliki kolom:<br>
                <strong>NAMA SISWA</strong>, <strong>KELAS</strong>, <strong>NIS</strong>,
                <strong>PENDAPATAN ORANG TUA</strong> (0/1/2),
                <strong>PEKERJAAN ORANG TUA</strong> (0/1/2/3),
                <strong>JUMLAH TANGGUNGAN</strong> (angka),
                <strong>STATUS RUMAH</strong> (0/1)
                </div>""",
                unsafe_allow_html=True,
            )

            template_df = pd.DataFrame([{"NAMA SISWA":"Ahmad Fauzi","KELAS":"X IPA 1","NIS":"2024001","PENDAPATAN ORANG TUA":0,"PEKERJAAN ORANG TUA":0,"JUMLAH TANGGUNGAN":4,"STATUS RUMAH":0}])
            st.markdown(make_excel_download(template_df, "template_input_data.xlsx", "⬇ Unduh Template File (.xlsx)"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            uploaded_bulk = st.file_uploader("Upload file data siswa (CSV atau Excel)", type=["csv","xlsx"], key="uploader_bulk")

            if uploaded_bulk is not None:
                try:
                    if uploaded_bulk.name.endswith(".csv"):
                        df_bulk = pd.read_csv(uploaded_bulk)
                    else:
                        df_bulk = pd.read_excel(uploaded_bulk)

                    df_bulk.columns = df_bulk.columns.str.upper().str.strip()
                    required_cols   = ["PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH"]
                    missing_cols    = [c for c in required_cols if c not in df_bulk.columns]

                    if missing_cols:
                        st.markdown(notif_html("warn","❌","Kolom Tidak Ditemukan",f"Kolom berikut tidak ada dalam file: {', '.join(missing_cols)}",None), unsafe_allow_html=True)
                    else:
                        st.markdown(
                            notif_html(
                                "upload","📂",
                                f"File Berhasil Dibaca — {len(df_bulk)} baris data ditemukan",
                                f"File <strong>{uploaded_bulk.name}</strong> siap diproses. Klik tombol prediksi untuk mulai.",
                                [f"📋 Total baris: {len(df_bulk)}", f"📊 Kolom: {len(df_bulk.columns)}"]
                            ),
                            unsafe_allow_html=True,
                        )
                        st.dataframe(df_bulk.head(10), use_container_width=True)

                        if st.button("🔍 Prediksi Semua Data dari File", use_container_width=False):
                            fitur_cols  = ["PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH"]
                            X_bulk      = df_bulk[fitur_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
                            labels_pred = nb["model"].predict(X_bulk).tolist()

                            hasil_bulk = df_bulk.copy()
                            hasil_bulk["LABEL (PREDIKSI)"] = labels_pred

                            for col_id in ["NAMA SISWA","KELAS","NIS"]:
                                if col_id not in hasil_bulk.columns:
                                    hasil_bulk[col_id] = "-"

                            cols_order = ["NAMA SISWA","KELAS","NIS","PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL (PREDIKSI)"]
                            cols_exist = [c for c in cols_order if c in hasil_bulk.columns]
                            hasil_bulk = hasil_bulk[cols_exist]

                            st.session_state["data_tambahan"] = pd.concat(
                                [st.session_state["data_tambahan"], hasil_bulk], ignore_index=True
                            )

                            layak_count = sum(1 for l in labels_pred if l == 1)
                            tidak_count = sum(1 for l in labels_pred if l == 0)

                            st.markdown(
                                notif_html(
                                    "prep","✅",
                                    f"Prediksi Selesai — {len(df_bulk)} data berhasil diproses!",
                                    f"Semua data dari file telah diprediksi dan ditambahkan ke riwayat.",
                                    [
                                        f"🎓 Layak BSM: {layak_count}",
                                        f"📋 Tidak Layak: {tidak_count}",
                                        f"📊 Total: {len(df_bulk)}",
                                    ]
                                ),
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"""<div class="metric-row" style="margin-top:.8rem">
                                <div class="metric-card mc-teal"><span class="mc-icon">🎓</span><div class="mc-label">Layak</div><div class="mc-value cv-teal">{layak_count}</div></div>
                                <div class="metric-card mc-rose"><span class="mc-icon">📋</span><div class="mc-label">Tidak Layak</div><div class="mc-value cv-rose">{tidak_count}</div></div>
                                <div class="metric-card mc-navy"><span class="mc-icon">👥</span><div class="mc-label">Total</div><div class="mc-value cv-navy">{len(df_bulk)}</div></div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                except Exception as e:
                    st.markdown(notif_html("warn","❌","Error Membaca File",f"Terjadi error: {e}",None), unsafe_allow_html=True)

        # ── Riwayat Prediksi ──────────────────────────────
        st.markdown("---")
        dt = st.session_state["data_tambahan"]

        col_hd, col_hapus = st.columns([3, 1])
        with col_hd:
            st.markdown(
                f'<div class="section-header">📋 Riwayat Prediksi Siswa '
                f'<span style="color:#6B80A8;font-size:.85rem;font-family:Plus Jakarta Sans,sans-serif;font-weight:500;">'
                f'({len(dt)} siswa)</span></div>',
                unsafe_allow_html=True,
            )
        with col_hapus:
            if st.button("🗑 Hapus Semua Riwayat", key="hapus_riwayat"):
                st.session_state["data_tambahan"] = pd.DataFrame(
                    columns=["NAMA SISWA","KELAS","NIS","PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL (PREDIKSI)"]
                )
                st.rerun()

        if len(dt) > 0:
            dt_display = dt.copy()
            dt_display.insert(0, "No", range(1, len(dt_display)+1))

            label_map  = {0:"Tidak Layak (0)", 1:"Layak (1)"}
            rumah_map  = {0:"Kontrak/Sewa (0)", 1:"Milik Sendiri (1)"}
            kerja_map  = {0:"Petani (0)", 1:"PNS (1)", 2:"Polisi (2)", 3:"Wiraswasta (3)"}
            income_map = {0:"Rendah (0)", 1:"Menengah (1)", 2:"Tinggi (2)"}

            for col_name, mapping in [("LABEL (PREDIKSI)", label_map), ("STATUS RUMAH", rumah_map), ("PEKERJAAN ORANG TUA", kerja_map), ("PENDAPATAN ORANG TUA", income_map)]:
                if col_name in dt_display.columns:
                    dt_display[col_name] = dt_display[col_name].apply(lambda x: mapping.get(int(x), x) if pd.notna(x) else x)

            st.dataframe(dt_display, use_container_width=True, height=320)

            label_dist = dt["LABEL (PREDIKSI)"].value_counts().rename({0:"Tidak Layak", 1:"Layak"})
            layak_n    = int(dt["LABEL (PREDIKSI)"].sum())
            tidak_n    = len(dt) - layak_n
            st.markdown(
                notif_html(
                    "prep","📊",
                    f"Ringkasan Riwayat — {len(dt)} siswa telah diprediksi",
                    "Berikut distribusi kelayakan dari semua data yang telah dimasukkan.",
                    [f"🎓 Layak BSM: {layak_n}", f"📋 Tidak Layak: {tidak_n}", f"👥 Total: {len(dt)}"]
                ),
                unsafe_allow_html=True,
            )

            st.markdown("---")
            col_dl_a, col_dl_b, col_dl_c = st.columns(3)
            with col_dl_a:
                judul_tambahan = st.text_input("Judul file riwayat", value="Riwayat Prediksi BSM", key="judul_tambahan")
            with col_dl_b:
                st.markdown("<br>", unsafe_allow_html=True)
                filename_riwayat = f"{judul_tambahan.replace(' ', '_')}.xlsx"
                st.markdown(make_excel_download(dt, filename_riwayat, f"⬇ Unduh {judul_tambahan} (.xlsx)", judul_tambahan), unsafe_allow_html=True)
            with col_dl_c:
                if df_clean is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    dt_gabung = dt[["PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL (PREDIKSI)"]].copy()
                    dt_gabung = dt_gabung.rename(columns={"LABEL (PREDIKSI)":"LABEL"})
                    gabungan  = pd.concat([df_clean, dt_gabung], ignore_index=True)
                    filename_gabungan = f"{judul_tambahan.replace(' ', '_')}_Gabungan.xlsx"
                    st.markdown(make_excel_download(gabungan, filename_gabungan, f"⬇ Unduh Data Gabungan {judul_tambahan} (.xlsx)", judul_tambahan+" (Gabungan)"), unsafe_allow_html=True)

            st.markdown('<p style="font-size:.68rem;color:#6B80A8;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.65rem;margin-top:1rem;">Distribusi Kelayakan — Data Riwayat</p>', unsafe_allow_html=True)
            fig_dt, ax_dt = plt.subplots(figsize=(4, 2.4))
            fig_dt.patch.set_facecolor("#FFFFFF")
            ax_dt.set_facecolor("#F5F7FF")
            bar_c_dt = ["#FF2D6E" if "Tidak" in str(k) else "#00B896" for k in label_dist.index]
            bars_dt  = ax_dt.bar(label_dist.index, label_dist.values, color=bar_c_dt, edgecolor="#ffffff", width=0.5, linewidth=1.5)
            ax_dt.tick_params(colors="#6B80A8")
            ax_dt.set_ylabel("Jumlah Siswa", color="#6B80A8", fontsize=9)
            for spine in ax_dt.spines.values(): spine.set_edgecolor("#C5D0F0")
            for bar in bars_dt:
                ax_dt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, str(int(bar.get_height())), ha="center", va="bottom", color="#080F1E", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_dt)

        else:
            st.markdown(
                '<div style="text-align:center;color:#6B80A8;padding:2.5rem 2rem;'
                'border:2px dashed #C5D0F0;border-radius:16px;margin-top:1rem;background:#F5F7FF;">'
                '<div style="font-size:2.2rem;margin-bottom:.7rem;">🧑‍🎓</div>'
                '<div style="font-weight:800;color:#3D5070;margin-bottom:.3rem;">Belum Ada Data yang Dimasukkan</div>'
                '<div style="font-size:.8rem;">Isi form manual atau upload file di atas, lalu klik prediksi.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#6B80A8;font-size:.78rem;padding:.5rem 0 1rem;
                letter-spacing:.3px;font-family:'Plus Jakarta Sans',sans-serif;">
        🎓 Sistem Klasifikasi Beasiswa Siswa Miskin &nbsp;·&nbsp;
        Algoritma Gaussian Naive Bayes &nbsp;·&nbsp;
        Dibangun dengan <span style="color:#FF2D6E;">♥</span> menggunakan Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)