import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="RenalOncoPredict · Kidney Diagnostics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD ----------------
model    = joblib.load("model.pkl")
pca      = joblib.load("pca.pkl")
scaler   = joblib.load("scaler.pkl")
qt       = joblib.load("quantile.pkl")
le       = joblib.load("label_encoder.pkl")
features = joblib.load("features.pkl")

# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@300;400&display=swap');

/* ── Root variables ── */
:root {
    --navy:       #0a0e1a;
    --navy-mid:   #111827;
    --navy-card:  #161d2e;
    --navy-border:#1e2d45;
    --gold:       #c9a84c;
    --gold-light: #e8c97a;
    --gold-dim:   #8a6d2f;
    --teal:       #00c9b1;
    --teal-dim:   #00896e;
    --text-primary:   #e8eaf0;
    --text-secondary: #8b9ab2;
    --text-muted:     #4a5568;
    --danger:     #ef4444;
    --success:    #10b981;
    --warn-bg:    #1a1408;
    --warn-border:#5a3e0a;
}

/* ── Reset & body ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy) !important;
    color: var(--text-primary);
}
.main { background-color: var(--navy) !important; }
.block-container {
    padding: 0 2.5rem 4rem !important;
    max-width: 1400px !important;
}
section[data-testid="stSidebar"] { display: none; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Top nav bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0 1rem;
    border-bottom: 1px solid var(--navy-border);
    margin-bottom: 2.8rem;
}
.brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.brand-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--gold), var(--gold-dim));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--navy);
}
.brand-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.02em;
}
.brand-name span { color: var(--gold); }
.brand-sub {
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 1px;
    font-family: 'DM Mono', monospace;
}
.nav-pills {
    display: flex;
    gap: 6px;
}
.nav-pill {
    padding: 6px 16px;
    border-radius: 50px;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-secondary);
    border: 1px solid var(--navy-border);
    font-family: 'DM Mono', monospace;
    cursor: default;
}
.nav-pill.active {
    background: rgba(201,168,76,0.12);
    border-color: var(--gold-dim);
    color: var(--gold);
}
.version-badge {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    border: 1px solid var(--navy-border);
    border-radius: 4px;
    padding: 3px 10px;
    letter-spacing: 0.1em;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2.6rem 0 1.2rem;
}
.section-rule {
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, var(--navy-border), transparent);
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
}
.section-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--gold);
    opacity: 0.6;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] > div:first-child {
    background: var(--navy-card) !important;
    border: 1.5px dashed var(--navy-border) !important;
    border-radius: 14px !important;
    padding: 2.5rem !important;
    text-align: center;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"] > div:first-child:hover {
    border-color: var(--gold-dim) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary) !important;
    font-size: 14px;
}
.upload-hint {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    margin-top: 0.4rem;
}

/* ── Data table ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--navy-border) !important;
}
[data-testid="stDataFrame"] table {
    background: var(--navy-card) !important;
}
[data-testid="stDataFrame"] th {
    background: var(--navy-mid) !important;
    color: var(--gold) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--navy-border) !important;
}
[data-testid="stDataFrame"] td {
    color: var(--text-secondary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    border-bottom: 1px solid rgba(30,45,69,0.5) !important;
}

/* ── Stat cards ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 1.6rem 0;
}
.stat-card {
    background: var(--navy-card);
    border: 1px solid var(--navy-border);
    border-radius: 14px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(to right, var(--gold), transparent);
}
.stat-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 36px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.stat-value .unit {
    font-size: 16px;
    color: var(--text-muted);
    margin-left: 3px;
}
.stat-label {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-secondary);
    font-family: 'DM Mono', monospace;
    margin-top: 6px;
}
.stat-icon-text {
    position: absolute;
    top: 18px; right: 20px;
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    font-weight: 300;
    color: var(--gold);
    opacity: 0.18;
    line-height: 1;
    user-select: none;
}

/* ── Sample header ── */
.sample-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2rem 0 0.8rem;
}
.sample-badge {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--teal);
    background: rgba(0,201,177,0.08);
    border: 1px solid rgba(0,201,177,0.2);
    border-radius: 4px;
    padding: 3px 12px;
}
.sample-line {
    flex: 1;
    height: 1px;
    background: var(--navy-border);
}

/* ── Diagnosis cards ── */
.diagnosis-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 1.2rem;
}
.diag-card {
    background: var(--navy-card);
    border: 1px solid var(--navy-border);
    border-radius: 16px;
    padding: 28px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease;
}
.diag-card.high-conf {
    border-color: rgba(0,201,177,0.35);
    background: linear-gradient(135deg, var(--navy-card), rgba(0,201,177,0.04));
}
.diag-card.high-conf::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(to right, var(--teal), transparent);
}
.diag-card.med-conf::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(to right, var(--gold), transparent);
}
.diag-card-num {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 14px;
}
.diag-class {
    font-family: 'Cormorant Garamond', serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.03em;
    line-height: 1;
    margin-bottom: 6px;
}
.diag-class.high { color: var(--teal); }
.diag-class.med  { color: var(--gold); }
.diag-full-name {
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 20px;
}
.conf-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 5px;
    overflow: hidden;
    margin-bottom: 6px;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
}
.conf-bar-fill.high { background: linear-gradient(to right, var(--teal-dim), var(--teal)); }
.conf-bar-fill.med  { background: linear-gradient(to right, var(--gold-dim), var(--gold)); }
.conf-text {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
}

/* ── Legend / Info section ── */
.info-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 1.2rem;
}
.info-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: var(--navy-card);
    border: 1px solid var(--navy-border);
    border-radius: 12px;
    padding: 16px 18px;
}
.info-abbr {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: var(--gold);
    min-width: 60px;
}
.info-desc {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── Warning banner ── */
.warn-banner {
    margin-top: 2rem;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 10px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 13px;
    color: #c9a84c;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.04em;
}

/* ── Plotly charts background fix ── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly bg {
    background: transparent !important;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid var(--navy-border) !important;
    margin: 2.4rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    border-top: 1px solid var(--navy-border);
    margin-top: 3rem;
}
.footer-brand {
    font-family: 'Cormorant Garamond', serif;
    font-size: 18px;
    color: var(--gold);
    letter-spacing: 0.06em;
}
.footer-copy {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# TOP NAV BAR
# ============================================================
st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">R</div>
    <div>
      <div class="brand-name">RenalOnco<span>Predict</span></div>
      <div class="brand-sub">Renal Oncology · Genomic Intelligence Platform</div>
    </div>
  </div>
  <div class="nav-pills">
    <div class="nav-pill active">Diagnostics</div>
    <div class="nav-pill">Cohort View</div>
    <div class="nav-pill">Reports</div>
  </div>
  <div class="version-badge">v2.4.1 · Clinical Research Build</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# UPLOAD SECTION
# ============================================================
st.markdown("""
<div class="section-header">
  <div class="section-dot"></div>
  <div class="section-label">Patient Genomic Data</div>
  <div class="section-rule"></div>
</div>
""", unsafe_allow_html=True)

col_up, col_hint = st.columns([3, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "",
        type=["csv"],
        help="Upload normalized gene expression CSV. Each row = one patient sample."
    )

with col_hint:
    st.markdown("""
    <div style="background:var(--navy-card);border:1px solid var(--navy-border);
                border-radius:14px;padding:20px 22px;height:100%;box-sizing:border-box;">
      <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.18em;
                  text-transform:uppercase;color:var(--gold);margin-bottom:10px;">
        Expected Format
      </div>
      <div style="font-size:12px;color:var(--text-secondary);line-height:1.8;">
        · CSV, UTF-8 encoded<br>
        · Rows: patient samples<br>
        · Columns: gene expression values<br>
        · Header row required
      </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# RESULTS (only shown after upload)
# ============================================================
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # ── Pre-process ──
    X = data.copy()
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]
    X_qt      = qt.transform(X)
    X_scaled  = scaler.transform(X_qt)
    X_pca     = pca.transform(X_scaled)
    preds     = model.predict(X_pca)
    labels    = le.inverse_transform(preds)
    probs     = model.predict_proba(X_pca)
    class_names = le.classes_

    n_samples    = len(data)
    top_classes  = [class_names[np.argmax(probs[i])] for i in range(len(probs))]
    avg_conf     = np.mean([np.max(probs[i]) for i in range(len(probs))])
    high_conf_n  = sum(1 for i in range(len(probs)) if np.max(probs[i]) >= 0.90)

    # ── Stats row ──
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-icon-text">Σ</div>
        <div class="stat-value">{n_samples}<span class="unit">pts</span></div>
        <div class="stat-label">Samples Analysed</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon-text">μ</div>
        <div class="stat-value">{avg_conf*100:.0f}<span class="unit">%</span></div>
        <div class="stat-label">Avg. Prediction Probability</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon-text">✓</div>
        <div class="stat-value">{high_conf_n}<span class="unit">/{n_samples}</span></div>
        <div class="stat-label">High Probability ≥ 90%</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon-text">#</div>
        <div class="stat-value">{len(set(top_classes))}<span class="unit">cls</span></div>
        <div class="stat-label">Distinct Classes Found</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data Preview ──
    st.markdown("""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-label">Data Preview</div>
      <div class="section-rule"></div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(data.head(), use_container_width=True)

    # ── Confidence Analysis ──
    st.markdown("""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-label">Prediction Confidence Analysis</div>
      <div class="section-rule"></div>
    </div>
    """, unsafe_allow_html=True)

    CHART_COLORS = ["#00c9b1", "#c9a84c", "#3b82f6", "#8b5cf6"]

    for i in range(len(probs)):
        # Sample badge
        remaining = len(probs) - i - 1
        st.markdown(f"""
        <div class="sample-header">
          <div class="sample-badge">Sample {i+1:02d}</div>
          <div class="sample-line"></div>
          <div style="font-family:'DM Mono',monospace;font-size:11px;color:var(--text-muted);">
            {remaining} remaining in queue
          </div>
        </div>
        """, unsafe_allow_html=True)

        df_prob = pd.DataFrame({
            "Classification": class_names,
            "Probability": probs[i]
        }).sort_values("Probability", ascending=False)

        fig = go.Figure()
        for j, row in df_prob.iterrows():
            fig.add_trace(go.Bar(
                x=[row["Classification"]],
                y=[row["Probability"]],
                name=row["Classification"],
                marker=dict(
                    color=CHART_COLORS[list(class_names).index(row["Classification"]) % len(CHART_COLORS)],
                    opacity=0.85,
                    line=dict(width=0)
                ),
                text=[f"{row['Probability']:.3f}"],
                textposition="outside",
                textfont=dict(
                    family="DM Mono, monospace",
                    size=12,
                    color="#8b9ab2"
                ),
                width=0.45,
            ))

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(family="DM Mono, monospace", size=12, color="#8b9ab2"),
                linecolor="#1e2d45",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(30,45,69,0.6)",
                gridwidth=1,
                zeroline=False,
                tickformat=".0%",
                tickfont=dict(family="DM Mono, monospace", size=10, color="#4a5568"),
                range=[0, 1.15],
                linecolor="#1e2d45",
            ),
            bargroupgap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"conf_chart_{i}")

    # ── Final Diagnosis ──
    st.markdown("""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-label">Final Diagnosis</div>
      <div class="section-rule"></div>
    </div>
    """, unsafe_allow_html=True)

    CLASS_FULL = {
        "CCRCC":  "Clear Cell Renal Cell Carcinoma",
        "PRCC":   "Papillary Renal Cell Carcinoma",
        "CHRCC":  "Chromophobe Renal Cell Carcinoma",
        "NORMAL": "Healthy Tissue · No Malignancy Detected",
    }

    CARDS_PER_ROW = 3
    n_diag = len(labels)

    for row_start in range(0, n_diag, CARDS_PER_ROW):
        row_indices = range(row_start, min(row_start + CARDS_PER_ROW, n_diag))
        cols = st.columns(len(row_indices))

        for col, i in zip(cols, row_indices):
            top_idx     = np.argmax(probs[i])
            cls         = class_names[top_idx]
            probability = probs[i][top_idx]
            prob_pct    = int(probability * 100)
            full_name   = CLASS_FULL.get(cls, cls)

            if probability >= 0.85:
                accent_color  = "#00c9b1"
                text_color    = "#00c9b1"
                border_color  = "rgba(0,201,177,0.35)"
                bar_gradient  = "linear-gradient(to right, #00896e, #00c9b1)"
                bg_extra      = "rgba(0,201,177,0.04)"
            else:
                accent_color  = "#c9a84c"
                text_color    = "#c9a84c"
                border_color  = "rgba(201,168,76,0.35)"
                bar_gradient  = "linear-gradient(to right, #8a6d2f, #c9a84c)"
                bg_extra      = "rgba(201,168,76,0.04)"

            with col:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #161d2e, {bg_extra});
                    border: 1px solid {border_color};
                    border-radius: 16px;
                    padding: 28px;
                    position: relative;
                    overflow: hidden;
                    margin-bottom: 16px;
                ">
                  <div style="
                      position: absolute; top: 0; left: 0; right: 0;
                      height: 2px;
                      background: {bar_gradient};
                  "></div>
                  <div style="
                      font-family:'DM Mono',monospace;
                      font-size:10px;
                      letter-spacing:0.2em;
                      color:#4a5568;
                      text-transform:uppercase;
                      margin-bottom:14px;
                  ">Sample · {i+1:02d}</div>
                  <div style="
                      font-family:'Cormorant Garamond',serif;
                      font-size:32px;
                      font-weight:700;
                      color:{text_color};
                      letter-spacing:0.03em;
                      line-height:1;
                      margin-bottom:6px;
                  ">{cls}</div>
                  <div style="
                      font-size:12px;
                      color:#8b9ab2;
                      margin-bottom:20px;
                      line-height:1.4;
                  ">{full_name}</div>
                  <div style="
                      background:rgba(255,255,255,0.06);
                      border-radius:100px;
                      height:5px;
                      overflow:hidden;
                      margin-bottom:8px;
                  ">
                    <div style="
                        width:{prob_pct}%;
                        height:100%;
                        border-radius:100px;
                        background:{bar_gradient};
                    "></div>
                  </div>
                  <div style="
                      display:flex;
                      justify-content:space-between;
                      font-family:'DM Mono',monospace;
                      font-size:11px;
                      color:#4a5568;
                  ">
                    <span>Prediction Probability</span>
                    <span style="color:{accent_color};">{probability:.4f}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Clinical Interpretation ──
    st.markdown("""
    <div class="section-header" style="margin-top:3rem;">
      <div class="section-dot"></div>
      <div class="section-label">Clinical Interpretation Guide</div>
      <div class="section-rule"></div>
    </div>
    <div class="info-grid">
      <div class="info-item">
        <div class="info-abbr">CCRCC</div>
        <div class="info-desc">Clear Cell Renal Cell Carcinoma · Most common subtype (~70%). Associated with VHL gene mutations.</div>
      </div>
      <div class="info-item">
        <div class="info-abbr">PRCC</div>
        <div class="info-desc">Papillary Renal Cell Carcinoma · Second most common (~15%). Type 1 & 2 variants with distinct molecular profiles.</div>
      </div>
      <div class="info-item">
        <div class="info-abbr">CHRCC</div>
        <div class="info-desc">Chromophobe Renal Cell Carcinoma · Favourable prognosis (~5%). Arises from intercalated cells of collecting duct.</div>
      </div>
      <div class="info-item">
        <div class="info-abbr">NORMAL</div>
        <div class="info-desc">Healthy Renal Tissue · No oncogenic expression signature detected within classification threshold.</div>
      </div>
    </div>
    <div class="warn-banner">
      ⚠ &nbsp; This platform is intended exclusively for clinical research and academic use. 
      Results do not constitute a medical diagnosis and must not replace histopathological evaluation by a qualified pathologist.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Empty state ──
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;background:var(--navy-card);
                border:1px dashed var(--navy-border);border-radius:20px;margin-top:1rem;">
      <div style="font-family:'Cormorant Garamond',serif;font-size:64px;font-weight:300;
                  color:var(--gold);opacity:0.15;margin-bottom:18px;line-height:1;">Rx</div>
      <div style="font-family:'Cormorant Garamond',serif;font-size:26px;color:var(--text-secondary);
                  font-weight:600;margin-bottom:8px;">
        Awaiting Genomic Input
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:12px;color:var(--text-muted);
                  letter-spacing:0.12em;max-width:420px;margin:0 auto;line-height:1.8;">
        Upload a patient gene expression CSV above to begin AI-powered renal subtype classification.
        The model analyses thousands of genomic features in seconds.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
  <div class="footer-brand">RenalOncoPredict</div>
  <div class="footer-copy">
    Renal Oncology Genomics Platform · Research Use Only · 
    Model: KNN + PCA Pipeline · Trained on TCGA-KIRC/KIRP/KICH
  </div>
</div>
""", unsafe_allow_html=True)