"""
Lab 04 : Phish or Legit ? Interface Streamlit.

Détection de messages phishing-style vs legit (dataset synthétique).
Lancement : streamlit run app.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support, brier_score_loss

from src.pipeline import load_data, split, make_pipeline, contributions, global_top_features

st.set_page_config(page_title="Phish or Legit?", page_icon="🛡️", layout="wide")

# --- style ---
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: linear-gradient(135deg,#fff,#f7f9fc);
    border:1px solid #e6eaf0; border-radius:14px; padding:14px 18px;
    box-shadow:0 2px 10px rgba(20,40,80,.05);
}
.hero {
    background:linear-gradient(135deg,#0f172a,#1e3a8a 60%,#2563eb);
    color:#fff; border-radius:18px; padding:26px 32px; margin-bottom:18px;
    box-shadow:0 10px 30px rgba(30,58,138,.25);
}
.hero h1 { color:#fff; margin:0 0 6px 0; font-size:2.1rem; }
.hero p  { color:#dbeafe; margin:0; }
.notice {
    background:#fff7ed; border-left:5px solid #f59e0b; border-radius:10px;
    padding:12px 16px; color:#7c2d12; margin-bottom:14px;
}
.result-phish {
    background:linear-gradient(135deg,#fef2f2,#fee2e2);
    border:1px solid #fecaca; border-radius:14px; padding:18px 22px; color:#7f1d1d;
}
.result-legit {
    background:linear-gradient(135deg,#f0fdf4,#dcfce7);
    border:1px solid #bbf7d0; border-radius:14px; padding:18px 22px; color:#14532d;
}
.result-title { font-size:1.25rem; font-weight:700; margin:0 0 4px 0; }
.result-sub   { font-size:.9rem; opacity:.85; margin:0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛡️ Phish or Legit?</h1>
  <p>Détection de messages <b>phishing-style</b> vs <b>legit</b> — TF-IDF + régression logistique.
  Chaque décision est expliquée mot par mot.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
  ⚠️ <b>This tool supports a human decision and does not block messages by itself.</b>
  The data is <b>synthetic</b>.
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_everything():
    df = load_data()
    Xtr, Xte, ytr, yte = split(df)
    pipes = {
        "Logistic Regression": make_pipeline("lr").fit(Xtr, ytr),
        "Naive Bayes":         make_pipeline("nb").fit(Xtr, ytr),
    }
    probs, metrics = {}, {}
    for name, pipe in pipes.items():
        p = pipe.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        metrics[name] = {
            "precision": pr, "recall": rc, "f1": f1,
            "accuracy": float((pred == yte).mean()),
            "brier": brier_score_loss(yte, p),
        }
        probs[name] = p
    thresholds = np.linspace(0.05, 0.95, 19)
    tradeoff = {"threshold": thresholds}
    for name, p in probs.items():
        prs, rcs = [], []
        for th in thresholds:
            pred = (p >= th).astype(int)
            pr, rc, _, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
            prs.append(pr); rcs.append(rc)
        tradeoff[f"{name} · precision"] = prs
        tradeoff[f"{name} · recall"] = rcs
    return pipes, metrics, pd.DataFrame(tradeoff).set_index("threshold")


pipes, metrics, tradeoff_df = load_everything()

# --- input ---
col_left, col_right = st.columns([1.05, 1.15], gap="large")

with col_left:
    st.markdown("### 1 · Choisis un message")

    EXAMPLES = {
        "🎣 Phishing évident":
            "Northbank: your account has been locked. Verify now http://bit.ly/ab12cd",
        "🎣 Phishing raté (FN, score ≈ 0.13)":
            "Dear customer, A payment of $7.67 is pending. You can manage this in "
            "the app settings. https://vg9ys-login.info/aqjv",
        "✅ Legit évident":
            "Hi, are we still on for dinner at 7:30? Thanks",
        "✅ Legit classé phishing (FP, score ≈ 0.61)":
            "Northban: there is a message about your account. Log in to confirm. "
            "https://docs.example.com/d/7estf1",
    }
    choice = st.selectbox("Message d'exemple :", list(EXAMPLES.keys()))
    text = st.text_area("Ou colle ton propre message :", EXAMPLES[choice], height=110)

    st.markdown("### 2 · Seuil de décision")
    threshold = st.slider("Phish si score ≥ seuil", 0.0, 1.0, 0.5, 0.01,
        help="Monte = plus de précision. Baisse = plus de rappel.")
    st.button("🔍 Classify", type="primary", use_container_width=True)

    st.markdown("### 3 · Modèle")
    model_name = st.radio("Classifieur :", list(pipes.keys()), horizontal=True)
    model = pipes[model_name]

with col_right:
    st.markdown("### Résultat")
    if not text.strip():
        st.info("Colle un message à gauche.")
    else:
        score = float(model.predict_proba([text])[0][1])
        label = "phish" if score >= threshold else "legit"

        if label == "phish":
            st.markdown(f"""
            <div class="result-phish">
              <p class="result-title">⚠️ Phishing-style détecté</p>
              <p class="result-sub">Score : <b>{score:.3f}</b> · seuil : {threshold:.2f}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legit">
              <p class="result-title">✅ Message légitime</p>
              <p class="result-sub">Score : <b>{score:.3f}</b> · seuil : {threshold:.2f}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="position:relative;margin:8px 0 4px 0;">
          <div style="height:14px;border-radius:8px;
               background:linear-gradient(90deg,#22c55e,#eab308,#ef4444);opacity:.35;"></div>
          <div style="position:absolute;left:calc({score*100}% - 1px);top:-6px;
               width:3px;height:26px;background:#0f172a;border-radius:2px;"></div>
          <div style="position:absolute;left:calc({threshold*100}% - 1px);top:-3px;
               width:2px;height:20px;background:#2563eb;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:.78rem;color:#64748b;">
          <span>0.0 (legit)</span>
          <span>seuil {threshold:.2f} (bleu) · score {score:.2f} (noir)</span>
          <span>1.0 (phish)</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### Pourquoi cette décision ?")

        contrib = contributions(model, text)
        pos = [(k, v) for k, v in contrib.items() if v > 0][:8]
        neg = [(k, v) for k, v in contrib.items() if v < 0][:8]

        weights = {k: v for k, v in contrib.items() if " " not in k}
        parts = []
        for tok in re.findall(r"\w+|\W+", text):
            w = weights.get(tok.lower(), 0.0)
            if tok.strip() and re.match(r"\w+", tok) and abs(w) > 0.3:
                parts.append(f":red[**{tok}**]" if w > 0 else f":green[**{tok}**]")
            else:
                parts.append(tok)
        st.markdown("".join(parts))
        st.caption("🔴 rouge = pousse vers phishing · 🟢 vert = pousse vers legit")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**↑ augmente le score**")
            if pos:
                st.bar_chart(pd.DataFrame(pos, columns=["feature", "contribution"]).set_index("feature"),
                             color="#ef4444", height=220)
        with c2:
            st.markdown("**↓ diminue le score**")
            if neg:
                st.bar_chart(pd.DataFrame(neg, columns=["feature", "contribution"]).set_index("feature"),
                             color="#22c55e", height=220)

# --- tabs ---
st.divider()
tab_how, tab_metrics, tab_tradeoff = st.tabs(["📘 How it works", "📊 Metrics", "🎚️ Threshold trade-off"])

with tab_how:
    st.markdown("""
    ### Le concept en 4 étapes
    1. **Nettoyage** : URLs, montants, nombres → `urltok`, `moneytok`, `numtok`.
    2. **TF-IDF** : chaque mot/bigramme devient un nombre (fréquent ici, rare ailleurs = poids fort).
    3. **Classification** : NB compte les mots par classe ; LR apprend un coefficient par feature.
    4. **Décision** : phish si score ≥ seuil (le seuil dépend de la politique).
    """)
    pos, neg = global_top_features(pipes["Logistic Regression"], k=8)
    a, b = st.columns(2)
    with a:
        st.markdown("**Associées à *phishing***")
        st.table(pd.DataFrame(pos, columns=["feature", "coefficient"]))
    with b:
        st.markdown("**Associées à *legit***")
        st.table(pd.DataFrame(neg, columns=["feature", "coefficient"]))

with tab_metrics:
    lr = metrics["Logistic Regression"]
    k1, k2, k3, k4 = st.columns(4)
    c1, c2, c3, c4 = st.columns(4)
    for col, title, value in zip(
        [c1, c2, c3, c4],
        ["Precision", "Recall", "F1", "Brier"],
        [lr['precision'], lr['recall'], lr['f1'], lr['brier']]
    ):
        col.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255,255,255,0.2); 
                    border-radius: 10px; padding: 15px; text-align: center;">
            <p style="color: #a0aec0; margin: 0; font-size: 0.9rem;">{title}</p>
            <p style="color: #ffffff; margin: 0; font-size: 1.8rem; font-weight: bold;">{value:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(metrics).T.round(3), use_container_width=True)
    img = Path("results/phish_results.png")
    if img.exists():
        st.image(str(img), use_container_width=True)
        st.caption("NB + calibration (rose) est la mieux calibrée, suivie de la LR (verte), tandis que NB brut (bleu) est surconfiant.")

with tab_tradeoff:
    st.line_chart(tradeoff_df, height=340)
    st.markdown(f"**Seuil actuel : `{threshold:.2f}`** — monter = precision ↑, recall ↓.")
    st.dataframe(pd.DataFrame({
        "seuil": [0.20, 0.35, 0.50, 0.70, 0.90],
        "precision": [0.604, 0.672, 0.747, 0.893, 1.000],
        "recall": [0.930, 0.800, 0.710, 0.500, 0.090],
        "F1": [0.732, 0.731, 0.728, 0.641, 0.165],
    }), use_container_width=True, hide_index=True)

st.divider()
st.caption("Lab 04 : Phish or Legit · dataset synthétique · TF-IDF + LogisticRegression · Streamlit")