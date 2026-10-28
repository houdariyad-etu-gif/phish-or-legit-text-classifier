import re
import streamlit as st
from src.pipeline import load_data, split, make_pipeline, predict, contributions

st.set_page_config(page_title="Phish or Legit?")
st.title("Phish or Legit?")
st.caption("TF-IDF + logistic regression on a SYNTHETIC dataset (fictional brands). "
           "Labels are 'phishing-style' vs 'legit', not real-world ground truth.")

st.warning("⚠️ This tool supports a human decision and does not block messages by itself. "
           "The data is synthetic.")

@st.cache_resource
def get_model():
    df = load_data()
    Xtr, _, ytr, _ = split(df)
    return make_pipeline("lr").fit(Xtr, ytr)

model = get_model()

# --- 4 messages d'exemple (2 phish, 2 legit, dont 1 mal classé) ---
EXAMPLES = {
    "Phishing (évident)": "Northbank: your account has been locked. Verify now http://bit.ly/ab12cd",
    "Phishing (raté par le modèle, FN)": "Dear customer, A payment of $7.67 is pending. You can manage this in the app settings. https://vg9ys-login.info/aqjv",
    "Legit (évident)": "Hi, are we still on for dinner at 7:30? Thanks",
    "Legit (classé phishing, FP)": "Northban: there is a message about your account. Log in to confirm. https://docs.example.com/d/7estf1",
}
choice = st.selectbox("Message d'exemple :", list(EXAMPLES.keys()))
default_text = EXAMPLES[choice]
text = st.text_area("Paste a message:", default_text, height=100)

# --- slider de seuil ---
threshold = st.slider("Decision threshold (phish si score ≥ seuil)", 0.0, 1.0, 0.5, 0.01)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.error("The message is empty - paste some text to classify.")
    else:
        score = float(model.predict_proba([text])[0][1])
        label = "phish" if score >= threshold else "legit"

        if label == "phish":
            st.error(f"⚠️ Looks like PHISHING-style text  ·  score {score:.2f}  ·  seuil {threshold:.2f}")
        else:
            st.success(f"✅ Looks legit  ·  score for phish {score:.2f}  ·  seuil {threshold:.2f}")

        # --- explication : positif ET négatif ---
        contrib = contributions(model, text)
        pos = [(k, v) for k, v in contrib.items() if v > 0][:6]
        neg = [(k, v) for k, v in contrib.items() if v < 0][:6]

        # surlignage
        weights = {k: v for k, v in contrib.items() if " " not in k}  # unigrammes
        parts = []
        for tok in re.findall(r"\w+|\W+", text):
            w = weights.get(tok.lower(), 0.0)
            if tok.strip() and re.match(r"\w+", tok) and abs(w) > 0.3:
                parts.append(f":red[**{tok}**]" if w > 0 else f":green[**{tok}**]")
            else:
                parts.append(tok)
        st.markdown("".join(parts))
        st.caption("rouge = pousse vers phishing, vert = pousse vers legit (mots à poids fort)")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Features qui augmentent le score")
            st.table({"feature": [t for t, _ in pos], "contribution": [round(v, 2) for _, v in pos]})
        with c2:
            st.subheader("Features qui diminuent le score")
            st.table({"feature": [t for t, _ in neg], "contribution": [round(v, 2) for _, v in neg]})














# """Lab 04 - Phish or Legit? (Streamlit UI). Run:  streamlit run app.py"""
# import re

# import streamlit as st

# from src.pipeline import load_data, split, make_pipeline, predict, word_weights, contributions

# st.set_page_config(page_title="Phish or Legit?")
# st.title("Phish or Legit?")
# st.caption("TF-IDF + logistic regression on a SYNTHETIC dataset (fictional brands). "
#            "Labels are 'phishing-style' vs 'legit', not real-world ground truth.")


# @st.cache_resource
# def get_model():
#     df = load_data()
#     Xtr, _, ytr, _ = split(df)
#     return make_pipeline("lr").fit(Xtr, ytr), make_pipeline("lr_cal").fit(Xtr, ytr)


# model, model_cal = get_model()
# text = st.text_area("Paste a message:", "Northbank: your account has been locked. Verify now http://bit.ly/ab12cd",
#                     height=100)

# if st.button("Classify", type="primary"):
#     try:
#         label, score = predict(model, text)
#         _, cal_score = predict(model_cal, text)
#     except ValueError as err:
#         st.error(str(err))
#     else:
#         if label == "phish":
#             st.error(f"⚠️ Looks like PHISHING-style text  ·  model score {score:.2f}")
#         else:
#             st.success(f"✅ Looks legit  ·  model score for phish {score:.2f}")
#         st.caption(f"Calibrated score (sigmoid calibration on training data): {cal_score:.2f}. "
#                    "Scores are only meaningful as probabilities if calibration was checked - see the lab report.")

#         weights = word_weights(model, text)
#         parts = []
#         for tok in re.findall(r"\w+|\W+", text):
#             w = weights.get(tok.lower(), 0.0)
#             if tok.strip() and re.match(r"\w+", tok) and abs(w) > 0.3:
#                 parts.append(f":red[**{tok}**]" if w > 0 else f":green[**{tok}**]")
#             else:
#                 parts.append(tok)
#         st.markdown("".join(parts))
#         st.caption("red = pushes toward phishing, green = pushes toward legit (words with a strong weight only)")
#         top = list(contributions(model, text).items())[:6]
#         st.table({"feature": [t for t, _ in top], "contribution to log-odds": [round(v, 2) for _, v in top]})
