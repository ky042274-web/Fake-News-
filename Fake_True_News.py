import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ---------- ADVANCED CSS ----------
st.markdown("""
<style>

/* -------- LIGHT BACKGROUND WITH SOFT OVERLAY -------- */
.stApp {
    background: linear-gradient(rgba(255,255,255,0.65), rgba(255,255,255,0.65)),
                url("https://images.unsplash.com/photo-1495020689067-958852a7765e");
    background-size: cover;
    background-attachment: fixed;
}

/* -------- MAIN CONTAINER (LIGHT GLASS) -------- */
.glass {
    background: rgba(255, 255, 255, 0.92);  /* light + readable */
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 20px;
    color: #000000;  /* dark text for contrast */
    box-shadow: 0px 4px 20px rgba(0,0,0,0.2);
}

/* -------- TITLE -------- */
.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #000;
}

.subtitle {
    text-align: center;
    color: #333;
    margin-bottom: 20px;
}

/* -------- RESULT BOX -------- */
.result-real {
    background: rgba(40,167,69,0.9);
    color: white;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.result-fake {
    background: rgba(220,53,69,0.9);
    color: white;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

/* -------- TEXT AREA -------- */
textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 16px !important;
}

/* -------- BUTTON -------- */
.stButton button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
}

/* -------- SIDEBAR TEXT -------- */
.css-1d391kg, .stMarkdown, .stText {
    color: #000 !important;
}

/* -------- INFO / MODEL INSIGHT -------- */
.stAlert {
    background-color: rgba(255,255,255,0.95) !important;
    color: #000 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_resources():
    model = load_model("fake_news_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()
max_len = 200

# ---------- FUNCTIONS ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

def predict_news(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)[0][0]
    return pred

# ---------- SIDEBAR ----------
st.sidebar.title("📌 About System")
st.sidebar.write("""
**AI Fake News Detection System**

✔ Deep Learning (LSTM)  
✔ NLP Preprocessing  
✔ Binary Classification  

📊 Dataset: Kaggle Fake & Real News  

💡 This system predicts authenticity of news content
""")

# ---------- MAIN ----------
st.markdown('<div class="glass">', unsafe_allow_html=True)

st.markdown('<div class="title">📰 AI Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning based News Authenticity Checker</div>', unsafe_allow_html=True)

# ---------- LAYOUT ----------
col1, col2 = st.columns([2, 1])

# ===== LEFT SIDE (INPUT) =====
with col1:
    st.subheader("✍️ Enter News Content")
    user_input = st.text_area("Paste news article here...", height=200)

    analyze_btn = st.button("🔍 Analyze News")

# ===== RIGHT SIDE (INFO PANEL) =====
with col2:
    st.subheader("ℹ️ System Info")
    st.write("""
    - Model: LSTM  
    - Max Length: 200  
    - Output: Probability Score  
    """)

# ---------- RESULT ----------
if analyze_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter news text")
    elif len(user_input.split()) < 5:
        st.warning("⚠️ Please enter detailed content")
    else:
        with st.spinner("Analyzing using Deep Learning model..."):
            prediction = predict_news(user_input)

        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if prediction > 0.5:
            st.markdown('<div class="result-real">✅ REAL NEWS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fake">❌ FAKE NEWS</div>', unsafe_allow_html=True)

        # ---------- CONFIDENCE ----------
        st.write("### Confidence Level")
        st.progress(float(confidence))
        st.write(f"**Confidence Score:** {confidence:.2f}")

        # ---------- SIMPLE EXPLANATION (DL IMPORTANT) ----------
        st.write("### 🧠 Model Insight")
        st.info("""
        The model analyzes patterns, word sequences, and linguistic structure 
        using LSTM to determine whether the news is likely real or fake.
        """)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("🚀 Deep Learning Project | LSTM + NLP | Streamlit Deployment")

st.markdown('</div>', unsafe_allow_html=True)