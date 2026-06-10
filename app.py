import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from groq import Groq
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────
# Model Paths (relative — works on any machine)
# ─────────────────────────────────────────────
DISTILBERT_MODEL_PATH = "models/emotion_detection_fine_tuned_distilbert"
LSTM_MODEL_PATH       = "models/emotion_lstm_model/emotion_lstm_model.h5"
TOKENIZER_PATH        = "models/emotion_lstm_model/tokenizer.pkl"

# Emotion labels
EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# ─────────────────────────────────────────────
# Load Groq API Key from environment variable
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
# Load DistilBERT model
# ─────────────────────────────────────────────
@st.cache_resource
def load_distilbert_model():
    model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_PATH)
    model.eval()
    return model, tokenizer

# ─────────────────────────────────────────────
# Load LSTM model
# ─────────────────────────────────────────────
@st.cache_resource
def load_lstm_model():
    model = load_model(LSTM_MODEL_PATH)
    tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
    return model, tokenizer

# ─────────────────────────────────────────────
# Emotion prediction — LSTM
# ─────────────────────────────────────────────
def predict_emotion_lstm(text, model, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)
    top_index = np.argsort(pred[0])[-1]
    return EMOTION_LABELS[int(top_index)]

# ─────────────────────────────────────────────
# Emotion prediction — DistilBERT
# ─────────────────────────────────────────────
def predict_emotion_distilbert(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    label_map = {i: label for i, label in enumerate(EMOTION_LABELS)}
    return label_map[predicted_class]

# ─────────────────────────────────────────────
# Generate mental health response via Groq LLM
# ─────────────────────────────────────────────
def generate_counsellor_response(emotion):
    prompt = f"""
You are a compassionate licensed psychologist and mental health coach.
A person says they feel {emotion}.

1. Suggest a scientifically backed technique (e.g., CBT, DBT, ACT, or mindfulness) to manage this emotion.
2. Briefly mention a possible underlying cause they might reflect on.
3. Recommend 3 books and 3 movies that can help them understand and cope better.
4. Suggest a small 5-minute action they can take immediately to feel better.
5. Share one short affirmation to offer hope.

Be kind, realistic, and use bullet points. Keep it concise.
"""
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="AI Mental Health Counsellor", page_icon="🧠")
st.title("🧠 AI Mental Health Counsellor")
st.write("Share how you're feeling. Our AI detects your emotion and provides personalised mental health guidance.")

model_choice = st.selectbox("Choose Emotion Detection Model", ["DistilBERT", "LSTM"])
user_input = st.text_area("📝 How are you feeling today?", placeholder="e.g. I feel overwhelmed and anxious about everything...")

if st.button("Analyze & Get Guidance"):
    if user_input.strip():
        with st.spinner("Analyzing your emotion..."):

            if model_choice == "DistilBERT":
                bert_model, bert_tokenizer = load_distilbert_model()
                detected_emotion = predict_emotion_distilbert(user_input, bert_model, bert_tokenizer)
            else:
                lstm_model, lstm_tokenizer = load_lstm_model()
                detected_emotion = predict_emotion_lstm(user_input, lstm_model, lstm_tokenizer)

            st.success(f"**Detected Emotion:** `{detected_emotion.upper()}`")

            counsellor_response = generate_counsellor_response(detected_emotion)
            st.markdown("### 🌟 Personalised Guidance")
            st.write(counsellor_response)
    else:
        st.warning("Please enter how you're feeling before clicking Analyze.")
