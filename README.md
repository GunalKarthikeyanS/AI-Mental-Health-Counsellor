# 🧠 AI Mental Health Counsellor

An AI-powered mental health support application that detects a user's emotion from text input and provides personalised, evidence-based mental health guidance using Large Language Models.

---

## What It Does

Users type how they are feeling in natural language. The app detects the underlying emotion using either a fine-tuned DistilBERT model or an LSTM model, then generates compassionate, personalised guidance using the Groq LLM API — including coping techniques, book and movie recommendations, and immediate actionable steps.

---

## Demo

![App Screenshot](assets/screenshot.png)

---

## Tech Stack

| Component | Technology |
|---|---|
| Emotion Detection (Deep Learning) | Fine-tuned DistilBERT (HuggingFace Transformers) |
| Emotion Detection (Classical DL) | LSTM with Keras/TensorFlow |
| Counsellor Response Generation | Groq API (LLaMA 3 8B) |
| Frontend | Streamlit |
| Dataset | Emotion-balanced dataset (6 classes) |
| Language | Python |

---

## Emotion Classes

The model detects 6 emotions:
- 😢 Sadness
- 😊 Joy
- ❤️ Love
- 😠 Anger
- 😨 Fear
- 😲 Surprise

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/GunalKarthikeyanS/AI-Mental-Health-Counsellor
cd AI-Mental-Health-Counsellor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
```bash
cp .env.example .env
# Open .env and add your Groq API key
```

### 4. Add model files
Place your trained model files in the `models/` folder:
```
models/
├── emotion_detection_fine_tuned_distilbert/
└── emotion_lstm_model/
    ├── emotion_lstm_model.h5
    └── tokenizer.pkl
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## Project Structure

```
AI-Mental-Health-Counsellor/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── .env.example            ← API key template
├── .gitignore              ← Files excluded from GitHub
├── models/                 ← Trained model files (not uploaded — too large)
│   ├── emotion_detection_fine_tuned_distilbert/
│   └── emotion_lstm_model/
└── data/                   ← Training datasets
```

---

## Key Features

- Dual model support — choose between DistilBERT and LSTM for emotion detection
- Fine-tuned DistilBERT on 6-class emotion dataset for high accuracy
- LLM-powered counsellor responses using CBT, DBT, and mindfulness techniques
- Clean, accessible Streamlit UI
- Secure API key handling via environment variables

---

## Team

Built as a group project for MS in Artificial Intelligence at Yeshiva University (Spring 2025).

Team members: Gunal Karthikeyan Saravanan, Shivendra Gupta

---

## Disclaimer

This application is for educational purposes only and is not a substitute for professional mental health care. If you are experiencing a mental health crisis, please contact a licensed professional or call a crisis helpline.
