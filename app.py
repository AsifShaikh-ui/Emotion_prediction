import streamlit as st
import joblib
import numpy as np

# =====================================================
# LOAD SAVED COMPONENTS
# =====================================================
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =====================================================
# APP TITLE
# =====================================================
st.set_page_config(page_title="Emotion Prediction App", page_icon="😊", layout="centered")

st.title("🎭 Emotion Detection from Text")
st.write("Enter a sentence below and the model will predict the underlying emotion.")

# =====================================================
# USER INPUT
# =====================================================
user_input = st.text_area("🗣️ Your Text", placeholder="Type something like 'I am feeling great today!'")

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔍 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform input
        text_vec = vectorizer.transform([user_input])
        pred = model.predict(text_vec)
        pred_prob = model.predict_proba(text_vec)

        # Decode emotion label
        emotion = label_encoder.inverse_transform(pred)[0]

        # Confidence score
        confidence = np.max(pred_prob) * 100

        # Emoji map
        emoji_dict = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😡",
            "fear": "😨",
            "love": "❤️",
            "surprise": "😲",
            "neutral": "😐"
        }

        # Show results
        st.markdown(f"### Predicted Emotion: **{emotion.upper()}** {emoji_dict.get(emotion, '')}")
        st.progress(int(confidence))
        st.caption(f"Model Confidence: {confidence:.2f}%")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Logistic Regression")
