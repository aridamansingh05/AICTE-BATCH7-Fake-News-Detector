import streamlit as st
import pickle
from utils.preprocess import clean_text

# Load model and vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.title("📰Fake News Detector")

user_input = st.text_area("Paste News Article Here")

if st.button("Analyze"):
    
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    # Get probabilities properly
    probs = model.predict_proba(vectorized)[0]
    classes = model.classes_

    real_index = list(classes).index("REAL")
    real_prob = probs[real_index]

    confidence = round(real_prob * 100, 2)

    # Credibility logic
    if prediction == "REAL":
        credibility = confidence
    else:
        credibility = round((1 - real_prob) * 100, 2)

    st.subheader("Result")
    st.write("Prediction:", prediction)
    st.write("Confidence:", confidence, "%")
    st.write("Credibility Score:", credibility, "/100")
