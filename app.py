import streamlit as st
import numpy as np
import random
import pickle
from stable_baselines3 import PPO
from environment.news_env import NewsRecommendationEnv

@st.cache_resource
def load_model_and_preprocessors():
    model = PPO.load("model/ppo_news_model.zip")
    with open("model/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model_and_preprocessors()

st.title("📰 PPO-based News Recommender")

headline_input = st.text_input("Enter a News Headline")

if st.button("Recommend Category"):
    if headline_input.strip() == "":
        st.warning("Please enter a headline.")
    else:
        vectorized_headline = vectorizer.transform([headline_input])
        env = NewsRecommendationEnv(vectorizer, label_encoder, headline_input)
        obs = env.reset()
        action, _ = model.predict(obs)
        predicted_category = label_encoder.inverse_transform([action])[0]
        st.success(f"📢 Recommended Category: **{predicted_category}**")

