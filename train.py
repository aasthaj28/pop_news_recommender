import json
import numpy as np
import pickle
from gym import spaces

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from environment.news_env import NewsRecommendationEnv

# Load data
with open("data/News_Category_Dataset_v3.json", "r") as f:
    data = [json.loads(line) for line in f]

headlines = [item['headline'] for item in data]
categories = [item['category'] for item in data]

# Preprocessing
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(headlines)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(categories)

# Save vectorizer and encoder
with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Custom environment class for batch training
class TrainEnv(NewsRecommendationEnv):
    def __init__(self, vectorizer, label_encoder, X, y):
        self.X = X
        self.y = y
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.idx = 0
        self.n = X.shape[0]
        self.action_space = spaces.Discrete(len(label_encoder.classes_))
        self.observation_space = spaces.Box(low=0, high=1, shape=(X.shape[1],), dtype=np.float32)

    def reset(self):
        self.idx = np.random.randint(0, self.n)
        return self.X[self.idx].toarray().flatten()

    def step(self, action):
        correct = (action == self.y[self.idx])
        reward = 1 if correct else 0
        done = True
        obs = self.reset()
        return obs, reward, done, {}

# Wrap environment
env = DummyVecEnv([lambda: TrainEnv(vectorizer, label_encoder, X, y)])

# Train PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save model
model.save("model/ppo_news_model.zip")
