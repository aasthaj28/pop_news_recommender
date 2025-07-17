import gym
import numpy as np
from gym import spaces

class NewsRecommendationEnv(gym.Env):
    def __init__(self, vectorizer, label_encoder, headline):
        super(NewsRecommendationEnv, self).__init__()
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.headline = headline
        self.vectorized = self.vectorizer.transform([self.headline]).toarray()
        self.action_space = spaces.Discrete(len(self.label_encoder.classes_))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.vectorized.shape[1],), dtype=np.float32)

    def reset(self):
        return self.vectorized.flatten()

    def step(self, action):
        done = True
