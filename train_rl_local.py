import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# ===== LOAD DL + TOKENIZER =====
dl_model = tf.keras.models.load_model("models/VULN_MODEL_SAVEDMODEL")

from tensorflow.keras.preprocessing.text import tokenizer_from_json # type: ignore
with open("models/tokenizer.json") as f:
    tokenizer = tokenizer_from_json(f.read())

MAX_LEN = 600

def clean_code(code):
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    code = re.sub(r"\s+", " ", code)
    return code.strip()
def dl_vulnerability_score(code):
    code = clean_code(code)
    seq = tokenizer.texts_to_sequences([code])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    infer = dl_model.signatures["serving_default"]
    outputs = infer(code_input=tf.convert_to_tensor(pad, dtype=tf.float32))

    score = list(outputs.values())[0].numpy()[0][0]
    return float(score)




# ===== ENV =====
class AdaptiveAttackEnv(gym.Env):
    def __init__(self, code_sample):
        super().__init__()
        self.code_sample = code_sample
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, (1,), np.float32)
        self.state = None

    def reset(self, seed=None, options=None):
        score = dl_vulnerability_score(self.code_sample)
        self.state = np.array([score], dtype=np.float32)
        return self.state, {}


    def step(self, action):
        score = self.state[0]

        if action == 1 and score > 0.6:
            reward = 1.0
        elif action == 2 and score > 0.5:
            reward = 0.7
        elif action == 3 and score > 0.55:
            reward = 0.8
        else:
            reward = -0.2

        terminated = True
        truncated = False
        return self.state, reward, terminated, truncated, {}


# TRAIN
code = """String query = "SELECT * FROM users WHERE id = " + request.getParameter("id");"""
env = AdaptiveAttackEnv(code)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=3000)

model.save("models/adaptive_attack_rl_agent.zip")
print("RL agent retrained & saved")

