from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import re
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

app = Flask(__name__)

MODEL_PATH = "models/VULN_MODEL_SAVEDMODEL"
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

dl_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open("models/tokenizer.json") as f:
    tokenizer = tokenizer_from_json(f.read())

MAX_LEN = 600

def clean_code(code):
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    code = re.sub(r"\s+", " ", code)
    return code.strip()

def predict_vulnerability(code):
    seq = tokenizer.texts_to_sequences([clean_code(code)])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    score = dl_model.predict(padded, verbose=0)[0][0]
    return float(score)

def rule_based_attack(score):
    if score > 0.75:
        return "High Risk: SQL Injection / Command Injection"
    elif score > 0.5:
        return "Medium Risk: Cross-Site Scripting (XSS)"
    elif score > 0.3:
        return "Low Risk: Input Validation Issues"
    else:
        return "No significant vulnerability detected"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        code = request.form["code"]
        score = predict_vulnerability(code)
        attack = rule_based_attack(score)
        result = {"score": round(score, 3), "attack": attack}
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
