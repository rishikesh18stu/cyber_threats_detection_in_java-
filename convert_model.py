import tensorflow as tf

old_model = tf.keras.models.load_model("models/VULN_MODEL_SAVEDMODEL", compile=False)
old_model.save("models/VULN_MODEL_CPU.keras", save_format="keras")

print("Model converted successfully")
