from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

app = Flask(__name__)


# Load the trained model
MODEL_PATH = "C:/Users/mouly/OneDrive/Desktop/h/backend/models/gesture_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
@app.route("/", methods=["GET"])
def home():
    return "Flask server is running!"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input

    prediction = model.predict(img)
    label = np.argmax(prediction)

    return jsonify({"gesture": str(label)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
