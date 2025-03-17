import numpy as np
import tensorflow as tf
import os
import pickle

# Load Dataset
DATASET_PATH = "dataset/"
GESTURES = ["thumbs_up", "thumbs_down", "peace", "stop", "fist", "ok"]

X, Y = [], []

for label, gesture in enumerate(GESTURES):
    gesture_path = os.path.join(DATASET_PATH, gesture)
    for file in os.listdir(gesture_path):
        data = np.load(os.path.join(gesture_path, file))
        X.append(data)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)

# Save Label Encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(GESTURES, f)

# Build Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(GESTURES), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X, Y, epochs=20)

# Save Model
model.save("models/gesture_model.h5")

print("✅ Model trained and saved successfully!")
