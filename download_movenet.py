import tensorflow as tf
import tensorflow_hub as hub
import os

# URL of the MoveNet SinglePose Lightning model on TensorFlow Hub
model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

# Load the model from TensorFlow Hub
model = hub.load(model_url)

# Set the model save path to the specified directory
model_save_path = r"C:\Users\Varshith\Downloads\files (4)\movenet_singlepose_lightning_4"

# Ensure the directory exists
os.makedirs(model_save_path, exist_ok=True)

# Save the model with the correct signature
module = hub.load(model_url)
concrete_function = module.signatures['serving_default']
tf.saved_model.save(module, model_save_path, signatures={'serving_default': concrete_function})

print(f"Model saved to {model_save_path}")