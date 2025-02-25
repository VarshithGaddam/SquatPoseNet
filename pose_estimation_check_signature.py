import tensorflow as tf
import os

# Load the MoveNet model
model_path = r'C:\Users\Varshith\Downloads\files (4)\movenet_singlepose_lightning_4'
saved_model_file = os.path.join(model_path, "saved_model.pb")

if os.path.exists(saved_model_file):
    try:
        model = tf.saved_model.load(model_path)
        print("✅ Model loaded successfully.")
        
        # Print available signatures
        available_signatures = list(model.signatures.keys())
        print("🔍 Available signatures:", available_signatures)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("⚠️ Model not found. Please run download_movenet.py to download and save the model.")