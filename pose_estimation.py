import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

# ✅ Model Path
model_path = r'C:\Users\Varshith\Downloads\files (4)\movenet_singlepose_lightning_4'

# ✅ Check if the model exists
saved_model_file = os.path.join(model_path, "saved_model.pb")
if not os.path.exists(saved_model_file):
    print("⚠️ Model not found! Downloading MoveNet...")
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    tf.saved_model.save(model, model_path)
    print("✅ Model downloaded and saved successfully.")

# ✅ Load Model
try:
    model = tf.saved_model.load(model_path)
    print("✅ Model loaded successfully.")

    # ✅ Get available signatures
    available_signatures = list(model.signatures.keys())
    print("🔍 Available signatures:", available_signatures)

    if "serving_default" not in available_signatures:
        raise ValueError("❌ No valid signatures found in the model!")

    # ✅ Get inference function
    infer = model.signatures["serving_default"]

except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise


def preprocess_image(image):
    """
    Preprocesses the image for MoveNet.

    :param image: Numpy array of shape (H, W, 3)
    :return: Preprocessed image tensor
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert to Tensor
    image = tf.image.resize_with_pad(image, 192, 192)  # Resize with padding
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.cast(image, tf.int32)  # Convert to int32
    return image


def movenet(input_image):
    """
    Runs MoveNet model on the input image.

    :param input_image: Numpy array of shape (H, W, 3)
    :return: Keypoints with scores
    """
    input_tensor = preprocess_image(input_image)
    outputs = infer(tf.constant(input_tensor))  # ✅ Run inference
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores


def extract_keypoints(image_path):
    """
    Reads an image and extracts keypoints using MoveNet.

    :param image_path: Path to the input image
    :return: Keypoints with scores
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Failed to read image from: {image_path}")

    keypoints_with_scores = movenet(image)
    return keypoints_with_scores


# ✅ Example Usage
image_path = r"C:\Users\Varshith\Downloads\archive\img_align_celeba\img_align_celeba\000001.jpg"  # Replace with actual image path

try:
    keypoints = extract_keypoints(image_path)
    print("📍 Keypoints Tensor:", keypoints)
except Exception as e:
    print(f"❌ Error during inference: {e}")