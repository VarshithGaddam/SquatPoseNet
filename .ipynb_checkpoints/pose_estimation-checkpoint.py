import cv2
import numpy as np
import tensorflow as tf

# Load the MoveNet model
model_path = r'C:\Users\Varshith\Downloads\files (4)\movenet_singlepose_lightning_4'
model = tf.saved_model.load(model_path)

# Check available signatures
available_signatures = list(model.signatures.keys())
print("Available signatures:", available_signatures)

# Use the correct signature (replace 'serving_default' with the correct one if needed)
signature = 'serving_default' if 'serving_default' in available_signatures else available_signatures[0]

def movenet(input_image):
    # Prepare input image
    input_image = tf.convert_to_tensor(input_image, dtype=tf.int32)
    input_image = tf.image.resize_with_pad(input_image, 192, 192)
    input_image = tf.expand_dims(input_image, axis=0)

    # Run model inference
    outputs = model.signatures[signature](input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    
    return keypoints_with_scores

def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read the image from the path: {image_path}")
    keypoints_with_scores = movenet(image)
    return keypoints_with_scores

# Example usage
image_path = r"C:\Users\Varshith\Downloads\archive\img_align_celeba\img_align_celeba\202529.jpg"  # Replace with the correct image path
keypoints = extract_keypoints(image_path)
print(keypoints)