import numpy as np
import networkx as nx
import json
import os
import pickle
from pose_estimation import extract_keypoints
from tqdm import tqdm  # For progress bar

def load_coco_annotations(annotations_path):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def label_squats(keypoints):
    # Placeholder function to label squats as correct or incorrect
    # Replace this with your own logic based on keypoint analysis
    labels = []
    for kp in keypoints:
        if np.random.rand() > 0.5:
            labels.append(1)  # Correct squat
        else:
            labels.append(0)  # Incorrect squat
    return labels

def convert_to_graph(keypoints, labels):
    graphs = []
    for kp, label in zip(keypoints, labels):
        G = nx.Graph()
        for i, point in enumerate(kp[0]):
            G.add_node(i, x=point[0], y=point[1], score=point[2])
        graphs.append((G, label))
    return graphs

def process_image(img):
    """
    Process a single image and return its keypoints.
    """
    img_path = os.path.join(images_dir, img['file_name'])
    if not os.path.exists(img_path):
        return None
        
    try:
        return extract_keypoints(img_path)
    except Exception as e:
        print(f"⚠️ Error processing image {img['file_name']}: {e}")
        return None

# Load annotations
annotations_path = r"C:\Users\Varshith\Downloads\archive\coco2017\annotations\person_keypoints_train2017.json"
images_dir = r"C:\Users\Varshith\Downloads\archive\coco2017\train2017"

# Verify dataset exists
if not os.path.exists(annotations_path):
    raise FileNotFoundError(f"❌ Annotations file not found: {annotations_path}")

if not os.path.exists(images_dir):
    raise FileNotFoundError(f"❌ Images directory not found: {images_dir}")

annotations = load_coco_annotations(annotations_path)

# Process images with progress bar
keypoints = []
for img in tqdm(annotations['images'][:100]):  # Process first 100 images for testing
    result = process_image(img)
    if result is not None:
        keypoints.append(result)
    print(f"Processed {len(keypoints)} images so far...")

# Label squats and convert to graph format
labels = label_squats(keypoints)
graphs = convert_to_graph(keypoints, labels)

# Save preprocessed data
output_path = r'C:\Users\Varshith\Downloads\files (4)\preprocessed_data.pkl'
try:
    with open(output_path, 'wb') as f:
        pickle.dump(graphs, f)
    print(f"✅ Preprocessed data saved to {output_path}")
except Exception as e:
    print(f"❌ Error saving preprocessed data: {e}")
    raise