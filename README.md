# Squat Classification using GNN

## Overview
This project classifies squats as "correct" or "incorrect" based on keypoint data from pose estimation. The project uses a Graph Neural Network (GNN) to analyze squat movements and the trained model is converted to ONNX format for deployment.

## Files
- `pose_estimation.py`: Script to extract keypoints using MoveNet.
- `data_preprocessing.py`: Script for data preprocessing and labeling.
- `gnn_model.py`: Script for GNN model creation and training.
- `onnx_conversion.py`: Script to convert the trained GNN model to ONNX format.
- `test_model.py`: Script to evaluate the model's performance.
- `preprocessed_data.pkl`: Preprocessed data in graph format.
- `gnn_model.pth`: Trained GNN model.
- `gnn_model.onnx`: Trained GNN model in ONNX format.
- `README.md`: Documentation.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- OpenCV
- NetworkX
- ONNX

## Setup
Install the required libraries:
```bash
pip install torch torchvision opencv-python-headless networkx onnx
```

## Usage
1. **Pose Estimation**:
   Extract keypoints from images using the pose estimation script:
   ```bash
   python pose_estimation.py
   ```

2. **Data Preprocessing**:
   Preprocess the extracted keypoints and label the data:
   ```bash
   python data_preprocessing.py
   ```

3. **Train GNN Model**:
   Train the GNN model on the preprocessed data:
   ```bash
   python gnn_model.py
   ```

4. **Convert to ONNX**:
   Convert the trained GNN model to the ONNX format:
   ```bash
   python onnx_conversion.py
   ```

5. **Test Model**:
   Evaluate the model's performance:
   ```bash
   python test_model.py
   ```

## Contact
