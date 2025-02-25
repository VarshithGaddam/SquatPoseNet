import torch
import torch.onnx
from gnn_model import GCN

def convert_to_onnx(model_path, onnx_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model with correct parameters
    model = GCN(num_features=3, hidden_channels=16, num_classes=2).to(device)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create dummy input
    x = torch.randn(1, 3)  # Batch of 1 node with 3 features
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop
    
    # Export the model
    torch.onnx.export(
        model,
        (x, edge_index),
        onnx_path,
        input_names=['x', 'edge_index'],
        output_names=['output'],
        dynamic_axes={
            'x': {0: 'num_nodes'},
            'edge_index': {0: 'num_edges'}
        }
    )
    print(f"Model successfully exported to {onnx_path}")

# Convert the model
convert_to_onnx('gnn_model.pth', 'gnn_model.onnx')