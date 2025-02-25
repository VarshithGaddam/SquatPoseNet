import torch
import pickle
import os
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from gnn_model import GCN

def evaluate_model(model, data_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_list:
            out = model(data.x, data.edge_index)
            # Use the first node's output for classification
            pred = out[0].argmax(dim=0)  # Get prediction for the first node
            correct += int(pred == data.y[0])  # Compare with the label
            total += 1
    
    accuracy = correct / total
    return accuracy

def create_test_data():
    """
    Create a small test dataset if none exists.
    Returns a list of Data objects.
    """
    test_data = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create 10 simple test graphs
    for i in range(10):
        G = nx.Graph()
        # Add nodes with random features
        for j in range(3):  # 3 nodes per graph
            G.add_node(j, x=np.random.rand(), y=np.random.rand(), score=np.random.rand())
        # Add edges
        G.add_edges_from([(0, 1), (1, 2)])
        
        # Convert to PyTorch Geometric Data object
        node_features = np.array([
            [G.nodes[node_id]['x'], 
             G.nodes[node_id]['y'], 
             G.nodes[node_id]['score']]
            for node_id in G.nodes()
        ])
        
        x = torch.tensor(node_features, dtype=torch.float).to(device)
        edge_index = torch.tensor(list(G.edges())).t().contiguous().to(device)
        y = torch.tensor([np.random.randint(0, 2)], dtype=torch.long).to(device)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        test_data.append(data)
    
    return test_data

def load_test_data(test_data_path):
    """
    Load test data and ensure it's in the correct format.
    """
    if not os.path.exists(test_data_path):
        print("⚠️ Test data not found. Creating a small test dataset...")
        test_data = create_test_data()
        # Save the test data for future use
        with open(test_data_path, 'wb') as f:
            pickle.dump(test_data, f)
        return test_data
    else:
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
            # Ensure the data is in the correct format
            if isinstance(test_data, list) and all(isinstance(d, Data) for d in test_data):
                return test_data
            else:
                print("⚠️ Test data format is incorrect. Regenerating test data...")
                test_data = create_test_data()
                with open(test_data_path, 'wb') as f:
                    pickle.dump(test_data, f)
                return test_data

# Load test data
test_data_path = r'C:\Users\Varshith\Downloads\files (4)\test_data.pkl'
test_data = load_test_data(test_data_path)

# Initialize and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features=3, hidden_channels=16, num_classes=2).to(device)
model.load_state_dict(torch.load('gnn_model.pth'))

# Evaluate model
accuracy = evaluate_model(model, test_data)
print(f'Model Accuracy: {accuracy:.2f}')