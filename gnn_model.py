import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import os
import pickle
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_model(graphs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_list = []
    for graph, label in graphs:
        # Extract node features
        node_features = np.array([
            [graph.nodes[node_id]['x'], 
             graph.nodes[node_id]['y'], 
             graph.nodes[node_id]['score']]
            for node_id in graph.nodes()
        ])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float).to(device)
        
        # Handle edge indices
        edges = list(graph.edges())
        if len(edges) == 0:
            # Add self-loops if no edges exist
            num_nodes = len(graph.nodes())
            edges = [(i, i) for i in range(num_nodes)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        y = torch.tensor([label], dtype=torch.long).to(device)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    # Create DataLoader
    loader = DataLoader(data_list, batch_size=32, shuffle=True)
    
    # Initialize model
    model = GCN(num_features=3, hidden_channels=16, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(1000):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            
            # Calculate loss for each graph in the batch
            batch_size = data.y.size(0)
            losses = []
            for i in range(batch_size):
                # Use the first node's output for classification
                loss = criterion(out[i][0].unsqueeze(0), data.y[i].unsqueeze(0))
                losses.append(loss)
            
            # Average the losses and backpropagate
            loss = torch.mean(torch.stack(losses))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(loader)}')
    
    return model

# Load preprocessed data
data_path = r'C:\Users\Varshith\Downloads\files (4)\preprocessed_data.pkl'
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Preprocessed data not found at {data_path}\n"
        "Please run data_preprocessing.py first to generate the data."
    )

with open(data_path, 'rb') as f:
    graphs = pickle.load(f)

# Train the model
model = train_model(graphs)
torch.save(model.state_dict(), r'C:\Users\Varshith\Downloads\files (4)\gnn_model.pth')