from turtle import forward
import torch 
import torch.nn as nn
import torch.optim as optim  
import numpy as np 
from sklearn.neighbors import kneighbors_graph


class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn(self.fc(x)))  
        out = out + residual 
        return out

    
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.5):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.res_block1 = ResidualBlock(hidden_size, hidden_size)
        self.res_block2 = ResidualBlock(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.fc_out(x)
        return x
    

def manifold_regularization(output, graph_laplacian, gamma):
    if not isinstance(graph_laplacian, torch.Tensor):
        graph_laplacian = torch.tensor(graph_laplacian, dtype=torch.float32, device=output.device)
    else:
        graph_laplacian = graph_laplacian.to(output.device)
    manifold_loss = gamma * torch.trace(torch.matmul(torch.matmul(output.T, graph_laplacian), output))
    return manifold_loss


def create_graph_laplacian(data, neighbors=5, mode='connectivity'):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    adjacency_matrix = kneighbors_graph(
        data, 
        n_neighbors=neighbors, 
        mode=mode, 
        include_self=False
    ).toarray()
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    laplacian = degree_matrix - adjacency_matrix
    return torch.tensor(laplacian, dtype=torch.float32)   

        
        
    
     
        
