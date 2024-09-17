from turtle import forward
import torch 
import torch.nn as nn
import torch.optim as optim  
import numpy as np 
from sklearn.neighbors import kneighbors_graph

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size) 
        self.fc4 = nn.Linear(hidden_size, num_classes)
    
    def forward(self,x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x 
    
def manifold_regularization(output, graph_laplacian, gamma):
    if not isinstance(graph_laplacian, torch.Tensor):
        graph_laplacian = torch.tensor(graph_laplacian, dtype=torch.float32)
    manifold_loss = gamma * torch.trace(torch.matmul(torch.matmul(output.T, graph_laplacian), output))
    return manifold_loss

def create_graph_laplacian(data, neighbors=5):
    adjacency_matrix = kneighbors_graph(data,neighbors,mode='connectivity',include_self=True).toarray()
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    laplacian = degree_matrix - adjacency_matrix
    return torch.tensor(laplacian, dtype=torch.float32)
    

        
        
    
     
        