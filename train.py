import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from model import NeuralNet, create_graph_laplacian, manifold_regularization
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data generation
n_samples = 40000  
n_features = 10 

X_class_0 = np.random.rand(n_samples // 2, n_features) + np.array([-2] * n_features)
X_class_1 = np.random.rand(n_samples // 2, n_features) + np.array([2] * n_features)
X = np.vstack((X_class_0, X_class_1))
y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
shuffle_indices = np.random.permutation(n_samples)
X = X[shuffle_indices]
y = y[shuffle_indices]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors and move to device
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.LongTensor(y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.LongTensor(y_test).to(device)

# Model parameters
input_size = X_train.shape[1]
hidden_size = 64
num_classes = 2

# Initialize the model and move to device
model = NeuralNet(input_size=10, hidden_size=128, num_classes=2).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Create graph Laplacian and move to device
graph_laplacian = create_graph_laplacian(X_train.cpu().numpy())  # Create Laplacian from CPU data
graph_laplacian = torch.FloatTensor(graph_laplacian).to(device)  # Move to GPU

# Training parameters
num_epochs = 10000  # Increased number of epochs
gamma = 0.001  # Adjusted manifold regularization strength

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    classification_loss = criterion(outputs, y_train)
    manifold_loss = manifold_regularization(outputs, graph_laplacian, gamma)
    total_loss = classification_loss + manifold_loss
    
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:  
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).float().mean()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Classification Loss: {classification_loss.item():.4f}, '
              f'Manifold Loss: {manifold_loss.item():.4f}, '
              f'Total Loss: {total_loss.item():.4f}, '
              f'Test Accuracy: {accuracy.item():.4f}')

# Final evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f'Final Test Accuracy: {accuracy.item():.4f}')

# Visualization
def plot_decision_boundary(X, y, model, pca=None):
    # Ensure X and y are numpy arrays
    X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
    y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Flatten the mesh grid points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # If PCA was used, inverse transform the mesh points
    if pca is not None:
        mesh_points = pca.inverse_transform(mesh_points)
    
    # Convert to PyTorch tensor and ensure it's float32
    mesh_tensor = torch.FloatTensor(mesh_points.astype(np.float32)).to(device)
    
    # Get predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        Z = model(mesh_tensor)
        Z = torch.argmax(Z, dim=1).cpu().numpy()
    
    # Reshape the predictions
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar()
    plt.show()

# Usage in your main script:
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train.cpu())  # Move back to CPU for PCA and plotting
plot_decision_boundary(X_train_2d, y_train.cpu(), model, pca)
