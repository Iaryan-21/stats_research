import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import NeuralNet, create_graph_laplacian, manifold_regularization

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)

def generate_torus(n_samples=1000, noise=0.05):
    t, p = np.random.uniform(0,2*np.pi,(2,n_samples))
    x = (1+0.5*np.cos(p))*np.cos(t)
    y = (1+0.5*np.cos(p))*np.sin(t)
    z = 0.5*np.sin(p)
    data = np.stack((x,y,z), axis=-1) + np.random.normal(0,noise,(n_samples,3))
    labels = (np.cos(t) > 0).astype(int)
    return data, labels

def generate_swiss_roll(n_samples=1000, noise=0.05):
    data, color = datasets.make_swiss_roll(n_samples=n_samples, noise=noise)
    labels = (color > np.median(color)).astype(int)
    return data, labels

def generate_data(manifold_type, n_samples=1000):
    return {'torus': generate_torus, 'swiss_roll': generate_swiss_roll}[manifold_type](n_samples)

manifold_type = 'torus'
X, y = generate_data(manifold_type, n_samples=1500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train, y_train = torch.FloatTensor(X_train).to(device), torch.LongTensor(y_train).to(device)
X_test, y_test = torch.FloatTensor(X_test).to(device), torch.LongTensor(y_test).to(device)

model = NeuralNet(input_size=X_train.shape[1], hidden_size=64, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

graph_laplacian = torch.FloatTensor(create_graph_laplacian(X_train.cpu().numpy())).to(device)
graph_laplacian.requires_grad_(True)

num_epochs, gamma = 1000, 0.0001

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    classification_loss = criterion(outputs, y_train)
    manifold_loss = manifold_regularization(outputs, graph_laplacian, gamma)
    
    total_loss = classification_loss + manifold_loss
    total_loss.backward()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}")
        print(f"Classification Loss: {classification_loss.item():.4f}")
        print(f"Manifold Loss: {manifold_loss.item():.4f}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.abs().mean():.4f}")
            else:
                print(f"No gradient for {name}")
        if graph_laplacian.grad is not None:
            print(f"Gradient for graph_laplacian: {graph_laplacian.grad.abs().mean():.4f}")
        else:
            print("No gradient for graph_laplacian")
    
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).float().mean()
        print(f'Epoch [{epoch+1}/{num_epochs}], Class. Loss: {classification_loss.item():.4f}, '
              f'Manifold Loss: {manifold_loss.item():.4f}, Total Loss: {total_loss.item():.4f}, '
              f'Test Accuracy: {accuracy.item():.4f}')


model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f'Final Test Accuracy: {accuracy.item():.4f}')

def plot_decision_boundary(X, y, model, device, manifold_type):
    x_min, x_max = X[:,0].min()-0.5,X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5,X[:,1].max()+0.5
    z_min, z_max = X[:,2].min()-0.5,X[:,2].max()+0.5
    
    grid_size = 20
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                             np.linspace(y_min, y_max, grid_size),
                             np.linspace(z_min, z_max, grid_size))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    mesh_tensor = torch.FloatTensor(scaler.transform(mesh_points).astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        Z = torch.argmax(model(mesh_tensor), dim=1).cpu().numpy().reshape(xx.shape)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.RdYlBu, edgecolor='black', s=30)
    ax.contourf(xx[:,:,0], yy[:,:,0], Z[:,:,10], zdir='z', offset=z_min, cmap=plt.cm.RdYlBu, alpha=0.5)
    ax.contourf(xx[:,0,:], Z[:,10,:], zz[:,0,:], zdir='y', offset=y_max, cmap=plt.cm.RdYlBu, alpha=0.5)
    ax.contourf(Z[10,:,:], yy[0,:,:], zz[0,:,:], zdir='x', offset=x_min, cmap=plt.cm.RdYlBu, alpha=0.5)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title(f'{manifold_type.capitalize()} with Decision Boundary')
    

    plt.colorbar(scatter, ax=ax, label='Class')
    plt.tight_layout()
    plt.show()

plot_decision_boundary(X, y, model, device, manifold_type)
