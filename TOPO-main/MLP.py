import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from castle import MetricsDAG

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据
def load_data():
    datasets = []
    for dataset in range(1, 29):
        if dataset in {2, 3, 4, 6, 11, 12, 17}:  # 排除指定的数据集
            continue
        data = loadmat(f'../datasets/sims/sim{dataset}.mat')
        ts = data['ts']  # shape: (Ntimepoints, Nnodes)
        net = data['net']  # shape: (Nsubjects, Nnodes, Nnodes)

        Nsubjects = int(data['Nsubjects'][0][0])
        Ntimepoints = int(data['Ntimepoints'][0][0])
        Nnodes = int(data['Nnodes'][0][0])

        # Reshaping ts to include each subject's time series data
        ts_all = ts.reshape(Nsubjects, Ntimepoints, Nnodes)  # shape: (Nsubjects, Ntimepoints, Nnodes)
        X_combined = ts_all.reshape(-1, Nnodes)  # Flatten all subjects' time series into one dataset

        # Creating B_true_linear based on the network (binary adjacency matrix for each subject)
        B_true_linear = (net != 0).astype(int)  # Binary encoding of network
        B_true_linear = B_true_linear.reshape(Nsubjects, -1)  # Flatten the adjacency matrix to match each subject

        # Repeat each subject's B_true_linear for each time point to match the number of time points
        B_true_linear = np.tile(B_true_linear, (Ntimepoints, 1))  # Now shape: (Nsubjects * Ntimepoints, 25)

        print('first B true linear shape: ', B_true_linear.shape)


        datasets.append((X_combined, B_true_linear))
    return datasets

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_and_evaluate(X_combined, B_true_linear, Nnodes):
    print('X_combined shape:', X_combined.shape)
    print('B_true_linear shape:', B_true_linear.shape)

    # Split the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_combined, B_true_linear, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors and send to the appropriate device (GPU or CPU)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Define the model
    input_dim = X_train.shape[1]
    hidden_dim = 256
    output_dim = Nnodes * Nnodes  # Output dimension should match Nnodes x Nnodes
    model = MLP(input_dim, hidden_dim, output_dim).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy with logits
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Validation loop
    model.eval()
    with torch.no_grad():
        predictions = model(X_val_tensor).squeeze()  # Get the predicted DAG matrix

        # Reshape predictions to Nnodes x Nnodes matrix (correct shape for each sample)
        predictions_reshaped = predictions.view(-1, Nnodes, Nnodes).cpu().numpy()

        # Convert B_true_linear and B_est to binary format (0 or 1)
        B_est = (predictions_reshaped != 0).astype(int)  # Binary matrix of predictions
        B_true_linear = np.array(y_val_tensor.cpu()).astype(int)  # Binary matrix of true values

        # Ensure that B_est and B_true_linear are 2D and properly shaped
        B_est = B_est.reshape(-1, Nnodes * Nnodes)  # Flatten to match B_true_linear shape
        B_true_linear = B_true_linear.reshape(-1, Nnodes * Nnodes)

        # Calculate metrics using MetricsDAG
        met = MetricsDAG(B_est, B_true_linear)
        print("Metrics:", met.metrics)  # Prints SHD, Precision, Recall, F1

        # Calculate and print validation loss
        val_loss = mean_squared_error(y_val_tensor.cpu(), predictions.cpu())
        print(f"Validation Loss: {val_loss:.4f}")

# Main part of the code to run training
if __name__ == '__main__':
    datasets = load_data()  # Load your datasets
    for idx, (X_combined, B_true_linear) in enumerate(datasets):
        Nnodes = int(B_true_linear.shape[1] ** 0.5)  # Assuming B_true_linear is Nnodes^2
        print(f"Processing dataset {idx + 1} with Nnodes={Nnodes}...")
        train_and_evaluate(X_combined, B_true_linear, Nnodes)
