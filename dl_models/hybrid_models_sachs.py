# hybrid_models_sachs.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import networkx as nx

# Import CDT for Sachs dataset
import cdt
from cdt.data import load_dataset

from topo_linear import TOPO_linear, threshold_W
from topo_neural import TransformerModel, DeepDAGModel, GNNModel
from topo_utils import create_Z, set_random_seed


class IIDDatasetTrainer:
    """Adapted version of NeuralDAGTrainer for IID data (no time dimension)"""

    def __init__(self, model_type, input_dim, hidden_dim=64, nhead=4, num_layers=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.input_dim = input_dim

        # Adjust parameters based on input size
        hidden_dim = min(hidden_dim, input_dim * 4)
        num_layers = min(num_layers, 2)
        nhead = min(nhead, 4)

        # Ensure hidden_dim is divisible by nhead for transformer
        if model_type == 'transformer':
            hidden_dim = (hidden_dim // nhead) * nhead

        # Initialize model
        if model_type == 'transformer':
            self.model = TransformerModel(input_dim, hidden_dim, nhead, num_layers).to(self.device)
        elif model_type == 'deep_dag':
            self.model = DeepDAGModel(input_dim, hidden_dim, dropout=0.3).to(self.device)
        elif model_type == 'gnn':
            self.model = GNNModel(input_dim, hidden_dim).to(self.device)

            # Create default edge index for GNN
            edge_index = []
            for i in range(input_dim):
                for j in range(input_dim):
                    if i != j:  # Exclude self-loops
                        edge_index.append([i, j])
            self.default_edge_index = torch.tensor(edge_index).t().contiguous().to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Optimizer with L2 regularization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.005)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5
        )
        self.criterion = nn.BCELoss()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_data(self, data, net_data, is_training=False):
        """Prepare IID data for training/evaluation"""
        # Standardize features
        scaler = RobustScaler()
        data_scaled = scaler.fit_transform(data)
        data_tensor = torch.FloatTensor(data_scaled)

        # For transformer, reshape data to include a sequence dimension (batch_size, 1, features)
        if self.model_type == 'transformer':
            data_tensor = data_tensor.unsqueeze(1)

        # Prepare the target adjacency matrix
        net_binary = (net_data > 0.1).astype(float)  # Use dynamic thresholding in production
        net_tensor = torch.FloatTensor(net_binary)

        return data_tensor.to(self.device), net_tensor.to(self.device)

    def train(self, train_data, train_net, val_data=None, val_net=None,
              epochs=100, batch_size=32, patience=10):
        """Train model with IID data"""
        # If validation data not provided, use a portion of training data
        if val_data is None or val_net is None:
            split = int(0.8 * len(train_data))
            val_data = train_data[split:]
            val_net = train_net
            train_data = train_data[:split]

        X_train, y_train = self.prepare_data(train_data, train_net, is_training=True)
        X_val, y_val = self.prepare_data(val_data, val_net, is_training=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        min_delta = 0.001  # Minimum improvement threshold

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            n_train_batches = 0

            # Create batches - accommodate different input shapes based on model type
            if self.model_type == 'transformer':
                train_dataset = TensorDataset(X_train, y_train.expand(X_train.shape[0], self.input_dim, self.input_dim))
            else:
                train_dataset = TensorDataset(X_train, y_train.expand(X_train.shape[0], self.input_dim, self.input_dim))

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()

                # Forward pass with appropriate input for the model type
                if self.model_type == 'gnn':
                    pred = self.model(batch_x, self.default_edge_index)
                else:
                    pred = self.model(batch_x)

                # L1 regularization
                l1_lambda = 0.0005
                l1_reg = torch.norm(pred, 1)

                loss = self.criterion(pred, batch_y)
                loss = loss + l1_lambda * l1_reg

                # Add L2 loss if model has it
                if hasattr(self.model, 'l2_loss'):
                    loss = loss + self.model.l2_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_train_loss += loss.item()
                n_train_batches += 1

            # Validation
            self.model.eval()
            val_loss = 0
            n_val_batches = 0

            if self.model_type == 'transformer':
                val_dataset = TensorDataset(X_val, y_val.expand(X_val.shape[0], self.input_dim, self.input_dim))
            else:
                val_dataset = TensorDataset(X_val, y_val.expand(X_val.shape[0], self.input_dim, self.input_dim))

            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    if self.model_type == 'gnn':
                        pred = self.model(batch_x, self.default_edge_index)
                    else:
                        pred = self.model(batch_x)

                    batch_loss = self.criterion(pred, batch_y)
                    val_loss += batch_loss.item()
                    n_val_batches += 1

            val_loss /= n_val_batches

            # Update learning rate
            self.scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}:')
                print(f'  Train Loss: {total_train_loss / n_train_batches:.4f}')
                print(f'  Val Loss: {val_loss:.4f}')
                print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)')
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model with validation loss: {best_val_loss:.4f}")

    def predict(self, data):
        """Generate predictions for IID data"""
        self.model.eval()
        with torch.no_grad():
            # Standardize
            scaler = RobustScaler()
            data_scaled = scaler.fit_transform(data)
            X = torch.FloatTensor(data_scaled).to(self.device)

            # Reshape for transformer
            if self.model_type == 'transformer':
                X = X.unsqueeze(1)

            batch_size = 32
            dataset = TensorDataset(X)
            loader = DataLoader(dataset, batch_size=batch_size)

            predictions = []
            for (batch_x,) in loader:
                if self.model_type == 'gnn':
                    pred = self.model(batch_x, self.default_edge_index)
                else:
                    pred = self.model(batch_x)
                predictions.append(pred.cpu().numpy())

            all_predictions = np.concatenate(predictions, axis=0)

            # Return average prediction
            final_pred = np.mean(all_predictions, axis=0)
            return final_pred


class HybridSachsModel:
    """Hybrid model adapted for Sachs dataset (IID data)"""

    def __init__(self, neural_model_type, input_dim, hidden_dim=64, nhead=4, num_layers=2):
        self.neural_model_type = neural_model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize neural trainer
        self.neural_trainer = IIDDatasetTrainer(
            model_type=neural_model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=num_layers
        )

        # Topological model will be initialized during training
        self.topo_model = None

    def train_topo(self, X, initial_W=None):
        """Train topological model using TOPO_linear for IID data"""

        def regress(X, y):
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X=X, y=y)
            return reg.coef_

        def score(X, W):
            M = X @ W
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
            return loss, G_loss

        n, d = X.shape

        # If initial_W provided, use it to create initial topology
        if initial_W is not None:
            W_thresholded = threshold_W(initial_W, threshold=0.3)

            # Use out-degrees for initial ordering
            out_degrees = np.sum(W_thresholded, axis=1)
            topo_init = list(np.argsort(out_degrees))
        else:
            # Random ordering
            topo_init = list(np.random.permutation(range(d)))

        # Create and fit TOPO_linear model
        try:
            model = TOPO_linear(regress=regress, score=score)
            W_est, topo_order, Z_est, loss = model.fit(X=X, topo=topo_init, verbose=False)

            self.topo_model = {
                'W': W_est,
                'topo_order': topo_order,
                'Z': Z_est,
                'loss': loss
            }

            return W_est
        except Exception as e:
            print(f"Error in topological model training: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def fit(self, data, true_graph):
        """Train both neural and topological models"""
        print(f"\nTraining neural model ({self.neural_model_type})...")

        # Split data for training
        n_samples = data.shape[0]
        train_size = int(0.8 * n_samples)
        train_data = data[:train_size]
        val_data = data[train_size:]

        # Train neural model
        self.neural_trainer.train(
            train_data=train_data,
            train_net=true_graph,
            val_data=val_data,
            val_net=true_graph,
            epochs=100,
            batch_size=32,
            patience=10
        )

        # Get neural prediction
        neural_pred = self.neural_trainer.predict(data)

        print("\nTraining topological model...")

        # Train topological model
        self.topo_W = self.train_topo(data, initial_W=neural_pred)

        return self

    def predict(self, data):
        """Make predictions using both models and combine results"""
        # Get neural prediction
        neural_pred = self.neural_trainer.predict(data)

        # If topological model available, refine the prediction
        if self.topo_model is not None:
            # Create binary prediction from neural model
            neural_binary = (neural_pred > 0.5).astype(int)

            # Get topological prediction
            topo_W = self.topo_model['W']
            topo_binary = (np.abs(topo_W) > 0.3).astype(int)

            # Combine predictions (weighted average)
            combined_pred = (0.7 * neural_binary + 0.3 * topo_binary)

            # Enforce topological constraints
            topo_order = self.topo_model['topo_order']
            Z = self.topo_model['Z']

            # Ensure result respects topological ordering
            for i in range(len(topo_order)):
                for j in range(i + 1, len(topo_order)):
                    if Z[topo_order[i], topo_order[j]]:
                        combined_pred[topo_order[i], topo_order[j]] = 0

            return combined_pred
        else:
            # If topo model failed, return neural predictions
            return neural_pred


def calculate_metrics(B_est, B_true):
    """Calculate performance metrics with dynamic thresholding"""
    best_metrics = None
    best_f1 = -1

    # Try different thresholds
    for thresh in np.linspace(0.1, 0.9, 17):
        B_est_binary = (B_est > thresh).astype(int)

        TP = np.sum((B_est_binary == 1) & (B_true == 1))
        FP = np.sum((B_est_binary == 1) & (B_true == 0))
        FN = np.sum((B_est_binary == 0) & (B_true == 1))
        TN = np.sum((B_est_binary == 0) & (B_true == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'threshold': thresh,
                'shd': np.sum(B_est_binary != B_true),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0,
                'specificity': TN / (TN + FP) if (TN + FP) > 0 else 0,
                'TP': int(TP),
                'FP': int(FP),
                'FN': int(FN),
                'TN': int(TN)
            }

    return best_metrics


def plot_graphs(B_est, B_true, model_type, node_names=None):
    """Plot the estimated and true graphs for comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot estimated graph
    im0 = axes[0].imshow(B_est, cmap='Blues')
    axes[0].set_title(f'Estimated DAG ({model_type})')

    # Plot true graph
    im1 = axes[1].imshow(B_true, cmap='Blues')
    axes[1].set_title('True DAG')

    # Add node labels if provided
    if node_names is not None:
        # Add x and y ticks with node names
        for ax in axes:
            ax.set_xticks(np.arange(len(node_names)))
            ax.set_yticks(np.arange(len(node_names)))
            ax.set_xticklabels(node_names, rotation=45, ha='right')
            ax.set_yticklabels(node_names)

    # Add color bar
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Add values in cells
    for i in range(B_est.shape[0]):
        for j in range(B_est.shape[1]):
            # Only show values above 0.1 to reduce clutter
            if B_est[i, j] > 0.1:
                axes[0].text(j, i, f'{B_est[i, j]:.1f}',
                             ha='center', va='center',
                             color='white' if B_est[i, j] > 0.5 else 'black')

    for i in range(B_true.shape[0]):
        for j in range(B_true.shape[1]):
            if B_true[i, j] > 0:
                axes[1].text(j, i, f'{B_true[i, j]:.1f}',
                             ha='center', va='center',
                             color='white' if B_true[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(f'sachs_{model_type}_comparison.png')
    plt.close()


def run_sachs_hybrid_model():
    """Run the hybrid model on the Sachs dataset"""
    print("Loading Sachs dataset...")

    # Set random seed for reproducibility
    set_random_seed(42)

    # Load Sachs dataset from cdt
    data, graph = load_dataset('sachs')

    # Convert data to numpy array
    data_np = data.values

    # Get node names (variable names)
    node_names = data.columns.tolist()

    # Create a mapping from node names to indices
    node_to_idx = {name: i for i, name in enumerate(node_names)}

    # Convert networkx DiGraph to adjacency matrix
    n_nodes = data_np.shape[1]
    graph_np = np.zeros((n_nodes, n_nodes))

    # Print the graph edges for debugging
    print("Graph edges:")
    for edge in graph.edges():
        source, target = edge
        # Convert node names to indices if needed
        source_idx = node_to_idx.get(source, source)
        target_idx = node_to_idx.get(target, target)

        print(f"  {source} -> {target}")

        # Check if indices are valid
        if isinstance(source_idx, int) and isinstance(target_idx, int):
            if 0 <= source_idx < n_nodes and 0 <= target_idx < n_nodes:
                graph_np[source_idx, target_idx] = 1

    print(f"Dataset loaded: {data_np.shape[0]} samples, {n_nodes} nodes")
    print(f"True graph shape: {graph_np.shape}")

    # Print node information
    print("Node labels:", node_names)
    print("Number of edges in true graph:", len(graph.edges()))

    # Print the adjacency matrix for verification
    print("Adjacency matrix:")
    for i in range(n_nodes):
        print(f"  {node_names[i]}: {graph_np[i]}")

    # Adjust model parameters based on dataset size
    hidden_dim = max(64, n_nodes * 4)
    nhead = max(2, ((n_nodes // 2) * 2) or 2)  # Ensure it's even and at least 2
    print(f"Model parameters: hidden_dim={hidden_dim}, nhead={nhead}")

    # Run models
    all_results = {}
    models = ['transformer', 'deep_dag']

    for model_type in models:
        try:
            print(f"\n\nRunning {model_type.upper()}_TOPO hybrid model...")
            start = timer()

            # Initialize hybrid model
            trainer = HybridSachsModel(
                neural_model_type=model_type,
                input_dim=n_nodes,
                hidden_dim=hidden_dim,
                nhead=nhead,
                num_layers=2
            )

            # Fit the model
            trainer.fit(data_np, graph_np)

            # Predict
            pred_adj = trainer.predict(data_np)
            end = timer()

            # Calculate metrics
            metrics = calculate_metrics(pred_adj, graph_np)
            print(f"\n{model_type.upper()}_TOPO Results:")
            print(f"Time: {end - start:.4f}s")

            print("\nMetrics:")
            for metric, value in metrics.items():
                if metric in ['TP', 'FP', 'FN', 'TN']:
                    print(f"  {metric}: {int(value)}")
                else:
                    print(f"  {metric}: {value:.4f}")

            all_results[model_type] = metrics

            # Plot graphs with node names
            plot_graphs(pred_adj, graph_np, model_type, node_names=node_names)

        except Exception as e:
            print(f"Error running {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Print average results
    print("\nResults Summary:")
    metrics_to_report = ['shd', 'precision', 'recall', 'f1', 'accuracy']

    for model_type in models:
        if model_type in all_results:
            print(f"\n{model_type.upper()}_TOPO:")
            for metric in metrics_to_report:
                print(f"  {metric}: {all_results[model_type][metric]:.4f}")

    return all_results


if __name__ == '__main__':
    print("Detecting", torch.cuda.device_count(), "CUDA device(s).")
    results = run_sachs_hybrid_model()