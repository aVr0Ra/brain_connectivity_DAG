import numpy as np
from cdt.data import load_dataset
import networkx as nx
from topo_linear import TOPO_linear, threshold_W
from sklearn.linear_model import LinearRegression
import utils
from timeit import default_timer as timer
from castle import MetricsDAG

if __name__ == '__main__':
    # Load Sachs dataset
    data, graph = load_dataset('sachs')
    print('\nLoaded Sachs dataset')

    # Convert dataset to numpy arrays
    X = data.values

    # Convert NetworkX DiGraph to numpy adjacency matrix
    B_true = nx.to_numpy_array(graph).astype(int)

    print("Data shape:", X.shape)
    print("True graph shape:", B_true.shape)

    # Set random seed
    rd_int = int(np.random.randint(10000, size=1)[0])
    print(f"Random seed: {rd_int}")
    utils.set_random_seed(rd_int)

    n, d = X.shape
    verbose = False


    # Linear model
    def regress(X, y):
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X=X, y=y)
        return reg.coef_


    def score(X, W):
        M = X @ W
        R = X - M
        loss = 0.5 / X.shape[0] * np.sum(R ** 2)
        G_loss = -1.0 / X.shape[0] * (X.T @ R)
        return loss, G_loss


    model_linear = TOPO_linear(regress=regress, score=score)
    topo_init = list(np.random.permutation(range(d)))

    print("Starting model fitting...")
    start = timer()
    W_linear, topo_linear_est, Z_linear, loss_linear = model_linear.fit(X=X, topo=topo_init, verbose=verbose)
    end = timer()
    print(f"Linear model training time: {end - start:.4f}s")

    # Evaluate results
    B_est = (threshold_W(W=W_linear) != 0).astype(int)
    print("Estimated adjacency matrix:")
    print(B_est)
    print("True adjacency matrix:")
    print(B_true)

    # Calculate metrics
    met = MetricsDAG(B_est, B_true)
    metrics = met.metrics

    print("\nPerformance Metrics:")
    print(f"SHD: {metrics['shd']}")
    print(f"F1 Score: {metrics['F1']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")

    # Optional: Run multiple trials with different seeds
    if False:  # Set to True to run multiple trials
        n_trials = 10
        results = {
            'SHD': [],
            'F1': [],
            'precision': [],
            'recall': []
        }

        for trial in range(n_trials):
            seed = int(np.random.randint(10000, size=1)[0])
            utils.set_random_seed(seed)
            print(f"\nTrial {trial + 1}/{n_trials}, seed: {seed}")

            topo_init = list(np.random.permutation(range(d)))
            W_linear, _, _, _ = model_linear.fit(X=X, topo=topo_init, verbose=False)
            B_est = (threshold_W(W=W_linear) != 0).astype(int)

            met = MetricsDAG(B_est, B_true)
            m = met.metrics

            for metric in results:
                if not np.isnan(m[metric if metric != 'SHD' else 'shd']):
                    results[metric].append(m[metric if metric != 'SHD' else 'shd'])

        # Print summary statistics
        print("\nSummary of multiple trials:")
        for metric in results:
            if results[metric]:
                print(f"{metric} mean: {np.mean(results[metric])}")
                print(f"{metric} variance: {np.var(results[metric])}")