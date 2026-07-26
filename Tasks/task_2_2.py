import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


CSV_PATH = str(Path(__file__).resolve().parent / "archive" / "kl.csv")

Principal_Components = [
    "Finishing", "ShortPassing", "Dribbling", "SprintSpeed",
    "Strength", "Stamina", "Interceptions", "StandingTackle", "Value"
]

N_COMPONENTS = 2


def money(v):
    if v is None:
        return np.nan
    if v[-1] == "M":
        return float(v[1:-1]) * (10**6)
    elif v[-1] == "K":
        return float(v[1:-1]) * (10**3)
    else:
        return float(v[1:])


class PCA:
    def __init__(self, path, cols, n_components=2):
        self.path = path
        self.cols = cols
        self.n_components = n_components

        self.df = None
        self.names = None
        self.positions = None
        self.mean = None
        self.std = None
        self.X_std = None
        self.cov_matrix = None
        self.eigenvalues_sorted = None
        self.eigenvectors_sorted = None
        self.explained_variance_ratio = None
        self.projection_matrix = None
        self.scores = None

    def run(self):
        # Loading the file and formatting the Value of each player
        df = pd.read_csv(self.path, encoding="cp1252")
        df["Value"] = df["Value"].apply(money)

        required_cols = self.cols + ["Position"]

        # Checks if the features are in the file
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column not found: {col}")

        cleany = df.dropna(subset= required_cols)
        X = cleany[self.cols].to_numpy()
      
        
        positions = cleany["Position"].to_numpy()
        names = cleany["Name"].to_numpy()

        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        X_std = (X - mean) / std # Z-score

        n = X_std.shape[0]
        X_cov = (X_std.T @ X_std) / (n - 1)

        eigenvalues, eigenvectors = np.linalg.eig(X_cov)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[sort_idx]
        eigenvectors_sorted = eigenvectors[:, sort_idx]

        explained_variance_ratio = eigenvalues_sorted / eigenvalues_sorted.sum()

        projection_matrix = eigenvectors_sorted[:, :self.n_components]
        scores = X_std @ projection_matrix

        # Store everything on the instance so main() can access it after run()
        self.df = df
        self.names = names
        self.positions = positions
        self.mean = mean
        self.std = std
        self.X_std = X_std
        self.cov_matrix = X_cov
        self.eigenvalues_sorted = eigenvalues_sorted
        self.eigenvectors_sorted = eigenvectors_sorted
        self.explained_variance_ratio = explained_variance_ratio
        self.projection_matrix = projection_matrix
        self.scores = scores

        return scores


def plot_pca_scatter(scores, positions, explained_variance_ratio, output_path):
  
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_positions = np.unique(positions)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_positions)))

    for pos, color in zip(unique_positions, colors):
        mask = positions == pos
        ax.scatter(scores[mask, 0], scores[mask, 1], label=pos,
                   color=color, s=10, alpha=0.75, edgecolors='none')

    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}% variance)")
    ax.set_title("PCA of FIFA 19 Player Attributes, Colored by Position")
    ax.legend(title="Position", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved to {output_path}")


def main():
    pca = PCA(path=CSV_PATH, cols=Principal_Components, n_components=N_COMPONENTS)
    pca.run()

    print()
    print("Eigenvalues (sorted, descending):")
    print(np.round(pca.eigenvalues_sorted, 4))
    print()
    print("Explained variance ratio per component:")
    print(np.round(pca.explained_variance_ratio, 4))
    print(f"First {N_COMPONENTS} components explain "
          f"{pca.explained_variance_ratio[:N_COMPONENTS].sum()*100:.2f}% of total variance.")
    print()
    print("Top-2 principal component loadings (which original features drive each PC):")
    loadings = pd.DataFrame(
        pca.projection_matrix,
        index=Principal_Components,
        columns=[f"PC{i+1}" for i in range(N_COMPONENTS)]
    )
    print(loadings.round(3))

    plot_pca_scatter(pca.scores, pca.positions, pca.explained_variance_ratio, "pca_scatter.png")

    return {
        "X_std": pca.X_std,
        "cov_matrix": pca.cov_matrix,
        "eigenvalues": pca.eigenvalues_sorted,
        "eigenvectors": pca.eigenvectors_sorted,
        "explained_variance_ratio": pca.explained_variance_ratio,
        "scores": pca.scores,
        "loadings": loadings,
    }


if __name__ == "__main__":
    results = main()