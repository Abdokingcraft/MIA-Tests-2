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
TARGET_NAME = "M. Salah"
TOP_N = 5


def money(v):
    if v is None:
        return np.nan
    if v[-1] == "M":
        return float(v[1:-1]) * (10**6)
    elif v[-1] == "K":
        return float(v[1:-1]) * (10**3)
    else:
        return float(v[1:])


# shared by PCA and ScoutingEngine, so it only lives in one place now.
# loads the csv, turns Value into a number, and drops any rows missing data
def load_players(path, cols, extra_col):
    df = pd.read_csv(path, encoding="cp1252")
    df["Value"] = df["Value"].apply(money)

    required_cols = cols + [extra_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    cleany = df.dropna(subset=required_cols).reset_index(drop=True)
    return df, cleany


# shared by PCA and ScoutingEngine too.
# standardizes each column: subtract the mean, divide by the std (Z-score)
def zscore(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    X_std = (X - mean) / std
    return X_std, mean, std


# same PCA class from task 2.2, not touching its logic, just reusing the
# helpers above instead of repeating the loading/z-score code
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
        df, cleany = load_players(self.path, self.cols, "Position")
        X = cleany[self.cols].to_numpy()

        # Getting the position and name of each player
        positions = cleany["Position"].to_numpy()
        names = cleany["Name"].to_numpy()

        X_std, mean, std = zscore(X)

        n = X_std.shape[0]
        # a matrix to relate between each feature
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


# task 2.3: finds the players most similar to a target player
class ScoutingEngine:
    def __init__(self, path, cols, target_name, n_top=TOP_N):
        self.path = path
        self.cols = cols
        self.target_name = target_name
        self.n_top = n_top

        self.names = None
        self.X = None
        self.X_std = None
        self.target_idx = None
        self.results_raw = None
        self.results_std = None
        self.final_shortlist = None

    def load(self):
        cleany = load_players(self.path, self.cols, "Name")[1]
        self.X = cleany[self.cols].to_numpy(dtype=float)
        self.names = cleany["Name"].to_numpy()
        self.X_std= zscore(self.X)[0]

    def find_target(self):
        matches = np.where(self.names == self.target_name)[0]
        if len(matches) == 0:
            raise ValueError(f"Could not find '{self.target_name}' in the Name column")
        self.target_idx = matches[0]




    # angle between 2 vectors, ignores magnitude
    def cosine_similarity(self, X, target):
        dot = X @ target
        norms = np.linalg.norm(X, axis=1) * np.linalg.norm(target)
        cos = dot / norms
        return np.maximum(cos, 0)

    # straight line distance between 2 players
    def euclidean_distance(self, X, target):
        diff = X - target
        return np.sqrt(np.sum(diff**2, axis=1))

    # sum of the absolute differences per feature
    def manhattan_distance(self, X, target):
        return np.sum(np.abs(X - target), axis=1)

    # correlates the shape of the stats, not the overall level
    def pearson_correlation(self, X, target):
        X_centered = X - X.mean(axis=1, keepdims=True)
        t_centered = target - target.mean()
        numerator = X_centered @ t_centered
        denominator = np.sqrt(np.sum(X_centered**2, axis=1)) * np.sqrt(np.sum(t_centered**2))
        return numerator / denominator





    # takes the scores of every player and returns the best n, skipping the target himself
    def top_n(self, scores, higher_is_better=True):
        if higher_is_better:
            order = np.argsort(scores)[::-1]
        else:
            order = np.argsort(scores)

        top = []
        for i in order:
            if i == self.target_idx:
                continue
            top.append((self.names[i], scores[i]))
            if len(top) == self.n_top:
                break
        return top

    def run_metrics(self, X):
        target = X[self.target_idx]
        results = {}
        results["cosine"] = self.top_n(self.cosine_similarity(X, target), higher_is_better=True)
        results["euclidean"] = self.top_n(self.euclidean_distance(X, target), higher_is_better=False)
        results["manhattan"] = self.top_n(self.manhattan_distance(X, target), higher_is_better=False)
        results["pearson"] = self.top_n(self.pearson_correlation(X, target), higher_is_better=True)
        return results

    def print_results(self, results):
        print("Cosine similarity (higher = more similar)")
        for name, val in results["cosine"]:
            print(f"{name} : {val:.4f}")
        print()

        print("Euclidean distance (lower = more similar)")
        for name, val in results["euclidean"]:
            print(f"{name} : {val:.4f}")
        print()

        print("Manhattan distance (lower = more similar)")
        for name, val in results["manhattan"]:
            print(f"{name} : {val:.4f}")
        print()

        print("Pearson correlation (higher = more similar)")
        for name, val in results["pearson"]:
            print(f"{name} : {val:.4f}")
        print()

    # counts how many of the 4 metrics agreed on each player
    def build_final_shortlist(self, results_std):
        votes = {}
        for metric_name, shortlist in results_std.items():
            for name, val in shortlist:
                if name in votes:
                    votes[name] += 1
                else:
                    votes[name] = 1

        sorted_names = sorted(votes, key=lambda n: votes[n], reverse=True)
        final = []
        for name in sorted_names[:self.n_top]:
            final.append((name, votes[name]))
        return final

    def run(self):
        self.load()
        self.find_target()
        self.results_raw = self.run_metrics(self.X)
        self.results_std = self.run_metrics(self.X_std)
        self.final_shortlist = self.build_final_shortlist(self.results_std)


def plot_pca_with_shortlist(pca, target_name, shortlist_names, output_path, label):
    target_idx = np.where(pca.names == target_name)[0][0]

    plt.figure(figsize=(10, 8))

    # everyone else in grey, in the background
    plt.scatter(pca.scores[:, 0], pca.scores[:, 1], color="lightgray", s=8, label="All players")

    # this method's shortlist, highlighted
    shortlist_idx = [i for i, name in enumerate(pca.names) if name in shortlist_names]
    plt.scatter(pca.scores[shortlist_idx, 0], pca.scores[shortlist_idx, 1],
                color="orange", s=70, edgecolors="black", label=label)

    # the target himself, drawn last so hes always on top
    plt.scatter(pca.scores[target_idx, 0], pca.scores[target_idx, 1],
                color="red", s=180, marker="*", edgecolors="black", label=f"{target_name} (target)")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA Projection: {target_name} and {label}")
    plt.legend()
    plt.axhline(0, color="gray")
    plt.axvline(0, color="gray")

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def main():
    engine = ScoutingEngine(path=CSV_PATH, cols=Principal_Components, target_name=TARGET_NAME, n_top=TOP_N)
    engine.run()

    print(f"Target found: {engine.names[engine.target_idx]}")
    print()

    print("Raw (unstandardized) features:")
    engine.print_results(engine.results_raw)

    print("Standardized features (z-scores):")
    engine.print_results(engine.results_std)

    print("Final shortlist (based on the standardized results):")
    for name, votes in engine.final_shortlist:
        print(f"{name} : appeared in {votes}/4 metrics")

    # reuse the PCA class, unchanged, to plot the target and each shortlist
    pca = PCA(path=CSV_PATH, cols=Principal_Components, n_components=N_COMPONENTS)
    pca.run()

    # one plot per method, so you can see how each one picks different players
    for metric_name, shortlist in engine.results_std.items():
        names = [name for name, val in shortlist]
        plot_pca_with_shortlist(pca, TARGET_NAME, names,
                                 f"pca_{metric_name}.png", label=metric_name.capitalize())

    # and one more plot for the combined final shortlist
    shortlist_names = [name for name, votes in engine.final_shortlist]
    plot_pca_with_shortlist(pca, TARGET_NAME, shortlist_names,
                             "pca_final_shortlist.png", label="Final shortlist")


if __name__ == "__main__":
    main()