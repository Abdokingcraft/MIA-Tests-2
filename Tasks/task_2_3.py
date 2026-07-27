import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


CSV_PATH = str(Path(__file__).resolve().parent / "archive" / "kl.csv")

# same attribute subset as task 2.2, so the vector representation of a player stays consistent
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


# same PCA class as task 2.2, reused as-is for the bonus visualization
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
        df = pd.read_csv(self.path, encoding="cp1252")
        df["Value"] = df["Value"].apply(money)

        required_cols = self.cols + ["Position"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column not found: {col}")

        cleany = df.dropna(subset=required_cols)
        X = cleany[self.cols].to_numpy()

        positions = cleany["Position"].to_numpy()
        names = cleany["Name"].to_numpy()

        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        X_std = (X - mean) / std

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


# ---------------------------------------------------------------------------
# task 2.3: the scouting engine. same "load once, store on self" pattern as
# the PCA class above, but instead of projecting the data it ranks every
# player by similarity to a single target player, using 4 metrics built
# from scratch (all vectorized: target vs the whole pool at once, no loop)
# ---------------------------------------------------------------------------
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
        self.picked_by = None

    def load(self):
        df = pd.read_csv(self.path, encoding="cp1252")
        df["Value"] = df["Value"].apply(money)

        # Name is required here (unlike the PCA class) since we need it to
        # look up the target and print out shortlists
        required_cols = self.cols + ["Name"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column not found: {col}")

        cleany = df.dropna(subset=required_cols).reset_index(drop=True)

        self.X = cleany[self.cols].to_numpy(dtype=float)
        self.names = cleany["Name"].to_numpy()

        mean = self.X.mean(axis=0)
        std = self.X.std(axis=0, ddof=1)
        self.X_std = (self.X - mean) / std  # Z-score, same formula as the PCA class

    def find_target(self):
        matches = np.where(self.names == self.target_name)[0]
        if len(matches) == 0:
            lowered = np.char.lower(self.names.astype(str))
            matches = np.where(lowered == self.target_name.lower())[0]
        if len(matches) == 0:
            raise ValueError(f"Could not find '{self.target_name}' in the Name column")
        if len(matches) > 1:
            print(f"warning: multiple rows matched '{self.target_name}', using the first one")
        self.target_idx = matches[0]

    # --- similarity metrics, all from scratch, all vectorized (target vs every player at once) ---

    # cosine similarity: measures the ANGLE between 2 vectors, ignores magnitude.
    # a weaker/stronger version of the same "shape" of player can still score close to 1.
    @staticmethod
    def cosine_similarity(X, target):
        dot = X @ target
        norms = np.linalg.norm(X, axis=1) * np.linalg.norm(target)
        return dot / norms

    # euclidean distance: straight-line distance in feature space.
    # very sensitive to raw scale - a huge-scale feature (like Value) can dominate it.
    @staticmethod
    def euclidean_distance(X, target):
        diff = X - target
        return np.sqrt(np.sum(diff**2, axis=1))

    # manhattan distance: sum of absolute per-feature differences.
    # still scale sensitive, but less dominated by 1 single huge outlier feature than euclidean.
    @staticmethod
    def manhattan_distance(X, target):
        return np.sum(np.abs(X - target), axis=1)

    # pearson correlation: correlates the SHAPE of each players own stat profile
    # (centered around their own row mean) against the targets shape - ignores overall level.
    @staticmethod
    def pearson_correlation(X, target):
        X_centered = X - X.mean(axis=1, keepdims=True)
        t_centered = target - target.mean()
        numerator = X_centered @ t_centered
        denominator = np.sqrt(np.sum(X_centered**2, axis=1)) * np.sqrt(np.sum(t_centered**2))
        return numerator / denominator

    # grabs the top N most similar players from a score array, excluding the target himself
    def top_n(self, scores, higher_is_better=True):
        order = np.argsort(scores)[::-1] if higher_is_better else np.argsort(scores)
        order = order[order != self.target_idx]
        top_idx = order[:self.n_top]
        return list(zip(self.names[top_idx], scores[top_idx]))

    # runs all 4 metrics against the target on whichever feature matrix its given (raw or standardized)
    def run_metrics(self, X):
        target = X[self.target_idx]
        results = {}
        results["cosine"] = self.top_n(self.cosine_similarity(X, target), higher_is_better=True)
        results["euclidean"] = self.top_n(self.euclidean_distance(X, target), higher_is_better=False)
        results["manhattan"] = self.top_n(self.manhattan_distance(X, target), higher_is_better=False)
        results["pearson"] = self.top_n(self.pearson_correlation(X, target), higher_is_better=True)
        return results

    def print_results(self, label, results):
        titles = {
            "cosine": "COSINE SIMILARITY (top 5, higher = more similar)",
            "euclidean": "EUCLIDEAN DISTANCE (top 5, lower = more similar)",
            "manhattan": "MANHATTAN DISTANCE (top 5, lower = more similar)",
            "pearson": "PEARSON CORRELATION (top 5, higher = more similar)",
        }
        print(f"\n=== {label} ===")
        for key, title in titles.items():
            print(title)
            for i, (name, val) in enumerate(results[key], start=1):
                print(f"{i}. {name:<20} : {val:.4f}")
            print("-----------------------")

    # builds 1 final shortlist out of the 4 standardized metric shortlists: counts how many
    # metrics agreed on each player, and remembers WHICH metrics picked them (so an odd
    # 1-metric-only pick can be traced back to explain why it looks like an outlier later)
    def build_final_shortlist(self):
        counter = Counter()
        picked_by = {}
        for metric_name, shortlist in self.results_std.items():
            for name, _ in shortlist:
                counter[name] += 1
                picked_by.setdefault(name, []).append(metric_name)
        self.final_shortlist = counter.most_common(self.n_top)
        self.picked_by = picked_by

    def run(self):
        self.load()
        self.find_target()
        self.results_raw = self.run_metrics(self.X)
        self.results_std = self.run_metrics(self.X_std)
        self.build_final_shortlist()


def plot_pca_with_shortlist(pca, target_name, shortlist_names, output_path):
    target_idx = np.where(pca.names == target_name)[0][0]

    fig, ax = plt.subplots(figsize=(10, 8))

    # everyone else, in grey, in the background
    ax.scatter(pca.scores[:, 0], pca.scores[:, 1], color="lightgray", s=8, alpha=0.5, label="All players")

    # the final shortlist, highlighted
    shortlist_idx = [i for i, name in enumerate(pca.names) if name in shortlist_names]
    ax.scatter(pca.scores[shortlist_idx, 0], pca.scores[shortlist_idx, 1],
               color="orange", s=70, edgecolors="black", label="Final shortlist")

    # the target himself, drawn last so hes always on top and visible
    ax.scatter(pca.scores[target_idx, 0], pca.scores[target_idx, 1],
               color="red", s=180, marker="*", edgecolors="black", label=f"{target_name} (target)")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA Projection: {target_name} and the Final Shortlist")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"PCA shortlist scatter saved to {output_path}")


def main():
    engine = ScoutingEngine(path=CSV_PATH, cols=Principal_Components, target_name=TARGET_NAME, n_top=TOP_N)
    engine.run()

    print(f"\ntarget found: {engine.names[engine.target_idx]} (row {engine.target_idx})")

    print("\n\n################ RAW / UNSTANDARDIZED FEATURES ################")
    engine.print_results("RAW (unstandardized)", engine.results_raw)

    print("\n\n################ STANDARDIZED FEATURES (z-scores) ################")
    engine.print_results("STANDARDIZED (z-scores)", engine.results_std)

    print("\n\n################ FINAL RECOMMENDED SHORTLIST ################")
    print("built from the standardized results only - raw distance metrics get dragged")
    print("around by Value's huge scale (millions vs 0-100 stats), so raw shortlists")
    print("mostly just reflect similar transfer value, not similar playing profile.")
    for i, (name, votes) in enumerate(engine.final_shortlist, start=1):
        metrics_str = ", ".join(engine.picked_by[name])
        print(f"{i}. {name:<20} : appeared in {votes}/4 standardized metric shortlists ({metrics_str})")
    print("-----------------------")
    print("note: a player picked ONLY by pearson tends to sit far from Salah in PCA space -")
    print("pearson matches the SHAPE of a players stats, not their overall level, so a much")
    print("weaker/stronger player with the same relative pattern can still score high on it.")

    # bonus: reuse the PCA class from task 2.2, unchanged, to project the whole pool
    # and highlight the target + final shortlist on top of it
    pca = PCA(path=CSV_PATH, cols=Principal_Components, n_components=N_COMPONENTS)
    pca.run()

    shortlist_names = [name for name, _ in engine.final_shortlist]
    plot_pca_with_shortlist(pca, TARGET_NAME, shortlist_names, "pca_salah_shortlist.png")

    return {
        "engine": engine,
        "pca": pca,
    }


if __name__ == "__main__":
    results = main()