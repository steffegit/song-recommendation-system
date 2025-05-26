import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF as SKNMF
import random

# Load data and build track × tag matrix
df = pd.read_csv("music_info.csv")

# Extract unique tags
all_tags = set()
for tags in df["tags"].dropna():
    all_tags.update(tag.strip() for tag in tags.split(","))
all_tags = sorted(all_tags)

# Create binary track × tag matrix
track_tag_matrix = np.zeros((len(df), len(all_tags)), dtype=np.float32)
tag_to_idx = {tag: idx for idx, tag in enumerate(all_tags)}

for i, tags in enumerate(df["tags"].fillna("")):
    for tag in tags.split(","):
        tag = tag.strip()
        if tag in tag_to_idx:
            track_tag_matrix[i, tag_to_idx[tag]] = 1


# Simplified NMF implementation
def nmf_multiplicative(V, r, max_iter=100, tol=1e-4):
    m, n = V.shape
    np.random.seed(42)
    W = np.abs(np.random.randn(m, r))
    H = np.abs(np.random.randn(r, n))

    for it in range(max_iter):
        # Update H
        WH = W @ H
        H *= (W.T @ V) / (W.T @ WH + 1e-10)

        # Update W
        WH = W @ H
        W *= (V @ H.T) / (WH @ H.T + 1e-10)

        # Check convergence
        if it > 0:
            error = np.linalg.norm(V - W @ H, "fro")
            if it % 10 == 0:
                print(f"Multiplicative NMF Iteration {it}: error = {error:.4f}")
            if error < tol:
                print(f"Multiplicative NMF converged at iteration {it}")
                break

    return W, H


def nmf_sklearn(V, r, max_iter=100, tol=1e-4):
    model = SKNMF(
        n_components=r,
        init="random",
        random_state=42,
        max_iter=max_iter,
        tol=tol,
        solver="cd",
    )
    W = model.fit_transform(V)
    H = model.components_
    print(f"sklearn NMF: final error = {np.linalg.norm(V - W @ H, 'fro'):.4f}")
    return W, H


def get_recommendations(idx, similarity_matrix, method_name, top_n=5):
    """Generic recommendation function"""
    similarities = cosine_similarity(
        similarity_matrix[idx : idx + 1], similarity_matrix
    )[0]
    top_indices = similarities.argsort()[::-1][1 : top_n + 1]  # Exclude self

    print(f"\nReference: {df.iloc[idx]['name']} - {df.iloc[idx]['artist']}")
    print(f"Recommendations ({method_name}):")
    for i in top_indices:
        print(f"- {df.iloc[i]['name']} - {df.iloc[i]['artist']}")


def main():
    r = 10

    # Multiplicative NMF
    W, H = nmf_multiplicative(track_tag_matrix, r)
    print("Multiplicative NMF error:", np.linalg.norm(track_tag_matrix - W @ H, "fro"))

    # sklearn NMF
    W_sklearn, H_sklearn = nmf_sklearn(track_tag_matrix, r)
    print(
        "sklearn NMF error:",
        np.linalg.norm(track_tag_matrix - W_sklearn @ H_sklearn, "fro"),
    )

    # Main recommendation loop
    while True:
        idx = random.randint(0, len(df) - 1)

        get_recommendations(idx, track_tag_matrix, "tag similarity")
        get_recommendations(idx, W, "NMF multiplicative")
        get_recommendations(idx, W_sklearn, "NMF (sklearn)")

        if input("\nPress Enter to continue (or 'q' to quit): ").lower() == "q":
            break


if __name__ == "__main__":
    main()
