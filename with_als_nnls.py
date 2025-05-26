import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF as SKNMF
import random


# -----------------------------------------------------------------------------
# Funcția pentru Non-Negative Least Squares (NNLS)
# -----------------------------------------------------------------------------
def manual_nnls(A, b, max_iter=200, tol=1e-5, learning_rate=None):
    x = np.zeros(A.shape[1])

    if learning_rate is None:
        try:
            if A.size > 0 and np.any(
                np.sum(A**2, axis=0) > 1e-9
            ):  # Avoid division by zero
                # A more robust learning rate
                lipschitz_constant_approx = np.linalg.norm(A.T @ A, ord=2)
                if lipschitz_constant_approx > 1e-9:
                    learning_rate = 1.0 / lipschitz_constant_approx
                else:
                    learning_rate = 1e-3  # Fallback
            else:
                learning_rate = 1e-3  # Fallback if A is empty or has zero columns
        except Exception:
            learning_rate = 1e-3

    if b.ndim > 1:
        b = b.flatten()

    for _ in range(max_iter):
        x_old = np.copy(x)
        grad = A.T @ (A @ x - b)
        x = x - learning_rate * grad
        x[x < 0] = 0

        if np.linalg.norm(x - x_old) < tol:
            break

    return x


# -----------------------------------------------------------------------------
# ALS with NNLS
# -----------------------------------------------------------------------------
def als_manual_nnls(
    V,
    r,
    max_iter_als=50,
    tol_als=1e-4,
    max_iter_nnls=50,
    tol_nnls=1e-5,
    lr_nnls=None,
    random_state=None,
):
    if random_state is not None:
        np.random.seed(random_state)

    m, n = V.shape

    W = np.abs(np.random.rand(m, r))
    H = np.abs(np.random.rand(r, n))

    print(f"Running ALS with Manual NNLS (r={r})...")

    prev_error_als = np.inf
    for iteration_als in range(max_iter_als):
        # 1. Fix H, optimize W
        if H.shape[0] > 0:  # Only if r > 0
            for i in range(m):
                if H.T.shape[1] > 0:  # Ensure H.T has columns
                    w_i_T = manual_nnls(
                        H.T,
                        V[i, :].T,
                        max_iter=max_iter_nnls,
                        tol=tol_nnls,
                        learning_rate=lr_nnls,
                    )
                    W[i, :] = w_i_T

        # 2. Fix W, optimize H
        if W.shape[1] > 0:  # Only if r > 0
            for j in range(n):
                if W.shape[1] > 0:  # Ensure W has columns
                    H[:, j] = manual_nnls(
                        W,
                        V[:, j],
                        max_iter=max_iter_nnls,
                        tol=tol_nnls,
                        learning_rate=lr_nnls,
                    )

        current_error_als = np.linalg.norm(V - W @ H, "fro")
        if iteration_als % 5 == 0 or iteration_als == max_iter_als - 1:
            print(
                f"ALS-NNLS Iteration {iteration_als}: Frobenius error = {current_error_als:.4f}"
            )

        if abs(prev_error_als - current_error_als) < tol_als:
            print(f"ALS-NNLS converged at iteration {iteration_als}.")
            break
        prev_error_als = current_error_als

    if (
        iteration_als == max_iter_als - 1
        and abs(prev_error_als - current_error_als) >= tol_als
    ):
        print(
            f"ALS-NNLS reached max_iter ({max_iter_als}) without full convergence based on tol_als."
        )

    print("ALS-NNLS finished.")
    return W, H


# -----------------------------------------------------------------------------
# Multiplicative NMF
# -----------------------------------------------------------------------------
def nmf_multiplicative(V, r, max_iter=100, tol=1e-4):
    m, n = V.shape
    if r == 0:
        return np.zeros((m, 0)), np.zeros((0, n))
    np.random.seed(42)
    W = np.abs(np.random.randn(m, r))
    H = np.abs(np.random.randn(r, n))

    for it in range(max_iter):
        # Update H
        WH = W @ H
        H_numerator = W.T @ V
        H_denominator = W.T @ WH + 1e-10
        H *= H_numerator / H_denominator

        # Update W
        WH = W @ H  # Recalculate WH after H update
        W_numerator = V @ H.T
        W_denominator = WH @ H.T + 1e-10
        W *= W_numerator / W_denominator

        if it > 0:
            error = np.linalg.norm(V - W @ H, "fro")
            if it % 10 == 0:
                print(f"Multiplicative NMF Iteration {it}: error = {error:.4f}")
            if error < tol:
                print(f"Multiplicative NMF converged at iteration {it}")
                break
    return W, H


# -----------------------------------------------------------------------------
# NMF din sklearn
# -----------------------------------------------------------------------------
def nmf_sklearn(V, r, max_iter=100, tol=1e-4):
    if r == 0:
        return np.zeros((V.shape[0], 0)), np.zeros((0, V.shape[1]))
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


# -----------------------------------------------------------------------------
# Recommendation function
# -----------------------------------------------------------------------------
def get_recommendations(idx, similarity_matrix, method_name, df_source, top_n=5):
    similarities = cosine_similarity(
        similarity_matrix[idx : idx + 1], similarity_matrix
    )[0]
    similarities[idx] = -np.inf  # Exclude self

    top_indices = similarities.argsort()[::-1][:top_n]

    print(
        f"\nReference: {df_source.iloc[idx]['name']} - {df_source.iloc[idx]['artist']}"
    )
    print(f"Recommendations ({method_name}):")
    for i in top_indices:
        print(f"- {df_source.iloc[i]['name']} - {df_source.iloc[i]['artist']}")


# -----------------------------------------------------------------------------
# Load and preprocess data
# -----------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv("music_info.csv")

print("Extracting unique tags...")
all_tags = set()
for tags_str in df["tags"].dropna():
    all_tags.update(tag.strip() for tag in tags_str.split(","))
all_tags = sorted(list(all_tags))  # Ensure it is a sorted list

print(f"Found {len(all_tags)} unique tags.")

print("Creating track x tag matrix...")
track_tag_matrix = np.zeros((len(df), len(all_tags)), dtype=np.float32)
tag_to_idx = {tag: idx for idx, tag in enumerate(all_tags)}

for i, tags_str in enumerate(df["tags"].fillna("")):
    for tag in tags_str.split(","):
        tag = tag.strip()
        if tag in tag_to_idx:
            track_tag_matrix[i, tag_to_idx[tag]] = 1
print(f"Track x tag matrix shape: {track_tag_matrix.shape}")


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    r_factor = 10

    als_iters = 30
    als_tolerance = 1e-4
    nnls_iters = 50
    nnls_tolerance = 1e-5
    nnls_lr = 1e-4  # Adjust if NNLS manual does not converge well

    print("\n--- Starting Multiplicative NMF ---")
    W_mult, H_mult = nmf_multiplicative(track_tag_matrix, r_factor)

    print("\n--- Starting sklearn NMF ---")
    W_sklearn, H_sklearn = nmf_sklearn(track_tag_matrix, r_factor)

    print("\n--- Starting ALS with Manual NNLS ---")
    W_als_manual, H_als_manual = als_manual_nnls(
        track_tag_matrix,
        r_factor,
        max_iter_als=als_iters,
        tol_als=als_tolerance,
        max_iter_nnls=nnls_iters,
        tol_nnls=nnls_tolerance,
        lr_nnls=nnls_lr,
        random_state=42,
    )

    print("\n--- Starting Recommendation Loop ---")
    while True:
        idx = random.randint(0, len(df) - 1)

        get_recommendations(
            idx, track_tag_matrix, "Tag Similarity (Cosine)", df_source=df
        )
        get_recommendations(idx, W_mult, "NMF Multiplicative", df_source=df)
        get_recommendations(idx, W_sklearn, "NMF (sklearn)", df_source=df)
        get_recommendations(idx, W_als_manual, "ALS (Manual NNLS)", df_source=df)

        if input("\nPress Enter to continue (or 'q' to quit): ").lower() == "q":
            break


if __name__ == "__main__":
    main()
