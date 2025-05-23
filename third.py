import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import random
from MusicRecommender import MusicRecommender

# Load data and build track × tag matrix
df = pd.read_csv("music_info.csv")

# Extract unique tags with frequency filtering
tag_counts = {}
for tags in df["tags"].dropna():
    for tag in tags.split(","):
        tag = tag.strip().lower()  # Normalize case
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

# Keep only tags that appear at least 3 times (filter noise)
min_frequency = 3
all_tags = sorted([tag for tag, count in tag_counts.items() if count >= min_frequency])
print(f"Using {len(all_tags)} tags (filtered from {len(tag_counts)} total)")

# Create TF-IDF weighted track × tag matrix instead of binary
track_tag_matrix = np.zeros((len(df), len(all_tags)), dtype=np.float32)
tag_to_idx = {tag: idx for idx, tag in enumerate(all_tags)}

# Build matrix with term frequencies
for i, tags in enumerate(df["tags"].fillna("")):
    tag_list = [tag.strip().lower() for tag in tags.split(",")]
    for tag in tag_list:
        if tag in tag_to_idx:
            track_tag_matrix[i, tag_to_idx[tag]] += 1


# Apply TF-IDF weighting
def apply_tfidf(matrix):
    # TF: already have term frequencies
    # IDF: log(N/df) where N=total docs, df=docs containing term
    N = matrix.shape[0]
    df_per_tag = np.sum(matrix > 0, axis=0)  # Document frequency per tag
    idf = np.log(N / (df_per_tag + 1))  # +1 to avoid division by zero
    return matrix * idf


# Enhanced NMF with better initialization
def nmf_multiplicative(V, r, max_iter=200, tol=1e-6):
    m, n = V.shape
    np.random.seed(0)

    # Better initialization using SVD
    U, s, Vt = np.linalg.svd(V, full_matrices=False)
    W = np.abs(U[:, :r] * np.sqrt(s[:r]))
    H = np.abs(np.sqrt(s[:r]).reshape(-1, 1) * Vt[:r, :])

    prev_error = float("inf")
    for it in range(max_iter):
        # Update H
        WH = W @ H
        H *= (W.T @ V) / (W.T @ WH + 1e-10)

        # Update W
        WH = W @ H
        W *= (V @ H.T) / (WH @ H.T + 1e-10)

        # Check convergence
        if it % 20 == 0:
            error = np.linalg.norm(V - W @ H, "fro")
            print(f"Iteration {it}: error = {error:.6f}")
            if abs(prev_error - error) < tol:
                print(f"Converged at iteration {it}")
                break
            prev_error = error

    return W, H


track_tag_matrix = apply_tfidf(track_tag_matrix)

# Run NMF with more factors for better representation
r = 20  # Increased from 10
W, H = nmf_multiplicative(track_tag_matrix, r)

# Normalize embeddings for better similarity computation
W_normalized = normalize(W, norm="l2", axis=1)
track_tag_normalized = normalize(track_tag_matrix, norm="l2", axis=1)


# Create recommender instance
recommender = MusicRecommender(
    df, track_tag_matrix, W, track_tag_normalized, W_normalized
)


# Interactive recommendation loop
def run_recommendations():
    while True:
        print("\n" + "=" * 60)
        idx = random.randint(0, len(df) - 1)

        # Try all three methods
        tag_results = recommender.recommend_by_tags(idx)
        nmf_results = recommender.recommend_by_nmf(idx)
        hybrid_results = recommender.recommend_hybrid(idx)

        recommender.print_recommendations(tag_results)
        recommender.print_recommendations(nmf_results)
        recommender.print_recommendations(hybrid_results)

        choice = input(
            "\nOptions: [Enter] = Next song, [s] = Search specific song, [q] = Quit: "
        ).lower()

        if choice == "q":
            break
        elif choice == "s":
            search_term = input("Enter song or artist name: ")
            matches = df[
                df["name"].str.contains(search_term, case=False, na=False)
                | df["artist"].str.contains(search_term, case=False, na=False)
            ]
            if not matches.empty:
                print(f"\nFound {len(matches)} matches:")
                for i, (_, row) in enumerate(matches.head(5).iterrows()):
                    print(f"{i + 1}. {row['name']} - {row['artist']}")
                try:
                    choice_idx = int(input("Select number (1-5): ")) - 1
                    if 0 <= choice_idx < len(matches):
                        idx = matches.iloc[choice_idx].name
                        # Show recommendations for selected song
                        results = recommender.recommend_hybrid(idx)
                        recommender.print_recommendations(results)
                except (ValueError, IndexError):
                    print("Invalid selection")
            else:
                print("No matches found")


if __name__ == "__main__":
    run_recommendations()
