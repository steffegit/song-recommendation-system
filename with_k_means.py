import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF as SKNMF

# Nu mai importăm KMeans din sklearn dacă facem manual
import random

# ... (restul codului de încărcare și preprocesare date rămâne la fel) ...
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
        WH = W @ H
        H *= (W.T @ V) / (W.T @ WH + 1e-10)
        WH = W @ H
        W *= (V @ H.T) / (WH @ H.T + 1e-10)
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


def manual_kmeans(V, k, max_iter=100, tol=1e-4, random_state=None):
    """
    Implementare manuală a algoritmului K-Means.

    Args:
        V (np.array): Matricea de date (melodii x caracteristici).
        k (int): Numărul de clustere.
        max_iter (int): Numărul maxim de iterații.
        tol (float): Toleranța pentru convergență.
                      Dacă suma pătratelor distanțelor de mișcare a centroizilor
                      este sub această valoare, algoritmul se oprește.
        random_state (int, optional): Seed pentru generatorul de numere aleatorii
                                      pentru reproductibilitate.

    Returns:
        tuple: (labels, centroids)
            labels (np.array): Etichetele clusterelor pentru fiecare punct de date.
            centroids (np.array): Coordonatele finale ale centroizilor.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = V.shape

    # 1. Inițializare: Alege k puncte de date aleatorii ca centroizi inițiali
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = V[random_indices]

    print(f"\nRunning Manual K-Means clustering with k={k}...")

    for iteration in range(max_iter):
        # Stochează centroizii vechi pentru verificarea convergenței
        old_centroids = np.copy(centroids)

        # 2. Atribuire: Atribuie fiecare punct celui mai apropiat centroid
        distances_sq = np.zeros((n_samples, k))
        for i in range(k):
            # Distanța pătrată Euclideană
            distances_sq[:, i] = np.sum((V - centroids[i]) ** 2, axis=1)

        labels = np.argmin(distances_sq, axis=1)

        # 3. Actualizare: Recalculează centroizii
        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            points_in_cluster = V[labels == i]
            if len(points_in_cluster) > 0:
                new_centroids[i] = np.mean(points_in_cluster, axis=0)
            else:
                # Gestionarea clusterelor goale: re-inițializează centroidul
                # la un punct aleatoriu (o strategie simplă)
                new_centroids[i] = V[np.random.choice(n_samples)]

        centroids = new_centroids

        # Verificarea convergenței
        # Suma pătratelor distanțelor de mișcare a centroizilor
        centroid_shift_sq = np.sum((centroids - old_centroids) ** 2)

        if iteration % 10 == 0 or iteration == max_iter - 1:
            print(
                f"Manual K-Means Iteration {iteration}: Centroid shift sq = {centroid_shift_sq:.6f}"
            )

        if centroid_shift_sq < tol:
            print(f"Manual K-Means converged at iteration {iteration}.")
            break

    if iteration == max_iter - 1 and centroid_shift_sq >= tol:
        print(
            f"Manual K-Means reached max_iter ({max_iter}) without full convergence based on tol."
        )

    print("Manual K-Means clustering finished.")
    return labels, centroids


# Funcție de recomandare generală
def get_recommendations(idx, similarity_matrix, method_name, top_n=5):
    similarities = cosine_similarity(
        similarity_matrix[idx : idx + 1], similarity_matrix
    )[0]
    similarities[idx] = -np.inf
    top_indices = similarities.argsort()[::-1][:top_n]

    print(f"\nReference: {df.iloc[idx]['name']} - {df.iloc[idx]['artist']}")
    print(f"Recommendations ({method_name}):")
    for i in top_indices:
        print(f"- {df.iloc[i]['name']} - {df.iloc[i]['artist']}")


# Funcție de recomandare specifică pentru K-Means
def get_recommendations_kmeans(
    ref_idx, labels, all_song_features, df_songs, method_name, top_n=5
):
    ref_song_name = df_songs.iloc[ref_idx]["name"]
    ref_song_artist = df_songs.iloc[ref_idx]["artist"]
    ref_label = labels[ref_idx]

    cluster_member_indices = [
        i for i, label in enumerate(labels) if label == ref_label and i != ref_idx
    ]

    if not cluster_member_indices:
        print(f"\nReference: {ref_song_name} - {ref_song_artist}")
        print(f"Recommendations ({method_name} - Cluster {ref_label}):")
        print(f"- No other songs found in the same K-Means cluster.")
        return

    ref_song_vector = all_song_features[ref_idx : ref_idx + 1]
    cluster_songs_vectors = all_song_features[cluster_member_indices]

    similarities_in_cluster = cosine_similarity(ref_song_vector, cluster_songs_vectors)[
        0
    ]

    sorted_cluster_indices = np.argsort(similarities_in_cluster)[::-1]
    top_original_indices_in_cluster = [
        cluster_member_indices[i] for i in sorted_cluster_indices[:top_n]
    ]

    print(f"\nReference: {ref_song_name} - {ref_song_artist}")
    print(f"Recommendations ({method_name} - Cluster {ref_label}):")
    for i in top_original_indices_in_cluster:
        print(f"- {df_songs.iloc[i]['name']} - {df_songs.iloc[i]['artist']}")


def main():
    r_nmf = 10
    k_kmeans = 20  # Alege un număr de clustere
    kmeans_max_iter = (
        50  # Număr mai mic de iterații pentru K-Means manual, poate fi lent
    )
    kmeans_tol = 1e-3  # Toleranță pentru K-Means manual

    # --- NMF Multiplicativ ---
    W_mult, H_mult = nmf_multiplicative(track_tag_matrix, r_nmf)
    print(
        "Multiplicative NMF error:",
        np.linalg.norm(track_tag_matrix - W_mult @ H_mult, "fro"),
    )

    # --- NMF sklearn ---
    W_sklearn, H_sklearn = nmf_sklearn(track_tag_matrix, r_nmf)
    print(
        "sklearn NMF error:",
        np.linalg.norm(track_tag_matrix - W_sklearn @ H_sklearn, "fro"),
    )

    # --- K-Means Clustering Manual ---
    # Folosim track_tag_matrix direct pentru K-Means
    manual_kmeans_labels, manual_kmeans_centroids = manual_kmeans(
        track_tag_matrix,
        k_kmeans,
        max_iter=kmeans_max_iter,
        tol=kmeans_tol,
        random_state=42,
    )

    # --- Bucla principală de recomandare ---
    while True:
        idx = random.randint(0, len(df) - 1)

        get_recommendations(idx, track_tag_matrix, "Tag Similarity")
        get_recommendations(idx, W_mult, "NMF Multiplicative")
        get_recommendations(idx, W_sklearn, "NMF (sklearn)")
        get_recommendations_kmeans(
            idx, manual_kmeans_labels, track_tag_matrix, df, "Manual K-Means Clustering"
        )

        if input("\nPress Enter to continue (or 'q' to quit): ").lower() == "q":
            break


if __name__ == "__main__":
    main()
