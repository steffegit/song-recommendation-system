import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# 1. Încarcă datele și construiește matricea piesă × tag
df = pd.read_csv("music_info.csv")  # înlocuiește cu numele fișierului tău

# Extrage lista de tag-uri unice
all_tags = set()
for tags in df["tags"].dropna():
    for tag in tags.split(","):
        all_tags.add(tag.strip())
all_tags = sorted(list(all_tags))

# Creează matricea piesă × tag (binary: 1 dacă piesa are tag-ul, 0 altfel)
track_tag_matrix = np.zeros((len(df), len(all_tags)), dtype=np.float32)
tag_to_idx = {tag: idx for idx, tag in enumerate(all_tags)}
for i, tags in enumerate(df["tags"].fillna("")):
    for tag in tags.split(","):
        tag = tag.strip()
        if tag in tag_to_idx:
            j = tag_to_idx[tag]
            track_tag_matrix[i, j] = 1


# 2. NMF cu multiplicative update (poți folosi și codul tău anterior)
def nmf_multiplicative(V, r, max_iter=100, tol=1e-4, verbose=True):
    m, n = V.shape
    np.random.seed(42)
    W = np.abs(np.random.randn(m, r))
    H = np.abs(np.random.randn(r, n))
    errors = []
    for it in range(max_iter):
        WH = np.dot(W, H)
        H *= (W.T @ V) / (W.T @ WH + 1e-10)
        WH = np.dot(W, H)
        W *= (V @ H.T) / (WH @ H.T + 1e-10)
        error = np.linalg.norm(V - np.dot(W, H), "fro")
        errors.append(error)
        if verbose and (it % 10 == 0 or it == max_iter - 1):
            print(f"Iterația {it}: eroare reconstrucție = {error:.4f}")
        if it > 0 and abs(errors[-2] - errors[-1]) < tol:
            print(f"Convergență la iterația {it} cu eroare {error:.4f}")
            break
    return W, H, errors


# Rulează NMF (dacă nu ai deja W, H)
r = 10
W, H, errors = nmf_multiplicative(track_tag_matrix, r, max_iter=100, verbose=True)


def recommend_next_song_by_tags(idx, track_tag_matrix, df, top_n=5):
    song_vec = track_tag_matrix[idx, :].reshape(1, -1)
    similarities = cosine_similarity(song_vec, track_tag_matrix)[0]
    top_indices = similarities.argsort()[::-1]
    top_indices = [i for i in top_indices if i != idx][:top_n]
    print(f"\nPiesă de referință: {df.iloc[idx]['name']} - {df.iloc[idx]['artist']}")
    print("Recomandări next song (cele mai similare pe tag-uri):")
    for i in top_indices:
        print(f"- {df.iloc[i]['name']} - {df.iloc[i]['artist']}")


def recommend_next_song_by_nmf(idx, W, df, top_n=5):
    song_vec = W[idx, :].reshape(1, -1)
    similarities = cosine_similarity(song_vec, W)[0]
    top_indices = similarities.argsort()[::-1]
    top_indices = [i for i in top_indices if i != idx][:top_n]
    print(f"\nPiesă de referință: {df.iloc[idx]['name']} - {df.iloc[idx]['artist']}")
    print("Recomandări next song (bazat pe factori latenti NMF):")
    for i in top_indices:
        print(f"- {df.iloc[i]['name']} - {df.iloc[i]['artist']}")


# 5. Exemplu de utilizare: alege o piesă random sau una anume
import random

idx = random.randint(0, len(df) - 1)
# Sau caută după nume:
# song_name = "I Like To Move It"
# idx = df[df["name"].str.contains(song_name, case=False, na=False)].index[0]

while True:
    idx = random.randint(0, len(df) - 1)
    recommend_next_song_by_tags(idx, track_tag_matrix, df, top_n=5)
    recommend_next_song_by_nmf(idx, W, df, top_n=5)

    input("Apăsați Enter pentru a continua...")
