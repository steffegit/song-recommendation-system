import pandas as pd
import numpy as np

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
for i, tags in enumerate(df["tags"].fillna("")):
    for tag in tags.split(","):
        tag = tag.strip()
        if tag in all_tags:
            j = all_tags.index(tag)
            track_tag_matrix[i, j] = 1

# 2. Ascunde 10% din tag-urile 1 pentru testare
V = track_tag_matrix.copy()
test_idx = []
np.random.seed(42)
for i in range(V.shape[0]):
    ones = np.where(V[i, :] == 1)[0]
    if len(ones) > 0:
        hide = np.random.choice(ones, size=max(1, int(0.1 * len(ones))), replace=False)
        for j in hide:
            test_idx.append((i, j))
            V[i, j] = 0


# 3. NMF cu multiplicative update
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


# 4. Rulează NMF și evaluează
r = 10
W, H, errors = nmf_multiplicative(V, r, max_iter=1000, verbose=True)
V_pred = np.dot(W, H)

# 5. Evaluează pe tag-urile ascunse
real = np.array([track_tag_matrix[i, j] for (i, j) in test_idx])
pred = np.array([V_pred[i, j] for (i, j) in test_idx])
rmse = np.sqrt(np.mean((real - pred) ** 2))
print(f"\nRMSE pe tag-urile ascunse: {rmse:.4f}")

# Hit-rate: procent tag-uri ascunse recuperate în top 3 scoruri pentru fiecare piesă
hit_count = 0
for (i, j), p in zip(test_idx, pred):
    top_tags = np.argsort(-V_pred[i, :])[:3]
    if j in top_tags:
        hit_count += 1
print(
    f"Procent tag-uri ascunse recuperate în top 3 scoruri: {100 * hit_count / len(test_idx):.2f}%"
)

# 6. Grafic evoluție eroare
import matplotlib.pyplot as plt

plt.plot(errors)
plt.xlabel("Iterație")
plt.ylabel("Eroare reconstrucție (Frobenius)")
plt.title("Convergența NMF (Multiplicative Update)")
plt.show()

# 7. Exemplu: Recomandă tag-uri noi pentru o piesă aleasă random
import random

while True:
    idx = random.randint(0, len(df) - 1)
    taguri_deja = [all_tags[j] for j in np.where(track_tag_matrix[idx, :] == 1)[0]]
    taguri_recomandate = [
        all_tags[j]
        for j in np.argsort(-V_pred[idx, :])
        if track_tag_matrix[idx, j] == 0
    ][:5]
    print(f"\nPiesă: {df.iloc[idx]['name']} - {df.iloc[idx]['artist']}")
    print("Tag-uri deja asociate:", taguri_deja)
    print("Tag-uri recomandate:", taguri_recomandate)

    input("Apăsați Enter pentru a continua...")
