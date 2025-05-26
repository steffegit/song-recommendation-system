import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF as SKNMF
import random
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import ttk


# -----------------------------------------------------------------------------
# Functia pentru Non-Negative Least Squares (NNLS)
# -----------------------------------------------------------------------------
def manual_nnls(A, b, max_iter=200, tol=1e-5, learning_rate=None):
    x = np.zeros(A.shape[1])

    if learning_rate is None:
        try:
            if A.size > 0 and np.any(np.sum(A**2, axis=0) > 1e-9):
                lipschitz_constant_approx = np.linalg.norm(A.T @ A, ord=2)
                if lipschitz_constant_approx > 1e-9:
                    learning_rate = 1.0 / lipschitz_constant_approx
                else:
                    learning_rate = 1e-3
            else:
                learning_rate = 1e-3
        except Exception:
            learning_rate = 1e-3

    if b.ndim > 1:
        b = b.flatten()  # Flatten b if it's a 2D array (e.g. if b is a column vector it turns into a row vector to be compatible with the matrix multiplication)

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
    error_list=None,
):
    if random_state is not None:
        np.random.seed(random_state)
    m, n = V.shape
    if r == 0:
        return np.zeros((m, 0)), np.zeros((0, n))
    W = np.abs(np.random.rand(m, r))
    H = np.abs(np.random.rand(r, n))
    print(f"Running ALS with Manual NNLS (r={r})...")
    prev_error_als = np.inf
    for iteration_als in range(max_iter_als):
        if H.shape[0] > 0:
            for i in range(m):
                if H.T.shape[1] > 0:
                    w_i_T = manual_nnls(
                        H.T, V[i, :].T, max_iter_nnls, tol_nnls, lr_nnls
                    )
                    W[i, :] = w_i_T
        if W.shape[1] > 0:
            for j in range(n):
                if W.shape[1] > 0:
                    H[:, j] = manual_nnls(W, V[:, j], max_iter_nnls, tol_nnls, lr_nnls)
        current_error_als = np.linalg.norm(V - W @ H, "fro")
        if error_list is not None:
            error_list.append(current_error_als)
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
        print(f"ALS-NNLS reached max_iter ({max_iter_als}) without full convergence.")
    print("ALS-NNLS finished.")
    return W, H


# -----------------------------------------------------------------------------
# Multiplicative NMF
# -----------------------------------------------------------------------------
def nmf_multiplicative(V, r, max_iter=100, tol=1e-4, error_list=None):
    m, n = V.shape
    if r == 0:
        return np.zeros((m, 0)), np.zeros((0, n))
    np.random.seed(42)
    W = np.abs(np.random.randn(m, r))
    H = np.abs(np.random.randn(r, n))
    for it in range(max_iter):
        WH = W @ H
        H_numerator = W.T @ V
        H_denominator = W.T @ WH + 1e-10
        H *= H_numerator / H_denominator
        WH = W @ H
        W_numerator = V @ H.T
        W_denominator = WH @ H.T + 1e-10
        W *= W_numerator / W_denominator
        error = np.linalg.norm(V - W @ H, "fro")
        if error_list is not None:
            error_list.append(error)
        if it % 10 == 0:
            print(f"Multiplicative NMF Iteration {it}: error = {error:.4f}")
        if error < tol:
            print(f"Multiplicative NMF converged at iteration {it}")
            break
    return W, H


# -----------------------------------------------------------------------------
# NMF din sklearn
# -----------------------------------------------------------------------------
def nmf_sklearn(V, r, max_iter=100, tol=1e-4, error_list=None):
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
    final_error = np.linalg.norm(V - W @ H, "fro")
    if error_list is not None:
        error_list.append(final_error)
    print(f"sklearn NMF: final error = {final_error:.4f}")
    return W, H


# -----------------------------------------------------------------------------
# Functie de Recomandare (modificata pentru a returna string)
# -----------------------------------------------------------------------------
def get_recommendations_text(idx, similarity_matrix, method_name, df_source, top_n=5):
    if (
        similarity_matrix.shape[0] == 0
        or similarity_matrix.shape[1] == 0
        or similarity_matrix.shape[0] != len(df_source)
    ):
        return f"\nReference: {df_source.iloc[idx]['name']} - {df_source.iloc[idx]['artist']}\nRecommendations ({method_name}):\n- Cannot generate recommendations (matrix issue)."

    similarities = cosine_similarity(
        similarity_matrix[idx : idx + 1], similarity_matrix
    )[0]
    similarities[idx] = -np.inf

    num_candidates = len(similarities)
    actual_top_n = min(top_n, num_candidates - 1 if num_candidates > 0 else 0)

    if actual_top_n <= 0:
        top_indices = []
    else:
        top_indices = similarities.argsort()[::-1][:actual_top_n]

    output_text = f"\nReference: {df_source.iloc[idx]['name']} - {df_source.iloc[idx]['artist']}\n"
    output_text += f"Recommendations ({method_name}):\n"
    if not top_indices.size:
        output_text += "- No recommendations found.\n"
    else:
        for i in top_indices:
            output_text += (
                f"- {df_source.iloc[i]['name']} - {df_source.iloc[i]['artist']}\n"
            )
    return output_text


# -----------------------------------------------------------------------------
# Incarcare si Preprocesare Date (global pentru GUI)
# -----------------------------------------------------------------------------
df = None
track_tag_matrix = None
all_tags = None
tag_to_idx = None


def load_and_preprocess_data():
    global df, track_tag_matrix, all_tags, tag_to_idx
    print("Loading data...")
    try:
        df = pd.read_csv("music_info.csv")
        if df.empty:
            messagebox.showerror("Error", "music_info.csv is empty or not found.")
            return False
    except FileNotFoundError:
        messagebox.showerror(
            "Error", "music_info.csv not found in the current directory."
        )
        return False

    print("Extracting unique tags...")
    all_tags_set = set()
    for tags_str in df["tags"].dropna():
        all_tags_set.update(tag.strip() for tag in tags_str.split(","))
    all_tags = sorted(list(all_tags_set))
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
    if (
        track_tag_matrix.size == 0
        or track_tag_matrix.shape[0] == 0
        or track_tag_matrix.shape[1] == 0
    ):
        messagebox.showerror(
            "Error",
            "Track x tag matrix is empty or has zero dimensions after preprocessing.",
        )
        return False
    return True


# -----------------------------------------------------------------------------
# Antrenare Modele (global pentru GUI)
# -----------------------------------------------------------------------------
W_mult, H_mult = None, None
W_sklearn, H_sklearn = None, None
W_als_manual, H_als_manual = None, None
error_mult, error_sklearn, error_als = [], [], []
r_factor_global = 10  # Seteaza r global


def train_models():
    global W_mult, H_mult, W_sklearn, H_sklearn, W_als_manual, H_als_manual
    global error_mult, error_sklearn, error_als, r_factor_global

    if track_tag_matrix is None or track_tag_matrix.size == 0:
        messagebox.showerror(
            "Error",
            "Data not loaded or track_tag_matrix is empty. Cannot train models.",
        )
        return

    als_iters = 20  # era 30 inainte
    als_tolerance = 1e-4
    nnls_iters = 50
    nnls_tolerance = 1e-5
    nnls_lr = 1e-3  # era 1e-4 inainte

    error_mult.clear()
    error_sklearn.clear()
    error_als.clear()

    print("\n--- Starting Multiplicative NMF ---")
    W_mult, H_mult = nmf_multiplicative(
        track_tag_matrix, r_factor_global, error_list=error_mult
    )

    print("\n--- Starting sklearn NMF ---")
    W_sklearn, H_sklearn = nmf_sklearn(
        track_tag_matrix, r_factor_global, error_list=error_sklearn
    )

    print("\n--- Starting ALS with Manual NNLS ---")
    W_als_manual, H_als_manual = als_manual_nnls(
        track_tag_matrix,
        r_factor_global,
        als_iters,
        als_tolerance,
        nnls_iters,
        nnls_tolerance,
        nnls_lr,
        42,
        error_list=error_als,
    )
    # Afisare metrici suplimentare pentru ALS (optional)
    if W_als_manual is not None and W_als_manual.size > 0:
        V_reconstructed_als = W_als_manual @ H_als_manual
        # Asigura-te ca nu sunt NaN/inf inainte de metrici
        if np.all(np.isfinite(track_tag_matrix)) and np.all(
            np.isfinite(V_reconstructed_als)
        ):
            try:
                print(
                    "Explained variance (ALS-NNLS):",
                    explained_variance_score(
                        track_tag_matrix.flatten(), V_reconstructed_als.flatten()
                    ),
                )
                print(
                    "MSE (ALS-NNLS):",
                    mean_squared_error(
                        track_tag_matrix.flatten(), V_reconstructed_als.flatten()
                    ),
                )
            except ValueError as e:
                print(f"Error calculating ALS metrics: {e}")
        else:
            print("Cannot calculate ALS metrics due to non-finite values.")


# -----------------------------------------------------------------------------
# Clasa GUI
# -----------------------------------------------------------------------------
class RecApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Sistem de Recomandare Muzicala")
        self.root.geometry("700x750")

        self.current_song_idx = -1
        self.current_song_label = tk.StringVar()
        self.reference_label = tk.StringVar()

        # Frame pentru informatii melodie si butoane
        self.info_frame = tk.Frame(self.root, pady=10)
        self.info_frame.pack(fill=tk.X)

        tk.Label(
            self.info_frame, text="Melodie de Referinta:", font=("Arial", 12)
        ).pack(side=tk.LEFT, padx=5)
        tk.Label(
            self.info_frame,
            textvariable=self.current_song_label,
            font=("Arial", 12, "bold"),
            fg="blue",
        ).pack(side=tk.LEFT, padx=5)

        self.new_song_button = tk.Button(
            self.info_frame, text="Alta Melodie", command=self.select_new_random_song
        )
        self.new_song_button.pack(side=tk.RIGHT, padx=10)

        self.select_song_button = tk.Button(
            self.info_frame, text="Selecteaza Melodie", command=self.select_song_by_name
        )
        self.select_song_button.pack(side=tk.RIGHT, padx=5)

        # Frame pentru butoanele de recomandare
        self.button_frame = tk.Frame(self.root, pady=5)
        self.button_frame.pack(fill=tk.X)

        btn_width = 20  # Latime comuna pentru butoane

        self.btn_tag_sim = tk.Button(
            self.button_frame,
            text="Tag Similarity",
            width=btn_width,
            command=lambda: self.show_recommendations("tag"),
        )
        self.btn_tag_sim.pack(side=tk.LEFT, expand=True, padx=2)

        self.btn_nmf_mult = tk.Button(
            self.button_frame,
            text="NMF Multiplicativ",
            width=btn_width,
            command=lambda: self.show_recommendations("nmf_mult"),
        )
        self.btn_nmf_mult.pack(side=tk.LEFT, expand=True, padx=2)

        self.btn_nmf_sklearn = tk.Button(
            self.button_frame,
            text="NMF sklearn",
            width=btn_width,
            command=lambda: self.show_recommendations("nmf_sklearn"),
        )
        self.btn_nmf_sklearn.pack(side=tk.LEFT, expand=True, padx=2)

        self.btn_als_manual = tk.Button(
            self.button_frame,
            text="ALS (Manual NNLS)",
            width=btn_width,
            command=lambda: self.show_recommendations("als_manual"),
        )
        self.btn_als_manual.pack(side=tk.LEFT, expand=True, padx=2)

        # Buton pentru afișarea graficului erorii
        self.btn_show_plot = tk.Button(
            self.root,
            text="Afișează graficul erorii",
            font=("Arial", 11),
            command=self.plot_loss_curves,
        )
        self.btn_show_plot.pack(pady=(0, 10))

        # Label pentru referinta si tabel pentru recomandari
        self.reference_frame = tk.Frame(self.root)
        self.reference_frame.pack(fill=tk.X, pady=(10, 0))
        self.reference_label_widget = tk.Label(
            self.reference_frame,
            textvariable=self.reference_label,
            font=("Arial", 11, "bold"),
            fg="darkgreen",
        )
        self.reference_label_widget.pack(anchor="w", padx=10)

        self.table_frame = tk.Frame(self.root)
        self.table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.recommendation_table = ttk.Treeview(
            self.table_frame, columns=("Nume", "Artist"), show="headings", height=10
        )
        self.recommendation_table.heading("Nume", text="Nume")
        self.recommendation_table.heading("Artist", text="Artist")
        self.recommendation_table.column("Nume", width=300)
        self.recommendation_table.column("Artist", width=200)
        self.recommendation_table.pack(fill=tk.BOTH, expand=True)

        if load_and_preprocess_data():
            train_models()
            self.select_new_random_song()  # Selecteaza prima melodie
        else:
            self.root.quit()  # Inchide daca datele nu s-au incarcat

    def select_new_random_song(self):
        if df is not None and not df.empty:
            self.current_song_idx = random.randint(0, len(df) - 1)
            song_info = f"{df.iloc[self.current_song_idx]['name']} - {df.iloc[self.current_song_idx]['artist']}"
            self.current_song_label.set(song_info)
            self.reference_label.set("")
            self.clear_table()
        else:
            self.current_song_label.set("Nicio melodie incarcata.")
            self.reference_label.set("")
            self.clear_table()

    def select_song_by_name(self):
        if df is None or df.empty:
            messagebox.showinfo("Info", "Datele nu sunt incarcate.")
            return

        song_name_query = simpledialog.askstring(
            "Selectare Melodie", "Introduceti numele melodiei (sau o parte din el):"
        )
        if song_name_query:
            song_name_query = song_name_query.lower()
            matches = df[df["name"].str.lower().str.contains(song_name_query, na=False)]
            if not matches.empty:
                if len(matches) == 1:
                    self.current_song_idx = matches.index[0]
                else:
                    match_list_str = "\n".join(
                        [
                            f"{idx}: {row['name']} - {row['artist']}"
                            for idx, row in matches.iterrows()
                        ]
                    )
                    selected_idx_str = simpledialog.askstring(
                        "Multiple Potriviri",
                        f"Melodii gasite:\n{match_list_str}\n\nIntroduceti indexul melodiei dorite:",
                    )
                    try:
                        selected_idx = int(selected_idx_str)
                        if selected_idx in matches.index:
                            self.current_song_idx = selected_idx
                        else:
                            messagebox.showerror("Eroare", "Index invalid.")
                            return
                    except (ValueError, TypeError):
                        messagebox.showerror(
                            "Eroare", "Introduceti un numar valid pentru index."
                        )
                        return

                song_info = f"{df.iloc[self.current_song_idx]['name']} - {df.iloc[self.current_song_idx]['artist']}"
                self.current_song_label.set(song_info)
                self.reference_label.set("")
                self.clear_table()
            else:
                messagebox.showinfo(
                    "Info", f"Nicio melodie gasita cu numele '{song_name_query}'."
                )

    def clear_table(self):
        for row in self.recommendation_table.get_children():
            self.recommendation_table.delete(row)

    def show_recommendations(self, method_key):
        if self.current_song_idx == -1 or df is None or track_tag_matrix is None:
            self.reference_label.set("Va rugam selectati o melodie de referinta intai.")
            self.clear_table()
            return

        self.clear_table()
        reference_text = ""
        recommendations = []
        method_name = ""

        if method_key == "tag":
            method_name = "Tag Similarity (Cosine)"
            similarities = cosine_similarity(
                track_tag_matrix[self.current_song_idx : self.current_song_idx + 1],
                track_tag_matrix,
            )[0]
            similarities[self.current_song_idx] = -np.inf
            top_indices = similarities.argsort()[::-1][:5]
            reference_text = f"Referinta: {df.iloc[self.current_song_idx]['name']} - {df.iloc[self.current_song_idx]['artist']}\nRecomandari ({method_name}):"
            for i in top_indices:
                recommendations.append((df.iloc[i]["name"], df.iloc[i]["artist"]))
        elif method_key == "nmf_mult" and W_mult is not None:
            method_name = "NMF Multiplicative"
            similarities = cosine_similarity(
                W_mult[self.current_song_idx : self.current_song_idx + 1], W_mult
            )[0]
            similarities[self.current_song_idx] = -np.inf
            top_indices = similarities.argsort()[::-1][:5]
            reference_text = f"Referinta: {df.iloc[self.current_song_idx]['name']} - {df.iloc[self.current_song_idx]['artist']}\nRecomandari ({method_name}):"
            for i in top_indices:
                recommendations.append((df.iloc[i]["name"], df.iloc[i]["artist"]))
        elif method_key == "nmf_sklearn" and W_sklearn is not None:
            method_name = "NMF (sklearn)"
            similarities = cosine_similarity(
                W_sklearn[self.current_song_idx : self.current_song_idx + 1], W_sklearn
            )[0]
            similarities[self.current_song_idx] = -np.inf
            top_indices = similarities.argsort()[::-1][:5]
            reference_text = f"Referinta: {df.iloc[self.current_song_idx]['name']} - {df.iloc[self.current_song_idx]['artist']}\nRecomandari ({method_name}):"
            for i in top_indices:
                recommendations.append((df.iloc[i]["name"], df.iloc[i]["artist"]))
        elif method_key == "als_manual" and W_als_manual is not None:
            method_name = "ALS (Manual NNLS)"
            similarities = cosine_similarity(
                W_als_manual[self.current_song_idx : self.current_song_idx + 1],
                W_als_manual,
            )[0]
            similarities[self.current_song_idx] = -np.inf
            top_indices = similarities.argsort()[::-1][:5]
            reference_text = f"Referinta: {df.iloc[self.current_song_idx]['name']} - {df.iloc[self.current_song_idx]['artist']}\nRecomandari ({method_name}):"
            for i in top_indices:
                recommendations.append((df.iloc[i]["name"], df.iloc[i]["artist"]))
        else:
            reference_text = (
                f"Modelul pentru '{method_key}' nu este antrenat sau disponibil."
            )

        self.reference_label.set(reference_text)
        for name, artist in recommendations:
            self.recommendation_table.insert("", "end", values=(name, artist))

    def plot_loss_curves(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_made = False
        if error_mult:
            ax.plot(error_mult, label="Multiplicative NMF", marker=".")
            plot_made = True
        if error_als:
            ax.plot(error_als, label="ALS (Manual NNLS)", marker="x")
            plot_made = True
        if error_sklearn:
            max_len = 0
            if error_mult:
                max_len = len(error_mult)
            if error_als:
                max_len = max(max_len, len(error_als))
            if max_len > 0:
                ax.plot(
                    [max_len - 1],
                    error_sklearn,
                    "o",
                    markersize=8,
                    label="NMF (sklearn) final",
                )
            else:
                ax.plot(
                    [0], error_sklearn, "o", markersize=8, label="NMF (sklearn) final"
                )
            plot_made = True
        if plot_made:
            ax.set_xlabel("Iteratie (aproximativa pentru sklearn)")
            ax.set_ylabel("Eroare Frobenius")
            ax.set_title("Evolutia Erorii de Reconstructie")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            plt.show()
        else:
            tk.messagebox.showinfo(
                "Grafic indisponibil",
                "Datele despre erori nu sunt disponibile pentru grafic.",
            )


# -----------------------------------------------------------------------------
# Pornire Aplicatie
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = RecApp(root)
    root.mainloop()
