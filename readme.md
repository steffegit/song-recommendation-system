# 🎼 Matrix Factorization-Based Music Recommendation System

> A modular system with three applications for delivering **personalized music recommendations** using **non-negative matrix factorization (NMF)** and **advanced constrained optimization techniques**.

---

## 📄 Documentation

📝 [**View Full Project Documentation (PDF)**](docs/document.pdf)

---

## 👨‍💻 Authors

<table>
  <tr>
    <td align="center"><a href="https://github.com/steffegit"><strong>Gatej Ștefan-Alexandru</strong></a></td>
    <td align="center"><a href="https://github.com/horicuz"><strong>Potop Horia-Ioan</strong></a></td>
  </tr>
</table>

---

## Overview

This recommendation system analyzes the **song–tag binary matrix** and applies **non-negative matrix factorization** to uncover latent musical features. These hidden traits enable the generation of personalized suggestions that go beyond simple genre matching.

### 🛠️ Applications

- 🎵 **Recommend next song (AutoPlay) **  
  Generates coherent playlists starting from a single track. Respects duration and relevance constraints using NMF components.

- 🎵 **Song → Playlist** 
  Recommends a playlist based on a single song, respecting users specifications like (Total duration, diversity, etc..) using ALS, NNLS, PGD, Acc PGD, and predefined functions.
  
- 🎵 **Playlist → Song**  
  Recommends new songs to complete an existing playlist. Uses **L1/L2-regularized optimization** on the latent space to select relevant additions.

---

## 📦 Installation

```bash
git clone https://github.com/steffegit/song-recommendation-system.git
cd song-recommendation-system
```

---

## 🧪 Tech Stack

- Python 🐍
- NumPy, SciPy (Minimize)
- Optimization: PGD (simple and acc), ALS, NNLS
- NMF for dimensionality reduction

---

## 📬 Contact

Feel free to reach out for questions or collaboration via GitHub Issues or directly on our profiles.

---

🔗 **Give us a ⭐ if you find this project useful!**
