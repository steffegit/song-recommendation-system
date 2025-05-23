from sklearn.metrics.pairwise import cosine_similarity


class MusicRecommender:
    def __init__(self, df, track_tag_matrix, W, track_tag_normalized, W_normalized):
        self.df = df
        self.track_tag_matrix = track_tag_matrix
        self.W = W
        self.track_tag_normalized = track_tag_normalized
        self.W_normalized = W_normalized

    def recommend_by_tags(self, idx, top_n=5, diversity_weight=0.1):
        """Tag-based recommendations with diversity boost"""
        song_vec = self.track_tag_normalized[idx : idx + 1]
        similarities = cosine_similarity(song_vec, self.track_tag_normalized)[0]

        # Add diversity: slightly penalize songs that are too similar to already high-scored ones
        sorted_indices = similarities.argsort()[::-1]
        adjusted_scores = similarities.copy()

        for i, idx_candidate in enumerate(sorted_indices[1:20]):  # Check top 20
            for j in range(i):
                prev_idx = sorted_indices[j]
                inter_similarity = cosine_similarity(
                    self.track_tag_normalized[idx_candidate : idx_candidate + 1],
                    self.track_tag_normalized[prev_idx : prev_idx + 1],
                )[0][0]
                adjusted_scores[idx_candidate] -= diversity_weight * inter_similarity

        top_indices = adjusted_scores.argsort()[::-1]
        top_indices = [i for i in top_indices if i != idx][:top_n]

        return self._format_recommendations(idx, top_indices, "Enhanced tag similarity")

    def recommend_by_nmf(self, idx, top_n=5):
        """NMF-based recommendations with artist diversity"""
        song_vec = self.W_normalized[idx : idx + 1]
        similarities = cosine_similarity(song_vec, self.W_normalized)[0]

        # Get more candidates than needed
        candidates = similarities.argsort()[::-1]
        candidates = [i for i in candidates if i != idx]

        # Apply artist diversity: avoid recommending too many songs from same artist
        selected = []
        artist_count = {}
        reference_artist = self.df.iloc[idx]["artist"]

        for candidate_idx in candidates:
            if len(selected) >= top_n:
                break

            candidate_artist = self.df.iloc[candidate_idx]["artist"]

            # Limit songs per artist (except reference artist)
            if candidate_artist == reference_artist:
                continue
            if artist_count.get(candidate_artist, 0) >= 2:  # Max 2 per artist
                continue

            selected.append(candidate_idx)
            artist_count[candidate_artist] = artist_count.get(candidate_artist, 0) + 1

        return self._format_recommendations(
            idx, selected, "NMF latent factors + diversity"
        )

    def recommend_hybrid(self, idx, top_n=5, tag_weight=0.6, nmf_weight=0.4):
        """Hybrid approach combining both methods"""
        # Get similarities from both methods
        tag_similarities = cosine_similarity(
            self.track_tag_normalized[idx : idx + 1], self.track_tag_normalized
        )[0]

        nmf_similarities = cosine_similarity(
            self.W_normalized[idx : idx + 1], self.W_normalized
        )[0]

        # Combine with weights
        hybrid_scores = tag_weight * tag_similarities + nmf_weight * nmf_similarities

        top_indices = hybrid_scores.argsort()[::-1]
        top_indices = [i for i in top_indices if i != idx][:top_n]

        return self._format_recommendations(idx, top_indices, "Hybrid (tag + NMF)")

    def _format_recommendations(
        self, reference_idx, recommendation_indices, method_name
    ):
        """Format and return recommendations"""
        results = {
            "reference": {
                "name": self.df.iloc[reference_idx]["name"],
                "artist": self.df.iloc[reference_idx]["artist"],
            },
            "method": method_name,
            "recommendations": [],
        }

        for i in recommendation_indices:
            results["recommendations"].append(
                {
                    "name": self.df.iloc[i]["name"],
                    "artist": self.df.iloc[i]["artist"],
                    "tags": self.df.iloc[i]["tags"],
                }
            )

        return results

    def print_recommendations(self, results):
        """Pretty print recommendations"""
        ref = results["reference"]
        print(f"\n🎵 Reference: {ref['name']} - {ref['artist']}")
        print(f"📋 Method: {results['method']}")
        print("🎯 Recommendations:")

        for i, rec in enumerate(results["recommendations"], 1):
            print(f"  {i}. {rec['name']} - {rec['artist']}")
            if "tags" in rec and rec["tags"]:
                tags = (
                    rec["tags"][:50] + "..." if len(rec["tags"]) > 50 else rec["tags"]
                )
                print(f"     Tags: {tags}")
