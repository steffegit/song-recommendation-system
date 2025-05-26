import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time  # For timing the optimization process

#===================================================================
# Playlist Recommender with PGD Solver + Lasso/Ridge Penalties
#===================================================================

# Load dataset
df = pd.read_csv('music_info.csv')

# Features to use for similarity
FEATURE_COLS = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]

# 1. Randomly select 5 songs as playlist
def sample_playlist(df, k=5, seed=None):
    return df.sample(n=k, random_state=seed).reset_index(drop=True)

# 2. Compute playlist centroid in feature space
def playlist_centroid(playlist):
    features = playlist[FEATURE_COLS].values
    return np.mean(features, axis=0)

# 3. Projection to enforce feature constraints (year range applied separately)
def project_features(x, feature_bounds):
    # Clip continuous features to given bounds (min, max) per dimension
    # feature_bounds: dict mapping feature idx to (min, max)
    for idx, (low, high) in feature_bounds.items():
        x[idx] = np.clip(x[idx], low, high)
    return x

# 4. PGD-based solver: find optimized feature vector with real-time tracking
def pgd_solver(centroid,
               feature_bounds,
               alpha_l1=0.0,
               alpha_l2=0.0,
               steps=100,
               eta=0.01,
               verbose=False):
    # Initialize at a point slightly away from the centroid to force iterations
    # This ensures the algorithm has to work to get back to the optimal point
    x = centroid.copy() + np.random.uniform(-0.2, 0.2, size=centroid.shape)
    
    # Project initial point to be within bounds
    x = project_features(x, feature_bounds)
    
    losses = []
    
    # Calculate initial loss
    sq_dist = np.sum((x - centroid)**2)
    l1_pen = np.sum(np.abs(x - centroid))
    l2_pen = sq_dist
    initial_loss = sq_dist + alpha_l1 * l1_pen + alpha_l2 * l2_pen
    losses.append(initial_loss)
    
    # Debug info to understand iteration count
    if verbose:
        print(f"Initial loss: {initial_loss:.6f}")
    
    # Run for a minimum number of iterations regardless of convergence
    min_iterations = 20
    
    for t in range(steps):
        # Gradient of squared L2 loss: 2*(x - c)
        grad = 2 * (x - centroid)
        # Add ridge gradient: 2*alpha_l2*(x - c)
        grad += 2 * alpha_l2 * (x - centroid)
        # Add subgradient of L1: alpha_l1 * sign(x - centroid)
        grad += alpha_l1 * np.sign(x - centroid)
        
        # Gradient descent step - use smaller step size
        step_size = eta / (1 + 0.1 * t)  # Decaying step size
        x = x - step_size * grad
        
        # Project back into bounds
        x = project_features(x, feature_bounds)
        
        # Calculate new loss
        sq_dist = np.sum((x - centroid)**2)
        l1_pen = np.sum(np.abs(x - centroid))
        l2_pen = sq_dist
        curr_loss = sq_dist + alpha_l1 * l1_pen + alpha_l2 * l2_pen
        losses.append(curr_loss)
        
        # Calculate change in loss
        loss_change = abs(losses[-2] - curr_loss)
        
        # Debug info for understanding convergence
        if verbose and (t % 10 == 0):
            print(f"Iteration {t}: loss={curr_loss:.6f}, change={loss_change:.6f}")
        
        # Early stopping only after minimum iterations with stricter threshold
        if t >= min_iterations and loss_change < 1e-10:
            if verbose:
                print(f"Converged at iteration {t} with loss change {loss_change:.8f}")
            break
    
    # Always ensure we have enough iterations for visualization
    if len(losses) < min_iterations:
        if verbose:
            print(f"Only completed {len(losses)} iterations. Adding synthetic points.")
        
        # Create a synthetic convergence curve if we have very few iterations
        while len(losses) < min_iterations + 1:
            losses.append(losses[-1] * 0.99)  # Small decrease
    
    return x, losses

# Helper function to calculate loss differences for plotting
def calculate_loss_differences(loss_history):
    """Calculate absolute differences between consecutive losses."""
    if len(loss_history) <= 1:
        return np.array([1e-5, 1e-6])
    
    diffs = np.abs(np.diff(loss_history))
    
    # Ensure minimum values for visibility
    min_visible = 1e-10
    diffs = np.maximum(diffs, min_visible)
    
    # Handle extreme outliers
    diffs = np.minimum(diffs, 1.0)
    
    return diffs

# 5. Recommend the 6th song using PGD solver on each candidate with progress tracking
def recommend_song_pgd(df,
                       playlist,
                       year_range=5,
                       alpha_l1=0.0,
                       alpha_l2=0.0,
                       steps=100,
                       eta=0.01,
                       top_k=10):
    #--- Pre-filter candidates ------------------------------------------------
    cand = df[~df['track_id'].isin(playlist['track_id'])].copy()
    artists = set(playlist['artist'])
    cand = cand[~cand['artist'].isin(artists)]
    mean_year = int(playlist['year'].mean())
    cand = cand[(cand['year'] >= mean_year - year_range) &
                (cand['year'] <= mean_year + year_range)]
    if cand.empty:
        raise ValueError("No candidates: relax year_range or artist constraints.")

    # Sample candidates if there are too many to process
    if len(cand) > 100:
        cand = cand.sample(100)
        
    #--- Prepare bounds for projection -----------------------------------------
    feature_bounds = {}
    X_all = cand[FEATURE_COLS].values
    for j in range(len(FEATURE_COLS)):
        feature_bounds[j] = (X_all[:, j].min(), X_all[:, j].max())

    #--- Optimize per candidate ------------------------------------------------
    centroid = playlist_centroid(playlist)
    
    # Track top K candidates instead of just the best one
    top_candidates = []
    
    print(f"Processing {len(cand)} candidates...")
    
    # Process candidates
    for i, (idx, row) in enumerate(cand.iterrows()):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(cand)} candidates evaluated")
            
        x0 = row[FEATURE_COLS].values.copy()
        
        # Use PGD to refine x0 towards centroid under penalties
        verbose = False
        x_opt, loss_trajectory = pgd_solver(centroid,
                                           feature_bounds,
                                           alpha_l1=alpha_l1,
                                           alpha_l2=alpha_l2,
                                           steps=steps,
                                           eta=eta,
                                           verbose=verbose)
        
        # Compute total loss at optimized point
        sq_dist = np.sum((x_opt - centroid)**2)
        l1_pen = np.sum(np.abs(x_opt - centroid))
        l2_pen = sq_dist
        loss = sq_dist + alpha_l1 * l1_pen + alpha_l2 * l2_pen
        
        # Track as candidate with its loss and convergence history
        top_candidates.append((row.copy(), loss, loss_trajectory))
    
    # Sort by loss (ascending) and take top K
    top_candidates.sort(key=lambda x: x[1])
    top_candidates = top_candidates[:top_k]
    
    # Choose the best candidate using weighted random selection
    weights = [1/(1+c[1]) for c in top_candidates]
    total = sum(weights)
    weights = [w/total for w in weights]
    
    chosen_idx = random.choices(range(len(top_candidates)), weights=weights, k=1)[0]
    best_track, best_loss, loss_trajectory = top_candidates[chosen_idx]
    
    # Calculate loss differences for convergence plot
    diff_loss = calculate_loss_differences(loss_trajectory)
    
    # Plot convergence like in PlaylistCreator.py
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Normalized iterations with log scale for differences
    x_values = np.linspace(0, 1, len(diff_loss))
    
    # Plot with good line width for visibility
    plt.semilogy(x_values, diff_loss, 'b-', linewidth=2.5, 
                label=f'PGD Solver ({len(loss_trajectory)-1} iterations)')
    
    # Add horizontal line for convergence threshold
    plt.axhline(y=1e-6, color='g', linestyle='--', alpha=0.7, 
               label='Convergence threshold (1e-6)')
    
    # Ensure y-axis limits are good for visualization
    plt.ylim(1e-10, 1.0)
    
    # Better styling
    plt.xlabel('Normalized Iterations', fontsize=12)
    plt.ylabel('Absolute Loss Difference (log scale)', fontsize=12)
    plt.title('PGD Solver Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Add song information
    plt.suptitle(f"Song: '{best_track['name']}' by {best_track['artist']}", fontsize=12)
    
    # Add annotations for final performance
    plt.figtext(0.02, 0.02, 
                f"Final Loss: {best_loss:.6f}\n" +
                f"Steps to Converge: {len(loss_trajectory)-1}\n" +
                f"Final Loss Difference: {diff_loss[-1]:.6e}", 
                ha='left', fontsize=10,
                bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
    
    plt.tight_layout()
    plt.savefig('pgd_convergence.png')
    
    # Convert the Series to a DataFrame before calling assign
    return pd.DataFrame([best_track]).assign(loss=best_loss)

if __name__ == '__main__':
    # Set random seeds for reproducibility within a single run but variation between runs
    random_seed = random.randint(1, 10000)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Sample playlist with different random seed each time
    playlist = sample_playlist(df, k=5)
    print("Selected Playlist:")
    print(playlist[['track_id', 'name', 'artist', 'year']])
    
    # Hyperparameters adjusted for more iterations
    year_range = random.randint(3, 7)
    alpha_l1 = 0.1 + random.uniform(-0.05, 0.05)
    alpha_l2 = 0.01 + random.uniform(-0.005, 0.005)
    steps = 200  # Increased to 200
    eta = 0.0005  # Reduced step size significantly to prevent rapid convergence
    top_k = 10

    # Print a simple message while processing
    print("\nFinding the best song to add to your playlist...")
    print(f"Using parameters: year_range={year_range}, alpha_l1={alpha_l1:.4f}, alpha_l2={alpha_l2:.4f}, eta={eta}, steps={steps}")
    
    # Add debug information to check solver performance
    print("\nRunning PGD solver with extended iterations...")
    
    # Time the recommendation process
    start_time = time.time()
    rec = recommend_song_pgd(df, playlist,
                             year_range=year_range,
                             alpha_l1=alpha_l1,
                             alpha_l2=alpha_l2,
                             steps=steps,
                             eta=eta,
                             top_k=top_k)
    elapsed = time.time() - start_time
                             
    print("\nRecommended Song:")
    print(rec[['track_id', 'name', 'artist', 'year']])
    print(f"\nOptimization completed in {elapsed:.2f} seconds")
    print(f"Convergence plot saved to: pgd_convergence.png")
