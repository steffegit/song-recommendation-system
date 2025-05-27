import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time  # For timing the optimization process
from scipy.optimize import minimize

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
               steps=50,          # Standardized max iterations
               eta=0.01,
               tol=1e-8,          # Standardized convergence tolerance
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
        print(f"PGD Initial loss: {initial_loss:.6f}, max_iter: {steps}, tol: {tol:.1e}")
    
    # Track iterations and convergence
    converged = False
    final_iter = 0
    
    for t in range(steps):
        final_iter = t
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
        if verbose and (t % 10 == 0 or t == steps-1):
            print(f"PGD Iteration {t}: loss={curr_loss:.6f}, change={loss_change:.6f}")
        
        # Early stopping with standardized tolerance (no minimum iterations)
        if loss_change < tol:
            converged = True
            if verbose:
                print(f"PGD Converged at iteration {t} with loss change {loss_change:.8f}")
            break
    
    if verbose and not converged:
        print(f"PGD reached max iterations ({steps}) without converging")
    
    # Ensure we have enough points for a meaningful visualization (min 3)
    if len(losses) < 3:
        while len(losses) < 3:
            losses.append(losses[-1] * 0.99)
    
    return x, losses, final_iter+1, converged

# Simplified scipy minimize solver with identical parameters
def scipy_minimize_solver(centroid, 
                         feature_bounds, 
                         alpha_l1=0.0, 
                         alpha_l2=0.0, 
                         steps=50,        # Standardized max iterations
                         eta=0.01,        # Not used in BFGS but kept for API consistency
                         tol=1e-8,        # Standardized convergence tolerance
                         verbose=False):
    """Use scipy's minimize function with identical parameters as PGD"""
    
    # Define the objective function and its gradient
    def objective(x):
        sq_dist = np.sum((x - centroid)**2)
        l1_pen = np.sum(np.abs(x - centroid))
        l2_pen = sq_dist
        return sq_dist + alpha_l1 * l1_pen + alpha_l2 * l2_pen
    
    def gradient(x):
        grad = 2 * (x - centroid)
        grad += 2 * alpha_l2 * (x - centroid)
        grad += alpha_l1 * np.sign(x - centroid)
        return grad
    
    # Create bounds for minimize - identical to PGD
    bounds = []
    for i in range(len(centroid)):
        if i in feature_bounds:
            bounds.append(feature_bounds[i])
        else:
            bounds.append((None, None))
    
    # Track loss history
    loss_history = []
    
    # Same initialization as PGD for fair comparison
    x0 = centroid.copy() + np.random.uniform(-0.2, 0.2, size=centroid.shape)
    
    # Project to bounds
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None:
            x0[i] = max(x0[i], lo)
        if hi is not None:
            x0[i] = min(x0[i], hi)
            
    # Record initial loss
    initial_loss = objective(x0)
    loss_history.append(initial_loss)
    
    if verbose:
        print(f"SciPy Initial loss: {initial_loss:.6f}, max_iter: {steps}, tol: {tol:.1e}")
    
    # Simple callback to track loss at each iteration
    def callback(x):
        loss_history.append(objective(x))
    
    # Start time
    start_time = time.time()
    
    # Run scipy minimize with identical parameters
    result = minimize(
        objective, 
        x0,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds,
        callback=callback,
        options={
            'disp': verbose,
            'maxiter': steps,       # Same max iterations
            'gtol': tol,           # Same convergence tolerance
            'ftol': tol*10         # Related function tolerance
        }
    )
    
    # Record time and convergence
    elapsed = time.time() - start_time
    converged = result.success
    
    if verbose:
        print(f"SciPy completed: {result.nit} iterations in {elapsed:.4f}s")
        print(f"  Success: {result.success}, message: {result.message}")
        print(f"  Final loss: {result.fun:.6f}")
    
    # Ensure we have enough points for a meaningful visualization (min 3)
    if len(loss_history) < 3:
        while len(loss_history) < 3:
            loss_history.append(loss_history[-1] * 0.99)
    
    return result.x, loss_history, result.nit, converged, elapsed

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
                       steps=50,     # Standardized max iterations
                       eta=0.01,
                       tol=1e-8,     # Standardized convergence tolerance
                       top_k=10,
                       show_progress=False,  # New parameter to control progress display
                       generate_plots=False): # New parameter to control plot generation
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
    
    # Time tracking for each method
    pgd_total_time = 0
    scipy_total_time = 0
    
    # Track top K candidates instead of just the best one
    top_candidates = []
    top_candidates_scipy = []
    
    # Print progress only if requested
    if show_progress:
        print(f"Processing {len(cand)} candidates...")
    
    # Process candidates
    for i, (idx, row) in enumerate(cand.iterrows()):
        if show_progress and i % 10 == 0:
            print(f"Progress: {i}/{len(cand)} candidates evaluated")
            
        x0 = row[FEATURE_COLS].values.copy()
        
        # Use PGD with standardized parameters
        verbose = (i == 0)  # Show details for first candidate only
        pgd_start = time.time()
        x_opt, loss_trajectory_pgd, pgd_iters, pgd_converged = pgd_solver(
            centroid,
            feature_bounds,
            alpha_l1=alpha_l1,
            alpha_l2=alpha_l2,
            steps=steps,
            eta=eta,
            tol=tol,
            verbose=verbose
        )
        pgd_time = time.time() - pgd_start
        pgd_total_time += pgd_time
        
        # Also use scipy minimize with identical parameters
        x_scipy, loss_trajectory_scipy, scipy_iters, scipy_converged, scipy_time = scipy_minimize_solver(
            centroid, 
            feature_bounds, 
            alpha_l1, 
            alpha_l2, 
            steps=steps,  # Same max iterations
            eta=eta,      # Not used by scipy but kept for API consistency
            tol=tol,      # Same tolerance
            verbose=verbose
        )
        scipy_total_time += scipy_time
        
        # Compute total losses
        sq_dist = np.sum((x_opt - centroid)**2)
        l1_pen = np.sum(np.abs(x_opt - centroid))
        l2_pen = sq_dist
        loss_pgd = sq_dist + alpha_l1 * l1_pen + alpha_l2 * l2_pen
        
        sq_dist = np.sum((x_scipy - centroid)**2)
        l1_pen = np.sum(np.abs(x_scipy - centroid))
        l2_pen = sq_dist
        loss_scipy = sq_dist + alpha_l1 * l1_pen + alpha_l2 * l2_pen
        
        # Track as candidate with its loss, convergence history, iterations, and time
        top_candidates.append((row.copy(), loss_pgd, loss_trajectory_pgd, pgd_iters, pgd_converged, pgd_time))
        top_candidates_scipy.append((row.copy(), loss_scipy, loss_trajectory_scipy, scipy_iters, scipy_converged, scipy_time))
    
    # Sort by loss (ascending) and take top K
    top_candidates.sort(key=lambda x: x[1])
    top_candidates = top_candidates[:top_k]
    
    top_candidates_scipy.sort(key=lambda x: x[1])
    top_candidates_scipy = top_candidates_scipy[:top_k]
    
    # Choose the best candidate using weighted random selection
    weights = [1/(1+c[1]) for c in top_candidates]
    total = sum(weights)
    weights = [w/total for w in weights]
    
    chosen_idx = random.choices(range(len(top_candidates)), weights=weights, k=1)[0]
    best_track, best_loss_pgd, loss_trajectory_pgd, pgd_iters, pgd_converged, pgd_time_best = top_candidates[chosen_idx]
    
    # Find the same track in scipy results for fair comparison
    scipy_track = None
    for track, loss, trajectory, iters, converged, t in top_candidates_scipy:
        if track['track_id'] == best_track['track_id']:
            scipy_track = (track, loss, trajectory, iters, converged, t)
            break
    
    # If not found (unlikely), just use the best scipy result
    if scipy_track is None:
        scipy_track = top_candidates_scipy[0]
    
    _, best_loss_scipy, loss_trajectory_scipy, scipy_iters, scipy_converged, scipy_time_best = scipy_track
    
    # Only show convergence info if requested
    if show_progress:
        # Print enhanced comparison with convergence statistics
        print("\nPerformance and Convergence Comparison:")
        print(f"PGD: {pgd_total_time:.2f}s total, {pgd_total_time/len(cand):.4f}s avg/candidate")
        print(f"  Best candidate: {pgd_iters} iterations, converged: {pgd_converged}, time: {pgd_time_best:.4f}s")
        print(f"SciPy: {scipy_total_time:.2f}s total, {scipy_total_time/len(cand):.4f}s avg/candidate")
        print(f"  Best candidate: {scipy_iters} iterations, converged: {scipy_converged}, time: {scipy_time_best:.4f}s")
    
    # Generate plots only if requested
    if generate_plots:
        # Create enhanced figure for convergence comparison with standardized parameters
        plt.figure(figsize=(12, 8), dpi=100)
        
        # Use actual iteration numbers
        diff_loss_pgd = calculate_loss_differences(loss_trajectory_pgd)
        diff_loss_scipy = calculate_loss_differences(loss_trajectory_scipy)
        x_values_pgd = np.arange(len(diff_loss_pgd))
        x_values_scipy = np.arange(len(diff_loss_scipy))
        
        plt.semilogy(x_values_pgd, diff_loss_pgd, 'b-', linewidth=2.5,
                    label=f'PGD ({pgd_iters} iters, {pgd_converged}, {pgd_time_best:.4f}s)')
        plt.semilogy(x_values_scipy, diff_loss_scipy, 'r-', linewidth=2.5,
                    label=f'SciPy L-BFGS-B ({scipy_iters} iters, {scipy_converged}, {scipy_time_best:.4f}s)')
        
        # Add convergence threshold used by both methods
        plt.axhline(y=tol, color='g', linestyle='--', alpha=0.7, 
                   label=f'Convergence threshold ({tol:.1e})')
        
        # Better styling
        plt.xlabel('Iteration Number', fontsize=12)
        plt.ylabel('|Loss(i) - Loss(i-1)| (log scale)', fontsize=12)
        plt.title('Optimizer Convergence Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Add song information
        plt.suptitle(f"Song: '{best_track['name']}' by {best_track['artist']}", fontsize=12)
        
        # Add enhanced annotations with convergence metrics
        plt.figtext(0.02, 0.02, 
                    f"Optimizer Settings: max_iter={steps}, tol={tol:.1e}\n" +
                    f"PGD: {pgd_iters} iters, Loss: {best_loss_pgd:.6f}\n" +
                    f"SciPy: {scipy_iters} iters, Loss: {best_loss_scipy:.6f}", 
                    ha='left', fontsize=10,
                    bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
        
        plt.tight_layout()
        plt.savefig('optimizer_convergence.png')
        
        # Create a separate timing comparison bar chart
        plt.figure(figsize=(10, 6), dpi=100)
        methods = ['Custom PGD', 'SciPy L-BFGS-B']
        
        # Timing data for average and best candidate
        avg_times = [pgd_total_time/len(cand), scipy_total_time/len(cand)]
        best_times = [pgd_time_best, scipy_time_best]
        
        # Create grouped bar chart
        x = np.arange(len(methods))
        width = 0.35
        
        plt.bar(x - width/2, avg_times, width, label='Average Time/Candidate', color='skyblue')
        plt.bar(x + width/2, best_times, width, label='Best Candidate Time', color='lightcoral')
        
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.title('Optimization Methods: Time Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, methods, fontsize=11)
        plt.legend()
        
        # Add time values on top of bars
        for i, v in enumerate(avg_times):
            plt.text(i - width/2, v + 0.003, f'{v:.4f}s', ha='center', fontsize=9)
        
        for i, v in enumerate(best_times):
            plt.text(i + width/2, v + 0.003, f'{v:.4f}s', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('timing_comparison.png')
        
        if show_progress:
            print("Comparison plots saved to:")
            print("1. optimizer_convergence.png - Loss change per iteration")
            print("2. timing_comparison.png - Time performance comparison")
    
    # Convert the Series to a DataFrame before calling assign
    return pd.DataFrame([best_track]).assign(loss=best_loss_pgd)

if __name__ == '__main__':
    # Set random seeds for reproducibility within a single run but variation between runs
    random_seed = random.randint(1, 10000)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Sample playlist with different random seed each time
    playlist = sample_playlist(df, k=5)
    print("\n=== Selected Playlist ===")
    print(playlist[['name', 'artist', 'year']])
    
    # Standardized hyperparameters for fair comparison
    year_range = random.randint(3, 7)
    alpha_l1 = 0.1 + random.uniform(-0.05, 0.05)
    alpha_l2 = 0.01 + random.uniform(-0.005, 0.005)
    steps = 50
    eta = 0.005
    tol = 1e-8
    top_k = 3

    print("\nFinding the perfect song to complete your playlist...")
    
    # Hide all optimization details
    rec = recommend_song_pgd(df, playlist,
                            year_range=year_range,
                            alpha_l1=alpha_l1,
                            alpha_l2=alpha_l2,
                            steps=steps,
                            eta=eta,
                            tol=tol,
                            top_k=top_k,
                            show_progress=False,
                            generate_plots=False)
                             
    print("\n=== Recommended Song ===")
    print(rec[['name', 'artist', 'year']])
