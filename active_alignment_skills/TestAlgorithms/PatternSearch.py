import numpy as np
import time
from typing import Callable, Tuple, Optional

class HexapodPatternSearch:
    """
    Enhanced Pattern Search Algorithm optimized for hexapod edge coupling control.
    """
    
    def __init__(self, 
                 objective_func: Callable,
                 x0: np.ndarray,
                 initial_step_size: float = 2.0,
                 reduction_factor: float = 0.5,
                 expansion_factor: float = 1.2,
                 permitted_error: float = 1e-6,
                 min_improvement: float = 1e-8,
                 max_iter: int = 1000,
                 lower_bounds: Optional[np.ndarray] = None,
                 upper_bounds: Optional[np.ndarray] = None,
                 use_pattern_motion: bool = True,
                 use_adaptive_step: bool = True,
                 verbose: bool = True):
        """
        Initialize the pattern search optimizer.
        """
        self.objective_func = objective_func
        self.x_current = np.array(x0, dtype=float)
        self.n = len(x0)
        self.step_size = initial_step_size
        self.initial_step_size = initial_step_size
        self.reduction_factor = reduction_factor
        self.expansion_factor = expansion_factor
        self.permitted_error = permitted_error
        self.min_improvement = min_improvement
        self.max_iter = max_iter
        self.use_pattern_motion = use_pattern_motion
        self.use_adaptive_step = use_adaptive_step
        self.verbose = verbose
        
        # Physical constraints
        self.lower_bounds = lower_bounds if lower_bounds is not None else np.full(self.n, -np.inf)
        self.upper_bounds = upper_bounds if upper_bounds is not None else np.full(self.n, np.inf)
        
        # Unit vectors
        self.axes = np.eye(self.n)
        
        # Tracking variables
        self.x_prev = None
        self.x_pattern = None
        self.prev_value = None
        self.iteration = 0
        self.history = []
        self.consecutive_failures = 0
        
        # Performance metrics
        self.start_time = None
        self.evaluations = 0
        self.best_value = -np.inf
        self.best_point = None
        
    def apply_bounds(self, x: np.ndarray) -> np.ndarray:
        """Apply physical bounds to hexapod joint positions."""
        return np.clip(x, self.lower_bounds, self.upper_bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate objective function with bounds checking."""
        x_clipped = self.apply_bounds(x)
        self.evaluations += 1
        value = self.objective_func(x_clipped)
        
        # Track best value globally
        if value > self.best_value:
            self.best_value = value
            self.best_point = x_clipped.copy()
            
        return value
    
    def adaptive_step_size(self, improvement: float) -> float:
        """Adapt step size based on improvement magnitude."""
        if not self.use_adaptive_step:
            return self.step_size
            
        if improvement > 0.1:
            # Big improvement - accelerate but cap
            return min(self.step_size * self.expansion_factor, self.initial_step_size * 1.5)
        elif improvement > 0.01:
            # Moderate improvement - maintain
            return self.step_size
        else:
            # Little improvement - decelerate
            return max(self.step_size * self.reduction_factor, self.permitted_error)
    
    def log_progress(self):
        """Log current optimization progress."""
        current_value = self.evaluate(self.x_current)
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print(f"Iter {self.iteration:4d} | "
              f"Value: {current_value:10.4f} | "
              f"Step: {self.step_size:8.4f} | "
              f"Evals: {self.evaluations:4d} | "
              f"Time: {elapsed:.2f}s")
    
    def exploratory_moves(self) -> Tuple[np.ndarray, bool, float]:
        """
        Perform exploratory moves along all axes with improved strategy.
        
        Returns:
            Tuple of (new position, whether improvement was found, improvement value)
        """
        x_trial = self.x_current.copy()
        best_found = False
        total_improvement = 0
        
        # Shuffle axes order for better exploration
        axes_order = np.random.permutation(self.n)
        
        for j in axes_order:
            e_j = self.axes[j]
            
            # Evaluate current point
            f_current = self.evaluate(x_trial)
            
            # Try positive direction
            x_pos = x_trial + self.step_size * e_j
            f_pos = self.evaluate(x_pos)
            
            if f_pos > f_current:
                improvement = f_pos - f_current
                x_trial = x_pos
                total_improvement += improvement
                best_found = True
                continue
                
            # Try negative direction
            x_neg = x_trial - self.step_size * e_j
            f_neg = self.evaluate(x_neg)
            
            if f_neg > f_current:
                improvement = f_neg - f_current
                x_trial = x_neg
                total_improvement += improvement
                best_found = True
        
        return x_trial, best_found, total_improvement
    
    def pattern_motion(self, x_trial: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Perform pattern motion with improved strategy.
        """
        if not self.use_pattern_motion or self.x_prev is None:
            return x_trial, False
            
        # Calculate pattern step
        pattern_step = x_trial + (x_trial - self.x_prev)
        
        # Apply bounds and evaluate
        pattern_step = self.apply_bounds(pattern_step)
        f_pattern = self.evaluate(pattern_step)
        f_trial = self.evaluate(x_trial)
        
        if f_pattern > f_trial:
            # Pattern step successful - do additional exploration
            x_pattern = pattern_step.copy()
            
            # One more exploratory move around pattern point
            for j in range(self.n):
                e_j = self.axes[j]
                f_curr = self.evaluate(x_pattern)
                
                x_pos = x_pattern + self.step_size * e_j
                if self.evaluate(x_pos) > f_curr:
                    x_pattern = x_pos
                else:
                    x_neg = x_pattern - self.step_size * e_j
                    if self.evaluate(x_neg) > f_curr:
                        x_pattern = x_neg
            
            return x_pattern, True
        
        return x_trial, False
    
    def multi_start_search(self, num_starts: int = 3) -> Tuple[np.ndarray, dict]:
        """
        Perform multi-start optimization to avoid local optima.
        
        Args:
            num_starts: Number of random restarts
            
        Returns:
            Best solution found and statistics
        """
        best_solution = None
        best_stats = None
        best_value = -np.inf
        
        for start in range(num_starts):
            if self.verbose:
                print(f"\n--- Multi-start {start+1}/{num_starts} ---")
            
            # Random perturbation for restart
            if start > 0:
                noise = np.random.uniform(-5, 5, self.n)
                self.x_current = np.clip(self.x_current + noise, 
                                        self.lower_bounds, 
                                        self.upper_bounds)
                self.step_size = self.initial_step_size
                self.x_prev = None
                self.iteration = 0
                self.history = []
            
            # Run optimization
            solution, stats = self.optimize()
            
            if stats['final_value'] > best_value:
                best_value = stats['final_value']
                best_solution = solution
                best_stats = stats
        
        return best_solution, best_stats
    
    def optimize(self) -> Tuple[np.ndarray, dict]:
        """
        Main optimization loop with improved convergence criteria.
        """
        self.start_time = time.time()
        self.evaluations = 0
        self.best_value = -np.inf
        
        # Initial evaluation
        initial_value = self.evaluate(self.x_current)
        self.prev_value = initial_value
        self.x_prev = self.x_current.copy()
        self.best_point = self.x_current.copy()
        self.best_value = initial_value
        
        if self.verbose:
            print("=" * 80)
            print(f"Pattern Search Optimization")
            print(f"Dimensions: {self.n}")
            print(f"Initial value: {initial_value:.6f}")
            print(f"Initial step: {self.step_size}")
            print("=" * 80)
            self.log_progress()
        
        # Track best values for convergence check
        best_values = [initial_value]
        stagnation_counter = 0
        
        while self.step_size >= self.permitted_error and self.iteration < self.max_iter:
            self.iteration += 1
            
            # Step 2-4: Exploratory moves
            x_trial, improved, total_improvement = self.exploratory_moves()
            
            # Check if exploratory moves found improvement
            f_new = self.evaluate(x_trial)
            f_old = self.evaluate(self.x_current)
            
            if f_new > f_old:
                # Successful detection
                improvement = f_new - f_old
                
                # Store previous best point
                self.x_prev = self.x_current.copy()
                self.x_current = x_trial.copy()
                self.consecutive_failures = 0
                
                # Step 6: Pattern motion
                pattern_success = False
                if self.iteration >= 2:
                    x_pattern, pattern_success = self.pattern_motion(x_trial)
                    if pattern_success:
                        self.x_current = x_pattern.copy()
                
                # Adapt step size
                if self.use_adaptive_step:
                    self.step_size = self.adaptive_step_size(improvement)
                
                # Check for convergence based on stagnation
                best_values.append(self.best_value)
                if len(best_values) > 20:
                    best_values = best_values[-20:]
                    # Check if we've made significant progress in last 20 iterations
                    if max(best_values) - min(best_values) < self.min_improvement * 10:
                        stagnation_counter += 1
                        if stagnation_counter >= 3:
                            if self.verbose:
                                print("Converged: Stagnation detected")
                            break
                    else:
                        stagnation_counter = 0
                
                if self.verbose and self.iteration % 5 == 0:
                    self.log_progress()
                    
            else:
                # Detection failed
                self.consecutive_failures += 1
                
                # More aggressive step reduction after multiple failures
                if self.consecutive_failures >= 3:
                    self.step_size = self.step_size * 0.3  # More aggressive reduction
                    self.consecutive_failures = 0
                else:
                    self.step_size = self.step_size * self.reduction_factor
            
            # Store history
            self.history.append({
                'iteration': self.iteration,
                'position': self.x_current.copy(),
                'value': f_new,
                'step_size': self.step_size,
                'best_value': self.best_value
            })
        
        # Final results
        final_value = self.evaluate(self.x_current)
        elapsed = time.time() - self.start_time
        
        if self.verbose:
            print("=" * 80)
            print(f"Optimization Complete!")
            print(f"Iterations: {self.iteration}")
            print(f"Evaluations: {self.evaluations}")
            print(f"Final value: {final_value:.6f}")
            print(f"Best value: {self.best_value:.6f}")
            print(f"Final step: {self.step_size:.4f}")
            print("=" * 80)
        
        stats = {
            'iterations': self.iteration,
            'evaluations': self.evaluations,
            'time': elapsed,
            'initial_value': initial_value,
            'final_value': final_value,
            'best_value': self.best_value,
            'improvement': final_value - initial_value,
            'final_step_size': self.step_size,
            'history': self.history
        }
        
        # Return best point found
        if self.best_value > final_value:
            return self.best_point, stats
        return self.x_current, stats


def create_objective_function(true_optimal: np.ndarray, noise_level: float = 0.0, 
                              coupling_width: float = 50.0):
    """
    Create a more realistic objective function for edge coupling.
    """
    def objective_func(x: np.ndarray) -> float:
        # Gaussian coupling efficiency
        distance_sq = np.sum((x - true_optimal) ** 2)
        
        # More realistic coupling profile (wider peak)
        efficiency = 100.0 * np.exp(-distance_sq / (2 * coupling_width ** 2))
        
        # Add small noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * efficiency / 100)
            efficiency = max(0, efficiency + noise)
        
        return efficiency
    
    return objective_func


def test_hexapod_optimization():
    """Test the improved pattern search on a simulated hexapod system."""
    
    dimensions = 6
    lower_bounds = np.array([-30, -30, -30, -30, -30, -30])
    upper_bounds = np.array([30, 30, 30, 30, 30, 30])
    
    # Improved algorithm parameters
    initial_step = 3.0  # Reduced from 5.0
    alpha = 0.5
    epsilon_0 = 0.05  # Smaller minimum step
    min_improvement = 0.01  # More realistic threshold
    
    num_runs = 10
    results = []
    
    print("\n" + "=" * 80)
    print("IMPROVED HEXAPOD EDGE COUPLING OPTIMIZATION TEST")
    print("=" * 80)
    
    for run in range(num_runs):
        print(f"\n--- RUN {run + 1}/{num_runs} ---")
        
        # Generate random true optimal position
        true_optimal = np.random.uniform(-20, 20, dimensions)
        
        # Generate random starting position
        start_position = np.random.uniform(-15, 15, dimensions)
        
        # Create objective function with realistic parameters
        objective_func = create_objective_function(
            true_optimal, 
            noise_level=0.1,  # Reduced noise
            coupling_width=30.0  # Wider coupling peak
        )
        
        # Initialize optimizer with improvements
        optimizer = HexapodPatternSearch(
            objective_func=objective_func,
            x0=start_position,
            initial_step_size=initial_step,
            reduction_factor=alpha,
            expansion_factor=1.1,  # Smaller expansion
            permitted_error=epsilon_0,
            min_improvement=min_improvement,
            max_iter=300,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            use_pattern_motion=True,
            use_adaptive_step=True,
            verbose=False
        )
        
        # Use multi-start for better results
        optimal_point, stats = optimizer.multi_start_search(num_starts=1)
        
        # Calculate error
        error = np.linalg.norm(optimal_point - true_optimal)
        final_efficiency = objective_func(optimal_point)
        
        results.append({
            'run': run + 1,
            'true_optimal': true_optimal,
            'start_position': start_position,
            'optimal_point': optimal_point,
            'error': error,
            'iterations': stats['iterations'],
            'evaluations': stats['evaluations'],
            'final_efficiency': final_efficiency,
            'time': stats['time']
        })
        
        print(f"True optimal: {true_optimal}")
        print(f"Found point:  {optimal_point}")
        print(f"Error: {error:.4f} units")
        print(f"Final efficiency: {final_efficiency:.2f}%")
        print(f"Iterations: {stats['iterations']}")
        print(f"Evaluations: {stats['evaluations']}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    errors = [r['error'] for r in results]
    iterations = [r['iterations'] for r in results]
    efficiencies = [r['final_efficiency'] for r in results]
    times = [r['time'] for r in results]
    
    print(f"Average error: {np.mean(errors):.4f} ± {np.std(errors):.4f} units")
    print(f"Average iterations: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")
    print(f"Average evaluations: {np.mean([r['evaluations'] for r in results]):.1f}")
    print(f"Average efficiency: {np.mean(efficiencies):.2f}% ± {np.std(efficiencies):.2f}%")
    print(f"Average time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
    
    success_rate = np.sum(np.array(errors) < 1.0) / num_runs * 100
    print(f"Success rate (error < 1.0): {success_rate:.1f}%")
    
    return results


if __name__ == "__main__":
    results = test_hexapod_optimization()