#!/usr/bin/env python3
"""
Production Optimization Example
Using Π-Topology to find optimal production plans
"""

import numpy as np
from pi_topology import PiFlow, Config, pi_merge, pi_involution
import time

def main():
    """Main example: Production optimization"""
    print("=" * 70)
    print("PRODUCTION OPTIMIZATION WITH Π-TOPOLOGY")
    print("=" * 70)
    
    # Configure for GPU if available
    config = Config(device='auto', dtype='float32', verbose=True)
    
    # Define product ranges
    # Product A: 0 to 100 units
    # Product B: 0 to 100 units
    a_domain = np.linspace(0, 100, 500, dtype=np.float32)
    b_domain = np.linspace(0, 100, 500, dtype=np.float32)
    
    print(f"\nSearch space: {len(a_domain)} × {len(b_domain)} = {len(a_domain)*len(b_domain):,} possibilities")
    
    start_time = time.time()
    
    # Create Π-Flows for constraints
    print("\n1. Creating constraint flows...")
    
    # Constraint 1: Resource usage (2A + 3B ≤ 200)
    resources = PiFlow("Resources", config)
    resources.create_from_2d_domain(
        a_domain, b_domain,
        lambda A, B: 2*A + 3*B
    )
    
    # Constraint 2: Time usage (A + 2B ≤ 150)
    time_constraint = PiFlow("Time", config)
    time_constraint.create_from_2d_domain(
        a_domain, b_domain,
        lambda A, B: A + 2*B
    )
    
    # Objective: Profit (5A + 4B, maximize)
    profit = PiFlow("Profit", config)
    profit.create_from_2d_domain(
        a_domain, b_domain,
        lambda A, B: 5*A + 4*B
    )
    
    # Merge constraints
    print("\n2. Merging constraints...")
    merged = pi_merge(resources, time_constraint, operation='concat')
    
    # Apply constraints (Π-Involution)
    print("\n3. Applying constraints (Π-Involution)...")
    constraints = [
        {'type': 'less', 'target': 200},  # Resource constraint
        {'type': 'less', 'target': 150},  # Time constraint
    ]
    
    result = pi_involution(merged, constraints)
    
    # Get feasible solutions
    feasible_solutions = result.get_solutions()
    
    if len(feasible_solutions) == 0:
        print("\nNo feasible solutions found!")
        return
    
    total_time = time.time() - start_time
    
    # Find optimal profit
    print("\n4. Finding optimal solution...")
    A_vals = feasible_solutions[:, 0]
    B_vals = feasible_solutions[:, 1]
    profits = 5 * A_vals + 4 * B_vals
    
    opt_idx = np.argmax(profits)
    opt_A = A_vals[opt_idx]
    opt_B = B_vals[opt_idx]
    opt_profit = profits[opt_idx]
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Calculation time: {total_time:.3f} seconds")
    print(f"Feasible solutions: {len(feasible_solutions):,}")
    print(f"Feasibility ratio: {len(feasible_solutions)/len(a_domain)/len(b_domain)*100:.1f}%")
    
    print(f"\nOPTIMAL PRODUCTION PLAN:")
    print(f"  Product A: {opt_A:.1f} units")
    print(f"  Product B: {opt_B:.1f} units")
    print(f"  Expected profit: ${opt_profit:.2f}")
    
    print(f"\nCONSTRAINT CHECK:")
    print(f"  Resources used: {2*opt_A + 3*opt_B:.1f} (limit: 200)")
    print(f"  Time used: {opt_A + 2*opt_B:.1f} (limit: 150)")
    
    # Show top 5 alternatives
    print(f"\nTOP 5 ALTERNATIVES:")
    top_5_idx = np.argsort(profits)[-5:][::-1]
    for i, idx in enumerate(top_5_idx):
        A = A_vals[idx]
        B = B_vals[idx]
        profit_val = profits[idx]
        resource_use = 2*A + 3*B
        time_use = A + 2*B
        
        print(f"  {i+1}. A={A:.1f}, B={B:.1f}, "
              f"Profit=${profit_val:.1f}, "
              f"Resources={resource_use:.1f}, "
              f"Time={time_use:.1f}")
    
    # Statistics
    stats = result.get_statistics()
    print(f"\nPERFORMANCE STATISTICS:")
    print(f"  Creation time: {stats['creation_time']:.3f}s")
    print(f"  Merge time: {stats['merge_time']:.3f}s")
    print(f"  Involution time: {stats['involution_time']:.3f}s")
    
    if config.device == 'cuda':
        print(f"  GPU memory: {stats.get('gpu_memory_allocated', 0):.1f} MB")
    
    # Export results
    print(f"\nResults exported to 'production_solutions.npy'")
    np.save('production_solutions.npy', feasible_solutions)
    
    # Create visualization if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        # Plot feasible region
        plt.subplot(1, 2, 1)
        plt.scatter(feasible_solutions[:, 0], feasible_solutions[:, 1], 
                   c='blue', alpha=0.1, s=1, label='Feasible')
        plt.scatter([opt_A], [opt_B], c='red', s=100, marker='*', 
                   label='Optimal', edgecolors='black')
        
        # Constraints lines
        A_line = np.linspace(0, 100, 100)
        B_resources = (200 - 2*A_line) / 3  # 2A + 3B = 200
        B_time = (150 - A_line) / 2        # A + 2B = 150
        
        plt.plot(A_line, B_resources, 'r--', alpha=0.5, label='Resource limit')
        plt.plot(A_line, B_time, 'g--', alpha=0.5, label='Time limit')
        
        plt.xlabel('Product A (units)')
        plt.ylabel('Product B (units)')
        plt.title('Feasible Production Region')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot profit distribution
        plt.subplot(1, 2, 2)
        plt.hist(profits, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(x=opt_profit, color='red', linestyle='--', 
                   label=f'Optimal: ${opt_profit:.1f}')
        plt.xlabel('Profit ($)')
        plt.ylabel('Frequency')
        plt.title('Profit Distribution of Feasible Plans')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('production_optimization.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to 'production_optimization.png'")
        
    except ImportError:
        print("Install matplotlib for visualization: pip install matplotlib")

if __name__ == "__main__":
    main()