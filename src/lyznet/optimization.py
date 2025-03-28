import numpy as np
from scipy.optimize import linprog
import sympy as sp
import lyznet
from itertools import combinations
import time


def LP_global_lf_synthesis(system, terms, num_samples=500, 
                           symbolic_reduction=False, 
                           CEGIS=False, max_iters=10,
                           return_weak=False, LaSalle=False,
                           time_out=60):
    """
    LP-based Global Lyapunov function synthesis with an optional CEGIS loop.

    Args:
    - system: The dynamical system, containing symbolic_vars (state variables) and symbolic_f (dynamics).
    - terms: List of SymPy expressions representing the monomial terms for the Lyapunov candidate.
    - num_samples: Number of random samples in the state space for constraint generation.
    - symbolic_reduction: Boolean indicating if we want to apply symbolic reduction on the terms.
    - CEGIS: Boolean indicating if CEGIS should be used for iterative synthesis.
    - max_iters: Maximum number of CEGIS iterations.

    Returns:
    - V_solution: The Lyapunov function candidate if synthesis is successful, else None.
    """
    # Generate symbolic function list based on reduction setting
    print('_' * 50)
    if symbolic_reduction:
        print(f"Computing Lyapunov condition with LP-CEGIS-SR and {num_samples} samples:")
        sympy_func_list, _ = lyznet.generate_lyap_template(system, terms)

    else:
        sympy_func_list = terms
        print(f"Computing Lyapunov condition with LP-CEGIS and {num_samples} samples:")

    start_time = time.time()

    # Generate random samples within the specified domain
    samples = generate_random_samples(system.domain, num_samples=num_samples)

    # Initialize lists for LP constraints
    A_ub = []  # Coefficients matrix for inequality constraints
    b_ub = []  # Right-hand side values for inequalities

    # Precompute gradients and functions
    grad_func_list = [sp.Matrix([sp.diff(term, var) for var in system.symbolic_vars]).dot(system.symbolic_f) for term in sympy_func_list]
    grad_numpy_func_list = [sp.lambdify(system.symbolic_vars, grad_term, "numpy") for grad_term in grad_func_list]
    numpy_func_list = [sp.lambdify(system.symbolic_vars, term, "numpy") for term in sympy_func_list]

    # Calculate degrees of terms
    grad_degrees = [max(term.as_poly().degree_list()) for term in grad_func_list]
    
    min_DV_degree = min(grad_degrees)
    max_DV_degree = max(grad_degrees)

    # print("min_DV_degree: ", min_DV_degree)
    # print("max_DV_degree: ", max_DV_degree)

    monomial_degrees = [
        sum(monomial.as_poly().degree_list()) 
        for term in sympy_func_list 
        for monomial in term.as_ordered_terms()
    ]

    min_V_degree = min(monomial_degrees)
    max_V_degree = max(monomial_degrees)

    # print("min_V_degree: ", min_V_degree)
    # print("max_V_degree: ", max_V_degree)

    # Construct LP constraints for each sample
    def add_constraints_for_sample(sample):
        V_constraints = []   # V(x) > 0 constraints
        LfV_constraints = [] # ∇V · f < 0 constraints
        norm_x_squared = sum(val**2 for val in sample)
        mu = 0.01

        # Calculate bound values for V and ∇V · f separately
        bound_V_value = mu * min(norm_x_squared**(min_V_degree / 2), norm_x_squared**(max_V_degree / 2))
        
        if return_weak or LaSalle:
            bound_DV_value = 0
        else:
            bound_DV_value = mu * min(norm_x_squared**(min_DV_degree / 2), norm_x_squared**(max_DV_degree / 2))

        for np_func, grad_np_func in zip(numpy_func_list, grad_numpy_func_list):
            v_i_val = np_func(*sample)
            grad_v_i_f_val = grad_np_func(*sample)
            V_constraints.append(v_i_val)
            LfV_constraints.append(grad_v_i_f_val)

        A_ub.append([-val for val in V_constraints])
        b_ub.append(-bound_V_value)
        A_ub.append([val for val in LfV_constraints])
        b_ub.append(-bound_DV_value)

    for sample in samples:
        add_constraints_for_sample(sample)

    if CEGIS and return_weak:
        LaSalle = True

    # CEGIS loop        
    for iteration in range(max_iters if CEGIS else 1):
        # Define the LP objective as zero (feasibility problem)
        c = [0] * len(sympy_func_list)

        print("Solving LP ... ")
        lyznet.tik()
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
        lyznet.tok()

        if res.success:
            # with rounding (to two decimal points)            
            # solution = {coeff: sp.Rational(str(round(res.x[i], 2))) for i, coeff in enumerate(sympy_func_list)}
            
            # without rounding 
            solution = {coeff: sp.Rational(str(res.x[i])) for i, coeff in enumerate(sympy_func_list)}
            
            V_solution = sum(c * term for c, term in zip(solution.values(), sympy_func_list))
            print("Candidate Lyapunov function computed with LP.")
            print("V = ", V_solution)

            # If CEGIS is enabled, verify the solution with the global LF verifier
            if CEGIS:
                verified, counterexample = lyznet.z3_global_lf_verifier(system, V_sympy=V_solution, LaSalle=LaSalle, time_out=time_out)
                if verified:
                    print(f"Feasible Lyapunov function found after iteration {iteration + 1}:", V_solution)
                    end_time = time.time()
                    print("Elapsed time:", end_time - start_time, "seconds")
                    return V_solution
                elif counterexample:
                    print(f"Counterexample found in iteration {iteration + 1}: {counterexample}")
                    add_constraints_for_sample(counterexample)
            else:
                print("Feasible Lyapunov function found:", V_solution)
                end_time = time.time()
                print("Elapsed time:", end_time - start_time, "seconds")
                return V_solution
        else:
            print("No feasible solution found.")
            return None

    print("No feasible Lyapunov function found after CEGIS iterations.")
    return None


def generate_random_samples(domain, num_samples=50, num_subspace_samples=1):
    """
    Generates random samples within the specified domain, including samples 
    with specific subspaces where at least one dimension vanishes.

    Args:
        domain: List of (min, max) tuples for each dimension.
        num_samples: Number of fully random samples in the state space.
        num_subspace_samples: Number of samples to generate for each subspace.

    Returns:
        List of sample points in the domain.
    """
    d = len(domain)  # Dimension of the domain
    samples = []

    # Fully random samples
    for _ in range(num_samples):
        sample = [np.random.uniform(bounds[0], bounds[1]) for bounds in domain]
        samples.append(sample)

    # Generate samples for each subspace
    for k in range(1, d):
        for zero_dims in combinations(range(d), k):
            for _ in range(num_subspace_samples):
                sample = [np.random.uniform(bounds[0], bounds[1]) if i not in zero_dims else 0 for i, bounds in enumerate(domain)]
                samples.append(sample)

    return samples
