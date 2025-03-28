# import z3
import lyznet
import sympy as sp
import numpy as np
from itertools import product
from collections import defaultdict


def z3_global_quadratic_synthesis(system, P_shape='diagonal'):
    x = sp.Matrix(system.symbolic_vars)
    num_vars = len(system.symbolic_vars)

    if P_shape == 'diagonal':
        P_vars = sp.symbols(f'p1:{num_vars+1}')
        P = sp.diag(*P_vars)
        z3_P_vars = [z3.Real(f"p{i+1}") for i in range(num_vars)]
        P_pos_def_condition = z3.And(*[p > 0 for p in z3_P_vars])

    elif P_shape == 'full':
        P_vars = sp.symbols(f'p1:{(num_vars*(num_vars+1))//2+1}')
        P = sp.Matrix(num_vars, num_vars, lambda i, j: P_vars[min(i, j) * num_vars + max(i, j) - (min(i, j) * (min(i, j) + 1)) // 2])
        z3_P_vars = [z3.Real(f"p{i+1}") for i in range(len(P_vars))]

        # Calculate leading minors for full matrices in general dimension
        leading_minors = []
        for i in range(1, num_vars+1):
            minor = P[:i, :i].det()
            leading_minors.append(minor)

        # Lambdify the leading minors
        leading_minors_lambdified = [sp.lambdify(tuple(list(P_vars)), minor, 'sympy') for minor in leading_minors]

        # Convert the lambdified minors to Z3 expressions
        leading_minors_z3 = [minor_lambdified(*z3_P_vars) for minor_lambdified in leading_minors_lambdified]

        # Ensure that all leading minors are positive
        P_pos_def_condition = z3.And(*[minor > 0 for minor in leading_minors_z3])

    else:
        raise ValueError("Invalid P_shape. Choose either 'diagonal' or 'full'.")

    V = x.T * P * x
    V = V[0]  
    print('_' * 50)
    print("Computing Lyapunov condition with Z3:")
    print("V(x) = ", sp.simplify(V))

    grad_V = sp.Matrix([sp.diff(V, var) for var in system.symbolic_vars])
    LfV = grad_V.dot(system.symbolic_f)

    z3_x = [z3.Real(f"x{i+1}") for i in range(num_vars)]
    z3_subs = {system.symbolic_vars[i]: z3_x[i] for i in range(num_vars)}

    V_lambdified = sp.lambdify(tuple(system.symbolic_vars + list(P_vars)), V, 'sympy')
    LfV_lambdified = sp.lambdify(tuple(system.symbolic_vars + list(P_vars)), LfV, 'sympy')

    V_z3 = V_lambdified(*[z3_subs[var] for var in system.symbolic_vars], *z3_P_vars)
    LfV_z3 = LfV_lambdified(*[z3_subs[var] for var in system.symbolic_vars], *z3_P_vars)

    z3_x_non_zero = z3.Or(*[xi != 0 for xi in z3_x])
    lf_condition = z3.Implies(z3_x_non_zero, LfV_z3 < 0)

    solver = z3.Solver()
    solver.add(P_pos_def_condition)
    solver.add(z3.ForAll(z3_x, lf_condition))

    lyznet.tik()

    if solver.check() == z3.sat:
        print("A Lyapunov function exists for the system!")
        model = solver.model()
        print("Model found for P:")
        if P_shape == 'diagonal':
            for i, var in enumerate(z3_P_vars):
                print(f"p{i+1} = {model[var]}")
        elif P_shape == 'full':
            P_full_extracted = sp.Matrix(num_vars, num_vars, lambda i, j: model[z3_P_vars[min(i, j) * num_vars + max(i, j) - (min(i, j) * (min(i, j) + 1)) // 2]])
            print("Full matrix P extracted:")
            sp.pprint(P_full_extracted)

        P_sympy = P.subs({P_vars[i]: float(model[z3_P_vars[i]].as_decimal(5)) 
                         for i in range(len(P_vars))})
    else:
        print("No Lyapunov function found. The global condition does not hold.")
        P_sympy = None

    lyznet.tok()
    return P_sympy


def z3_global_quadratic_clf_synthesis(system, P_shape='diagonal'):
    x = sp.Matrix(system.symbolic_vars)
    num_vars = len(system.symbolic_vars)

    if P_shape == 'diagonal':
        P_vars = sp.symbols(f'p1:{num_vars+1}')
        P = sp.diag(*P_vars)
        z3_P_vars = [z3.Real(f"p{i+1}") for i in range(num_vars)]
        P_pos_def_condition = z3.And(*[p > 0 for p in z3_P_vars])

    elif P_shape == 'full':
        P_vars = sp.symbols(f'p1:{(num_vars*(num_vars+1))//2+1}')
        P = sp.Matrix(num_vars, num_vars, lambda i, j: P_vars[min(i, j) * num_vars + max(i, j) - (min(i, j) * (min(i, j) + 1)) // 2])
        z3_P_vars = [z3.Real(f"p{i+1}") for i in range(len(P_vars))]

        leading_minors = []
        for i in range(1, num_vars+1):
            minor = P[:i, :i].det()
            leading_minors.append(minor)

        leading_minors_lambdified = [sp.lambdify(tuple(list(P_vars)), minor, 'sympy') for minor in leading_minors]
        leading_minors_z3 = [minor_lambdified(*z3_P_vars) for minor_lambdified in leading_minors_lambdified]
        P_pos_def_condition = z3.And(*[minor > 0 for minor in leading_minors_z3])

    else:
        raise ValueError("Invalid P_shape. Choose either 'diagonal' or 'full'.")

    V = x.T * P * x
    V = V[0]  
    print('_' * 50)
    print("Computing Control Lyapunov Function condition with Z3:")
    print("V(x) = ", sp.simplify(V))

    grad_V = sp.Matrix([sp.diff(V, var) for var in system.symbolic_vars])
    LfV = grad_V.dot(system.symbolic_f)
    LgV = grad_V.dot(system.symbolic_g)

    z3_x = [z3.Real(f"x{i+1}") for i in range(num_vars)]
    z3_subs = {system.symbolic_vars[i]: z3_x[i] for i in range(num_vars)}

    V_lambdified = sp.lambdify(tuple(system.symbolic_vars + list(P_vars)), V, 'sympy')
    LfV_lambdified = sp.lambdify(tuple(system.symbolic_vars + list(P_vars)), LfV, 'sympy')
    LgV_lambdified = sp.lambdify(tuple(system.symbolic_vars + list(P_vars)), LgV, 'sympy')

    V_z3 = V_lambdified(*[z3_subs[var] for var in system.symbolic_vars], *z3_P_vars)
    LfV_z3 = LfV_lambdified(*[z3_subs[var] for var in system.symbolic_vars], *z3_P_vars)
    LgV_z3_list = [LgV_lambdified(*[z3_subs[var] for var in system.symbolic_vars], *z3_P_vars)]

    LgV_zero_z3 = z3.And(*[expr == 0 for expr in LgV_z3_list])
    z3_x_non_zero = z3.Or(*[xi != 0 for xi in z3_x])
    clf_condition = z3.Implies(z3.And(LgV_zero_z3, z3_x_non_zero), LfV_z3 < 0)

    solver = z3.Solver()
    solver.add(P_pos_def_condition)
    solver.add(z3.ForAll(z3_x, clf_condition))

    lyznet.tik()

    if solver.check() == z3.sat:
        print("A Control Lyapunov function exists for the system!")
        model = solver.model()
        print("Model found for P:")
        if P_shape == 'diagonal':
            for i, var in enumerate(z3_P_vars):
                print(f"p{i+1} = {model[var]}")
        elif P_shape == 'full':
            P_full_extracted = sp.Matrix(num_vars, num_vars, lambda i, j: model[z3_P_vars[min(i, j) * num_vars + max(i, j) - (min(i, j) * (min(i, j) + 1)) // 2]])
            print("Full matrix P extracted:")
            sp.pprint(P_full_extracted)

        P_sympy = P.subs({P_vars[i]: float(model[z3_P_vars[i]].as_decimal(5)) 
                         for i in range(len(P_vars))})
    else:
        print("No Control Lyapunov function found. The global condition does not hold.")
        P_sympy = None

    lyznet.tok()
    return P_sympy


def z3_constraints_elimination(system, terms, heuristic=False):
    """
    Iteratively generate and solve constraints by identifying terms that should vanish to ensure negative definiteness,
    updating the Lyapunov candidate function V until no further constraints can be solved.
    
    Args:
    - system: The dynamical system, containing symbolic_vars (state variables) and symbolic_f (dynamics).
    - terms: List of SymPy expressions representing the monomial terms for Lyapunov candidate.
    
    Returns:
    - solved_coeffs: Dictionary of coefficients with solved values after all iterations.
    """
    x = sp.Matrix(system.symbolic_vars)
    coeffs = sp.symbols(f'c0:{len(terms)}')
    V = sum(c * term for c, term in zip(coeffs, terms))
    solved_coeffs = {}

    print("Initial Candidate V before reduction:")
    print("V(x) =", V)
    lyznet.tik()

    while True:
        # Substitute any previously solved coefficients into V
        V = V.subs(solved_coeffs)

        # Compute the Lie derivative ∇V * f
        grad_V = sp.Matrix([sp.diff(V, var) for var in x])
        LfV = grad_V.dot(system.symbolic_f)
        LfV_expanded = sp.expand(LfV)

        # Initialize a dictionary to collect each monomial and its associated coefficient
        LfV_collected = {}
        for term in LfV_expanded.as_ordered_terms():
            coeff, monomial = term.as_independent(*x, as_Add=False)
            if monomial != 0 and coeff != 0:
                if monomial in LfV_collected:
                    LfV_collected[monomial] += coeff
                else:
                    LfV_collected[monomial] = coeff

        # # Display organized terms
        # print("\nLie derivative LfV, organized by monomial terms with coefficients:")
        # for monomial, coefficient in LfV_collected.items():
        #     print(f"{monomial}: {coefficient}")

        # Generate constraints for the highest odd-degree terms
        monomial_degrees = {monomial: sum(monomial.as_poly().degree_list()) for monomial in LfV_collected}
        # max_degree = max(monomial_degrees.values())
        # min_degree = min(monomial_degrees.values())

        if monomial_degrees:  # Ensure non-empty before calling max and min
            max_degree = max(monomial_degrees.values())
            min_degree = min(monomial_degrees.values())
        else:
            # print("All monomials have been eliminated, resulting in an empty Lyapunov candidate.")
            return solved_coeffs

        constraints = []
        for monomial, coefficient in LfV_collected.items():
            monomial_degree = monomial_degrees[monomial]
            # Apply constraints based on heuristic flag
            if heuristic:
                # If heuristic is enabled, set all odd degree terms to zero
                if monomial_degree % 2 != 0:
                    constraints.append(sp.Eq(coefficient, 0))
            else:
                # If heuristic is disabled, set only the highest and lowest odd degree terms to zero
                if (monomial_degree == max_degree or monomial_degree == min_degree) and monomial_degree % 2 != 0:
                    constraints.append(sp.Eq(coefficient, 0))

        # Step 1: Identify constraints for even degrees on variables as needed
        even_filtered_monomials = {monomial: coeff for monomial, coeff in LfV_collected.items()
                                   if monomial_degrees[monomial] in {max_degree, min_degree} and monomial_degrees[monomial] % 2 == 0}

        variable_monomials = defaultdict(lambda: {'highest': [], 'lowest': []})
        for monomial, coefficient in even_filtered_monomials.items():
            for i, var in enumerate(system.symbolic_vars):
                if monomial.has(var):
                    degree = monomial.as_poly().degree(var)
                    if monomial_degrees[monomial] == max_degree:
                        variable_monomials[var]['highest'].append((monomial, coefficient, degree))
                    elif monomial_degrees[monomial] == min_degree:
                        variable_monomials[var]['lowest'].append((monomial, coefficient, degree))

        for var, monomials in variable_monomials.items():
            if monomials['highest']:
                max_var_degree = max(degree for _, _, degree in monomials['highest'])
                for monomial, coeff, degree in monomials['highest']:
                    if degree == max_var_degree and degree % 2 != 0:
                        constraints.append(sp.Eq(coeff, 0))
            if monomials['lowest']:
                max_var_degree = max(degree for _, _, degree in monomials['lowest'])
                for monomial, coeff, degree in monomials['lowest']:
                    if degree == max_var_degree and degree % 2 != 0:
                        constraints.append(sp.Eq(coeff, 0))

        # Step 2: Additional highest-degree odd constraints for individual factors across all monomials
        all_variable_monomials = defaultdict(list)
        for monomial, coeff in LfV_collected.items():
            for var in system.symbolic_vars:
                if monomial.has(var):
                    degree = monomial.as_poly().degree(var)
                    all_variable_monomials[var].append((monomial, coeff, degree))

        for var, monomials in all_variable_monomials.items():
            max_degree = max(degree for _, _, degree in monomials)
            for monomial, coeff, degree in monomials:
                if degree == max_degree and degree % 2 != 0:
                    constraints.append(sp.Eq(coeff, 0))
                    # print(f"Added constraint: {coeff} = 0 for highest odd degree term {monomial} (degree {degree}) of variable {var}")

        for monomial, coeff in LfV_collected.items():
            if (monomial.is_Pow or monomial.is_Symbol) and monomial.has(var):
                degree = monomial.as_poly().degree(var)

                # Check if the degree is both the highest and lowest for the variable
                max_var_degree = max(degree for _, coeff, degree in all_variable_monomials[var])
                min_var_degree = min(degree for _, coeff, degree in all_variable_monomials[var])

                if degree == max_var_degree or degree == min_var_degree:
                    if degree % 2 != 0:
                        constraints.append(sp.Eq(coeff, 0))
                    else:
                        constraints.append(coeff <= 0)

        # print("constraints: ", constraints)

        # Separate equalities and inequalities
        equality_constraints = [c for c in constraints if c.is_Equality]
        inequality_constraints = [c for c in constraints if not c.is_Equality]

        # Solve equality constraints initially
        new_solved_coeffs = sp.solve(equality_constraints, dict=True)
        if isinstance(new_solved_coeffs, list):
            new_solved_coeffs = new_solved_coeffs[0] if new_solved_coeffs else {}

        # Convert numerical solutions to exact symbolic form
        new_solved_coeffs = {coeff: sp.nsimplify(value, rational=True) for coeff, value in new_solved_coeffs.items()}
        
        # Update solved coefficients and substitute into inequalities
        solved_coeffs.update(new_solved_coeffs)
        reduced_inequalities = [ineq.subs(solved_coeffs).simplify() for ineq in inequality_constraints]

        # Check for inferred equalities from reduced inequalities
        inferred_equalities = []
        for i in range(len(reduced_inequalities)):
            for j in range(i + 1, len(reduced_inequalities)):
                # Look for opposite inequalities, e.g., c <= 0 and c >= 0 imply c = 0
                if isinstance(reduced_inequalities[i], sp.LessThan) and isinstance(reduced_inequalities[j], sp.GreaterThan):
                    if reduced_inequalities[i].lhs == reduced_inequalities[j].lhs:
                        inferred_equalities.append(sp.Eq(reduced_inequalities[i].lhs, 0))
                elif isinstance(reduced_inequalities[i], sp.GreaterThan) and isinstance(reduced_inequalities[j], sp.LessThan):
                    if reduced_inequalities[i].lhs == reduced_inequalities[j].lhs:
                        inferred_equalities.append(sp.Eq(reduced_inequalities[i].lhs, 0))

        # Solve inferred equalities and update solved coefficients
        if inferred_equalities:
            inferred_solutions = sp.solve(inferred_equalities, dict=True)
            if isinstance(inferred_solutions, list):
                inferred_solutions = inferred_solutions[0] if inferred_solutions else {}
            inferred_solutions = {coeff: sp.nsimplify(value, rational=True) for coeff, value in inferred_solutions.items()}
            solved_coeffs.update(inferred_solutions)

        # Stop if no further progress can be made
        if not new_solved_coeffs and not inferred_equalities:
            break

    lyznet.tok()
    return solved_coeffs


def sympy_to_z3_inequality(sympy_expr, z3_subs):
    lhs, rhs = sympy_expr.lhs, sympy_expr.rhs
    lhs_z3 = lyznet.utils.sympy_to_z3(lhs, z3_subs)
    rhs_z3 = lyznet.utils.sympy_to_z3(rhs, z3_subs)
    
    if isinstance(sympy_expr, sp.LessThan):
        return lhs_z3 <= rhs_z3
    elif isinstance(sympy_expr, sp.GreaterThan):
        return lhs_z3 >= rhs_z3
    elif isinstance(sympy_expr, sp.StrictLessThan):
        return lhs_z3 < rhs_z3
    elif isinstance(sympy_expr, sp.StrictGreaterThan):
        return lhs_z3 > rhs_z3
    else:
        raise ValueError("Unsupported inequality type.")


def z3_global_lf_synthesis(system, terms, symbolic_reduction=False, 
                           LaSalle=False, LaSalle_Steps=10,
                           return_weak=False, time_out=60):
    """
    Global Lyapunov function synthesis with optional symbolic reduction.
    
    Args:
    - system: The dynamical system, containing symbolic_vars (state variables) and symbolic_f (dynamics).
    - terms: List of SymPy expressions representing the monomial terms for Lyapunov candidate.
    - symbolic_reduction: Boolean to enable symbolic reduction for highest odd-degree terms.
    - LaSalle: Boolean to enable fallback synthesis for a weak Lyapunov function (LF) per LaSalle’s criteria if strict LF synthesis fails.
    - LaSalle_Steps: Number of iterative LaSalle steps to apply if strict LF synthesis fails.
    
    Returns:
    - V_sympy: The Lyapunov function candidate if synthesis is successful; else None.
    """

    print('_' * 50)
    x = sp.Matrix(system.symbolic_vars)
    num_vars = len(system.symbolic_vars)

    # Step 1: Generate constraints if symbolic reduction is enabled
    if symbolic_reduction:
        print("Computing Lyapunov condition with Z3-Complete-SR:")
        solved_coeffs = z3_constraints_elimination(system, terms)

        # Substitute solved coefficients directly into V without altering the structure of terms
        coeffs = sp.symbols(f'c0:{len(terms)}')

        V = sum(c * term for c, term in zip(coeffs, terms)).subs(solved_coeffs)
        # print("\nSubstituted coefficients based on constraints:", solved_coeffs)
        print("Candidate V after reduction: ")

    else:
        print("Computing Lyapunov condition with Z3-Complete:")
        coeffs = sp.symbols(f'c0:{len(terms)}')
        V = sum(c * term for c, term in zip(coeffs, terms))

    print("V(x) =", sp.simplify(V))

    # Step 2: Compute the gradient ∇V and Lie derivative LfV
    grad_V = sp.Matrix([sp.diff(V, var) for var in x])
    LfV = grad_V.dot(system.symbolic_f)

    # Define Z3 variables for each state variable and coefficient
    z3_x = [z3.Real(f"x{i+1}") for i in range(num_vars)]
    z3_subs = {system.symbolic_vars[i]: z3_x[i] for i in range(num_vars)}
    z3_coeffs = [z3.Real(str(c)) for c in coeffs]
    z3_coeffs_dict = {coeff: z3_coeffs[i] for i, coeff in enumerate(coeffs)}

    # Use sympy_to_z3 to maintain exact symbolic precision
    func_subs = {}  # Placeholder for function substitutions, if needed
    extra_constraints = []  # Placeholder for extra constraints

    V_z3 = lyznet.utils.sympy_to_z3(V, {**z3_subs, **z3_coeffs_dict}, func_subs, extra_constraints)
    LfV_z3 = lyznet.utils.sympy_to_z3(LfV, {**z3_subs, **z3_coeffs_dict}, func_subs, extra_constraints)

    # Positivity and Lyapunov conditions
    z3_x_non_zero = z3.Or(*[xi != 0 for xi in z3_x])
    pos_def_condition = V_z3 > 0
    lf_condition = LfV_z3 < 0

    lyapunov_conditions = z3.And(pos_def_condition, lf_condition)

    main_implication = z3.ForAll(z3_x, z3.Implies(z3_x_non_zero, lyapunov_conditions))

    # Z3 solver setup
    solver = z3.Solver()
    solver.set("timeout", int(time_out * 1000))
    solver.add(main_implication)

    print("\nSolving with Z3...")
    lyznet.tik()
    if solver.check() == z3.sat:
        print("A Lyapunov function exists for the system!")
        model = solver.model()
        lyznet.tok()

        # Extract substitutions for all coefficients at once
        substitutions = {
            coeff: sp.Rational(model[z3_coeffs_dict[coeff]].numerator_as_long(), model[z3_coeffs_dict[coeff]].denominator_as_long())
            for coeff in V.free_symbols if coeff in z3_coeffs_dict
        }

        # Print coefficients
        print("Model found for coefficients:")
        for coeff, value in substitutions.items():
            print(f"{coeff} =", value)

        # Print Lyapunov function
        V_sympy = V.subs(substitutions)
        print("\nFinal Lyapunov function V(x):", V_sympy)

        return V_sympy

    else:
        print("No strict Lyapunov function found. Attempting LaSalle synthesis." if LaSalle else "No Lyapunov function found.")
        lyznet.tok()

    # If strict Lyapunov synthesis fails and return_weak is True, attempt weak synthesis
    if return_weak:
        print("\nAttempting weak Lyapunov function synthesis...")

        # Reset the solver and modify to weak Lyapunov conditions
        solver.reset()
        solver.set("timeout", int(time_out * 1000))
        weak_lyap_condition = LfV_z3 <= 0  # Weak condition allows LfV <= 0
        main_implication_weak = z3.ForAll(z3_x, z3.Implies(z3_x_non_zero, z3.And(pos_def_condition, weak_lyap_condition)))

        solver.add(main_implication_weak)
        if solver.check() == z3.sat:
            print("A weak Lyapunov function exists for the system!")
            model = solver.model()

            # Extract substitutions for all coefficients at once
            substitutions = {
                coeff: sp.Rational(model[z3_coeffs_dict[coeff]].numerator_as_long(), model[z3_coeffs_dict[coeff]].denominator_as_long())
                for coeff in V.free_symbols if coeff in z3_coeffs_dict
            }

            # Substitute coefficients into V
            V_sympy = V.subs(substitutions)
            print("\nFinal weak Lyapunov function V(x):", V_sympy)
            return V_sympy
        else:
            print("No weak Lyapunov function found.")

    # LaSalle synthesis if enabled
    if LaSalle:
        print("\nAttempting iterative LaSalle synthesis...")

        # Initialize Z3 variables for the solver
        solver = z3.Solver()
        solver.set("timeout", int(time_out * 1000))
        lyznet.tik()
        
        # Define positive definiteness and weak Lyapunov conditions outside the loop
        positive_definite_condition = z3.ForAll(z3_x, z3.Implies(z3_x_non_zero, V_z3 > 0))
        weak_lyap_condition = z3.ForAll(z3_x, LfV_z3 <= 0)

        # Initialize lasalle_condition to a trivial false condition
        lasalle_condition = z3.BoolVal(False)  # Start with false so that it requires a non-zero condition to pass

        # Start with the initial Lie derivative of V
        h = LfV
        success = False

        # Iteratively add LaSalle conditions up to LaSalle_Steps
        for i in range(LaSalle_Steps):
            print(f"Iteration {i+1} for LaSalle synthesis...")

            # Reset the solver at the start of each iteration
            solver.reset()
            solver.set("timeout", int(time_out * 1000))

            # Compute ∇h * f for the current h
            grad_h = sp.Matrix([sp.diff(h, var) for var in x])
            Lfh = grad_h.dot(system.symbolic_f)  # Next Lie derivative of h
        
            # Lambdify the new derivative to Z3
            grad_h_lambdified = sp.lambdify(tuple(system.symbolic_vars + list(coeffs)), Lfh, 'sympy')
            grad_h_z3 = grad_h_lambdified(*[z3_subs[var] for var in system.symbolic_vars], *z3_coeffs)
            
            # Add the new condition to lasalle_condition as an OR
            # lasalle_condition = z3.Or(lasalle_condition, grad_h_z3 != 0)

            # Only use the last h to formulate LaSalle's condition (sufficient)
            lasalle_condition = grad_h_z3 != 0

            # Formulate the complete LaSalle condition with positive definiteness and weak Lyapunov conditions
            solver.add(positive_definite_condition)
            solver.add(weak_lyap_condition)
            solver.add(z3.ForAll(z3_x, z3.Implies(z3.And(z3_x_non_zero, LfV_z3 == 0), lasalle_condition)))

            # # Print all constraints currently in the solver
            # print("Current constraints in the solver:")
            # for constraint in solver.assertions():
            #     print(constraint)

            print("\nSolving current LaSalle conditions with Z3...")
            if solver.check() == z3.sat:
                print(f"Success with LaSalle synthesis at iteration {i+1}!")
                model = solver.model()
                success = True
                lyznet.tok()

                # Extract substitutions for all coefficients
                substitutions = {
                    coeff: sp.Rational(model[z3_coeffs_dict[coeff]].numerator_as_long(), model[z3_coeffs_dict[coeff]].denominator_as_long())
                    for coeff in V.free_symbols if coeff in z3_coeffs_dict
                }

                # Print coefficients
                print("Model found for coefficients:")
                for coeff, value in substitutions.items():
                    print(f"{coeff} =", value)

                # Print Lyapunov function
                V_sympy = V.subs(substitutions)
                print("\nFinal Lyapunov function V(x):", V_sympy)
                return V_sympy

            # Update h to LhV for the next iteration if the condition is not conclusive
            h = Lfh

        if not success:
            print("No weak Lyapunov function (LaSalle) found after max iterations.")
            lyznet.tok()
            return None

    return None  


def z3_global_instability_synthesis(system, terms, symbolic_reduction=False,
                                    time_out=60, heuristic=True):
    """
    Global instability synthesis with Z3. Finds a candidate function V that satisfies:
    - LfV(x) <= 0 whenever V(x) <= 0 (no strict requirement).
    - There exists x such that V(x) < 0.

    Args:
    - system: The dynamical system, containing symbolic_vars (state variables) and symbolic_f (dynamics).
    - terms: List of SymPy expressions representing the monomial terms for the instability candidate function.
    - symbolic_reduction: Boolean to enable symbolic reduction for highest odd-degree terms.
    
    Returns:
    - V_sympy: The instability candidate function if synthesis is successful; else None.
    """

    print('_' * 50)
    print("Computing instability Lyapunov function with Z3:")

    x = sp.Matrix(system.symbolic_vars)
    num_vars = len(system.symbolic_vars)

    # Step 1: Generate constraints if symbolic reduction is enabled
    if symbolic_reduction:
        solved_coeffs = z3_constraints_elimination(system, terms, heuristic)

        # Substitute solved coefficients directly into V without altering the structure of terms
        coeffs = sp.symbols(f'c0:{len(terms)}')
        V = sum(c * term for c, term in zip(coeffs, terms)).subs(solved_coeffs)
        # print("\nSubstituted coefficients based on constraints:", solved_coeffs)
        print("Candidate V after reduction: ")
    else:
        coeffs = sp.symbols(f'c0:{len(terms)}')
        V = sum(c * term for c, term in zip(coeffs, terms))

    print("V(x) =", sp.simplify(V))

    # Step 2: Compute the gradient ∇V and Lie derivative LfV
    grad_V = sp.Matrix([sp.diff(V, var) for var in x])
    LfV = grad_V.dot(system.symbolic_f)

    # Define Z3 variables for each state variable and coefficient
    z3_x = [z3.Real(f"x{i+1}") for i in range(num_vars)]
    z3_subs = {system.symbolic_vars[i]: z3_x[i] for i in range(num_vars)}
    z3_coeffs = [z3.Real(str(c)) for c in coeffs]
    z3_coeffs_dict = {coeff: z3_coeffs[i] for i, coeff in enumerate(coeffs)}

    # Convert to Z3-compatible functions
    func_subs = {}
    extra_constraints = []
    V_z3 = lyznet.utils.sympy_to_z3(V, {**z3_subs, **z3_coeffs_dict}, func_subs, extra_constraints)
    LfV_z3 = lyznet.utils.sympy_to_z3(LfV, {**z3_subs, **z3_coeffs_dict}, func_subs, extra_constraints)

    # Instability condition: LfV(x) <= 0 whenever V(x) <= 0
    instability_condition = z3.Implies(V_z3 <= 0, LfV_z3 <= 0)

    # Existence of x such that V(x) < 0
    exists_negative_condition = z3.Exists(z3_x, V_z3 < 0)

    # Z3 solver setup for instability synthesis
    solver = z3.Solver()
    solver.set("timeout", int(time_out * 1000))
    # solver.add(extra_constraints)
    # print("extra_constraints: ", extra_constraints)

    solver.add(z3.ForAll(z3_x, instability_condition))
    solver.add(exists_negative_condition)

    print("\nSolving with Z3...")
    lyznet.tik()
    if solver.check() == z3.sat:
        print("An instability Lyapunov function exists for the system!")
        model = solver.model()
        lyznet.tok()

        # Extract substitutions for all coefficients
        substitutions = {
            coeff: sp.Rational(model[z3_coeffs_dict[coeff]].numerator_as_long(), model[z3_coeffs_dict[coeff]].denominator_as_long())
            for coeff in V.free_symbols if coeff in z3_coeffs_dict
        }

        # Print coefficients
        print("Model found for coefficients:")
        for coeff, value in substitutions.items():
            print(f"{coeff} =", value)

        # Print instability candidate function
        V_sympy = V.subs(substitutions)
        print("\nFinal instability Lyapunov function V(x):", V_sympy)

        return V_sympy
    else:
        print("No instability Lyapunov function found.")
        lyznet.tok()

    return None


def generate_lyap_template(system, terms):
    # Solve constraints to get expressions for coefficients
    solved_coeffs = z3_constraints_elimination(system, terms)

    # Define symbolic coefficients for each term
    coeffs = sp.symbols(f'c0:{len(terms)}')
    
    # Generate the reduced Lyapunov expression with substituted constraints
    reduced_lyap_expr = sum(c * term for c, term in zip(coeffs, terms)).subs(solved_coeffs)
    
    print("Candidate V after reduction: ", reduced_lyap_expr)

    # Identify unsolved coefficients
    unsolved_coeffs = [c for c in coeffs if c not in solved_coeffs]

    # Expand the reduced Lyapunov expression
    expanded_expr = sp.expand(reduced_lyap_expr)

    # Collect terms with respect to unsolved_coeffs
    collected_expr = sp.collect(expanded_expr, unsolved_coeffs)

    # Initialize lists for SymPy and Torch functions
    sympy_func_list = []
    torch_func_list = []

    # Define sympy_to_torch_func function outside the loop
    def sympy_to_torch_func(term, variables):
        x_symbols = [var for var in term.free_symbols if var in variables]
        sym_to_index = {str(var): i for i, var in enumerate(variables) if var in x_symbols}
        indexed_term = term.subs(
            {var: sp.Symbol(f"x[:, {sym_to_index[str(var)]}]") for var in x_symbols}
        )
        lambda_str = f"lambda x: {sp.sstr(indexed_term)}"
        return eval(lambda_str)
    
    # For each unsolved coefficient, extract the associated terms
    for coeff in unsolved_coeffs:
        coeff_expr = collected_expr.coeff(coeff)
        sympy_func_list.append(coeff_expr)
        torch_func_list.append(sympy_to_torch_func(coeff_expr, system.symbolic_vars))
    
    # # Debug print statements to verify the grouping
    # print("Reduced Lyapunov Expression:", collected_expr)
    # print("Trainable Coefficients:", unsolved_coeffs)
    # print("SymPy Function List:", sympy_func_list)
    # print("Torch Function List:", torch_func_list)

    return sympy_func_list, torch_func_list


def z3_cegis_global_lf_synthesis(system, terms, num_samples=500, 
                                 symbolic_reduction=False, max_iters=10,
                                 return_weak=False, LaSalle=False,
                                 time_out=60):
    """
    Z3-based Global Lyapunov function synthesis with CEGIS loop.

    Args:
    - system: The dynamical system, containing symbolic_vars (state variables) and symbolic_f (dynamics).
    - terms: List of SymPy expressions representing the monomial terms for the Lyapunov candidate.
    - num_samples: Number of random samples in the state space for constraint generation.
    - symbolic_reduction: Boolean indicating if we want to apply symbolic reduction on the terms.
    - max_iters: Maximum number of CEGIS iterations.

    Returns:
    - V_solution: The Lyapunov function candidate if synthesis is successful, else None.
    """
    # Generate symbolic function list based on reduction setting

    print('_' * 50)
    if symbolic_reduction:
        print(f"Computing Lyapunov condition with Z3-CEGIS-SR and {num_samples} samples:")
        solved_coeffs = z3_constraints_elimination(system, terms)
        coeffs = sp.symbols(f'c0:{len(terms)}')
        V = sum(c * term for c, term in zip(coeffs, terms)).subs(solved_coeffs)
            
    else:
        print(f"Computing Lyapunov condition with Z3-CEGIS and {num_samples} samples:")
        coeffs = sp.symbols(f'c0:{len(terms)}')
        V = sum(c * term for c, term in zip(coeffs, terms))

    # Generate random samples within the specified domain
    samples = lyznet.optimization.generate_random_samples(system.domain, num_samples=num_samples)

    print("V(x) =", sp.simplify(V))

    def sp_to_z3(expr, coeffs, z3_coeffs):
        """Converts a SymPy expression to a Z3 expression using specified coefficients."""
        z3_expr = sp.lambdify(coeffs, expr, 'sympy')(*z3_coeffs)
        return z3_expr

    def add_constraints_for_sample(solver, sample):
        """Adds Z3 constraints for a given sample."""
        sample_dict = {var: sample[i] for i, var in enumerate(system.symbolic_vars)}
        V_at_sample = V.subs(sample_dict)
        grad_V = sp.Matrix([sp.diff(V, var) for var in system.symbolic_vars])
        LfV_at_sample = sp.expand(grad_V.dot(system.symbolic_f)).subs(sample_dict)

        # Convert constraints to Z3 expressions
        z3_coeffs = [z3.Real(str(c)) for c in coeffs]
        solver.add(sp_to_z3(V_at_sample > 0, coeffs, z3_coeffs))
        if return_weak or LaSalle:
            solver.add(sp_to_z3(LfV_at_sample <= 0, coeffs, z3_coeffs))
        else:
            solver.add(sp_to_z3(LfV_at_sample < 0, coeffs, z3_coeffs))

        # # Add a minimum threshold constraint for coefficients
        # for z3_c in z3_coeffs:
        #     solver.add(z3_c > 1e-3)

    if return_weak:
        LaSalle = True

    # CEGIS loop
    for iter_count in range(max_iters):
        print(f"\n--- CEGIS Iteration {iter_count + 1} ---")

        lyznet.tik()
        solver = z3.Solver()
        solver.set("timeout", int(time_out * 1000))

        for sample in samples:
            add_constraints_for_sample(solver, sample)

        # # Print all constraints currently in the solver
        # print("All constraints in the solver:")
        # for constraint in solver.assertions():
        #     print(constraint)

        # Check for a feasible solution
        if solver.check() == z3.sat:
            model = solver.model()

            # Extract substitutions, rounding to 2 decimal places for smaller rationals
            substitutions = {
                coeff: sp.Rational(str(round(model[z3.Real(str(coeff))].numerator_as_long() / model[z3.Real(str(coeff))].denominator_as_long(), 2)))
                for coeff in coeffs if model[z3.Real(str(coeff))] is not None
            }

            # # Extract substitutions for all coefficients as exact SymPy Rationals
            # substitutions = {
            #     coeff: sp.Rational(model[z3.Real(str(coeff))].numerator_as_long(), model[z3.Real(str(coeff))].denominator_as_long())
            #     for coeff in coeffs if model[z3.Real(str(coeff))] is not None
            # }

            V_synthesized = V.subs(substitutions)
            print("Candidate Lyapunov function computed with Z3.")
            print("V = ", V_synthesized)
            lyznet.tok()

            # # Print the decision variables (substituted coefficients)
            # print("Decision Variables (substitutions):")
            # for coeff, value in substitutions.items():
            #     print(f"{coeff} = {value}")

            # Verify with the global LF verifier
            verified, counterexample = lyznet.z3_global_lf_verifier(system, V_sympy=V_synthesized, LaSalle=LaSalle, time_out=time_out)
            if verified:
                print("Synthesis and verification successful!")
                return V_synthesized
            elif counterexample:
                print("Counterexample found. Adding to samples for next iteration.")
                samples.append(counterexample)
        else:
            print("No solution found that satisfies the sample constraints.")
            return None

    print("CEGIS process completed. No valid Lyapunov function synthesized.")
    return None
