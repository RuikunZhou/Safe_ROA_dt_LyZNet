# import z3
import lyznet
import sympy as sp


def extract_sympy_PolyNet(system, net):
    x = sp.Matrix(system.symbolic_vars)
    layers = len(net.layers)
    
    weights = [layer.weight.data.cpu().numpy() for layer in net.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in net.layers]
    final_layer_weight = net.final_layer.weight.data.cpu().numpy()
    final_layer_bias = net.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = sp.Matrix([sum(h[j] * weights[i][k][j] for j in range(h.shape[0])) 
                       + biases[i][k] for k in range(len(biases[i]))])
        h = sp.Matrix([z[k]**net.deg for k in range(z.shape[0])])

    V_net = sum(h[k] * final_layer_weight[0][k] 
                for k in range(h.shape[0])) + final_layer_bias[0]    

    # return V_net

    # Expand and filter out constant and linear terms (i.e., V(0)=0, DV(0)=0)
    V_net_expanded = sp.expand(V_net)
    V_net_terms = V_net_expanded.as_coefficients_dict()
    V_net_filtered = sum(term * coeff for term, coeff in V_net_terms.items() 
                         if sp.total_degree(term, *x) > 1) 
    return V_net_filtered


def extract_sympy_RidgePoly(system, net):
    import sympy as sp
    
    x = sp.Matrix(system.symbolic_vars)
    ridge_directions = net.ridge_directions.data.cpu().numpy()
    coefficients = net.coefficients.data.cpu().numpy()
    degree = net.degree

    # Convert ridge directions to SymPy matrices
    ridge_directions_sympy = [sp.Matrix(ridge_directions[i, :]) 
                              for i in range(ridge_directions.shape[0])]

    # Compute the symbolic representation
    V_net = sum(
        coefficients[i] * (ridge_directions_sympy[i].dot(x))**degree
        for i in range(len(coefficients))
    )

    return V_net


def extract_sympy_DirectPoly(system, net):
    x = sp.Matrix(system.symbolic_vars)
    V_net = 0
    for i, powers in enumerate(net.terms):
        term_symbolic = sp.prod([x[j] ** powers[j] for j in range(len(x))])
        V_net += net.coefficients[i].item() * term_symbolic
    return V_net


def extract_sympy_LogPolyNet(system, net):
    a_expression = extract_sympy_PolyNet(system, net.a_poly)
    b_expression = extract_sympy_PolyNet(system, net.b_poly)
    return a_expression, b_expression


def extract_sympy_FixedPoly(system, net):
    x = sp.Matrix(system.symbolic_vars)
    V_net = net.reduced_lyap_expr

    # Substitute the trained coefficient values back into the expression
    trained_coeffs = {sp.Symbol(k): v.item() for k, v in net.trainable_coeffs.items()}
    V_net_substituted = V_net.subs(trained_coeffs)

    # Expand and filter out terms with degree > 1
    V_net_expanded = sp.expand(V_net_substituted)
    V_net_terms = V_net_expanded.as_coefficients_dict()
    V_net_filtered = sum(term * coeff for term, coeff in V_net_terms.items() 
                         if sp.total_degree(term, *x) > 1)
    
    return V_net_filtered


def z3_global_quadratic_verifier(system, P=None, c2_P=None):
    x = sp.Matrix(system.symbolic_vars)   
    if P is None:
        P = system.P 
        if P is None: 
            print("No P matrix available for quadratic Lyapunov function!")
            return 
    V = x.T * P * x
    grad_V = V.jacobian(x)
    LfV = grad_V * system.symbolic_f

    print('_' * 50)
    print("Verifying global quadratic Lyapunov function (with Z3):")
    print("V = ", sp.simplify(V[0]))

    # Create Z3 variables for state variables
    num_vars = len(system.symbolic_vars)
    z3_x = [z3.Real(f"x{i+1}") for i in range(num_vars)]
    subs = {system.symbolic_vars[i]: z3_x[i] for i in range(num_vars)}

    # Dictionaries to hold function substitutions and extra constraints
    func_subs = {}
    extra_constraints = []

    # Convert SymPy expressions to Z3 expressions
    V_z3 = lyznet.utils.sympy_to_z3(V[0], subs, func_subs, extra_constraints)
    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, extra_constraints)

    # print("LfV: ", LfV_z3)

    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x])  # Condition x ≠ 0
    condition = z3_x_non_zero

    if c2_P is not None:
        # Add the constraint V(x) >= c2_P (ultimate attraction/boundedness)
        condition = z3.And(condition, V_z3 >= c2_P)

    clf_condition = z3.Implies(condition, LfV_z3 < 0)

    # Initialize the solver
    solver = z3.Solver()
    solver.add(extra_constraints)  # Add constraints for sin/cos variables
    solver.add(z3.Not(clf_condition))  # Add the negation of the CLF condition

    lyznet.tik()
    # Check for satisfiability
    if solver.check() == z3.unsat:
        print("Global Lyapunov condition is verified!")
        verified = True
    else:
        print("The global Lyapunov condition is not valid. Counterexample:")
        model = solver.model()
        for i, var in enumerate(z3_x):
            print(f"x{i+1} = {model[var]}")

        verified = False

    lyznet.tok()
    return verified


def z3_quadratic_verifier(system, c_max=100, accuracy=1e-3):
    x = sp.Matrix(system.symbolic_vars)    
    V = x.T * sp.Matrix(system.P) * x
    grad_V = V.jacobian(x)
    LfV = grad_V * system.symbolic_f
    
    print('_' * 50)
    print("Verifying quadratic Lyapunov function (with Z3):")
    print("V = ", sp.simplify(V[0]))

    # Create Z3 variables for state variables
    num_vars = len(system.symbolic_vars)
    z3_x = [z3.Real(f"x{i+1}") for i in range(num_vars)]
    subs = {system.symbolic_vars[i]: z3_x[i] for i in range(num_vars)}

    # Dictionaries to hold function substitutions and extra constraints
    func_subs = {}
    extra_constraints = []

    # Convert SymPy expressions to Z3 expressions
    V_z3 = lyznet.utils.sympy_to_z3(V[0], subs, func_subs, extra_constraints)
    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, extra_constraints)

    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x])  # Condition x ≠ 0
    condition = z3_x_non_zero

    def verify_level_c(c):
        solver = z3.Solver()
        
        # Construct the condition for V(x) <= c
        c_condition = z3.And(V_z3 <= c)
        
        # Final condition: LfV < 0 given the conditions for x ≠ 0 and V(x)
        clf_condition = z3.Implies(z3.And(condition, c_condition), LfV_z3 < 0)
        
        solver.add(extra_constraints)
        solver.add(z3.Not(clf_condition))  # Negation of the Lyapunov condition

        result = solver.check()
        if result == z3.unsat:
            return None
        else:
            return solver.model()

    # Perform bisection to find the largest c that satisfies the Lyapunov condition
    lyznet.tik()
    c_optimal = lyznet.utils.bisection_glb(verify_level_c, 0, c_max, 
                                           accuracy=accuracy)
    print(f"Lyapunov condition verified for x'Px <= {c_optimal}.")
    lyznet.tok()

    return c_optimal


def z3_lf_verifier(system, net=None, V_sympy=None, c1_V=None, c_max=100, 
                   accuracy=1e-2):
    """Lyapunov function verification using Z3. """
    """Verify Lyapunov conditions on {c1_V<V<=c} for largest c in (0,c_max)."""

    if V_sympy is None:
        if net is None:
            print("Either net or V_sympy must be provided!")
            return False
        V_net = extract_sympy_PolyNet(system, net)
    else:
        V_net = V_sympy

    print('_' * 50)
    print("Verifying Lyapunov function (with Z3):")    
    print("V = ", sp.simplify(V_net))

    V_net_matrix = sp.Matrix([V_net])
    
    # Symbolic variables and expressions for gradient and dynamics
    x = sp.Matrix(system.symbolic_vars)
    grad_V = V_net_matrix.jacobian(x)  # Compute Jacobian of V_net
    LfV = grad_V * system.symbolic_f

    # Generate Z3 variables based on the system's symbolic variables
    z3_x = [z3.Real(f"x{i+1}") for i in range(len(system.symbolic_vars))]
    subs = {system.symbolic_vars[i]: z3_x[i] 
            for i in range(len(system.symbolic_vars))}
    
    # Convert the SymPy expressions to Z3
    func_subs = {}
    extra_constraints = []
    V_net_z3 = lyznet.utils.sympy_to_z3(V_net, subs, func_subs, 
                                        extra_constraints)

    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, 
                                      extra_constraints)
    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x]) 
    V_positive_condition = z3.Implies(z3_x_non_zero, V_net_z3 > 0)
    
    def verify_level_c(c2_V):
        solver = z3.Solver()
        c_bound = z3.And(V_net_z3 <= c2_V)   
        if c1_V is not None:
            c_bound = z3.And(c_bound, V_net_z3 >= c1_V) 
        lf_condition = z3.Implies(z3.And(z3_x_non_zero, c_bound), LfV_z3 < 0)
        lf_condition = z3.And(lf_condition, V_positive_condition)
        
        # Add extra constraints for sin/cos variables
        solver.add(extra_constraints)
        solver.add(z3.Not(lf_condition))  # Negation of the LF condition
        
        # Check for satisfiability
        result = solver.check()        
        if result == z3.unsat:
            return None
        else:
            return solver.model()

    # Perform bisection to find the largest c2_P s.t. the LF condition holds
    lyznet.tik()
    c_optimal = lyznet.utils.bisection_glb(verify_level_c, 0, c_max, 
                                           accuracy=accuracy)
    print(f"Lyapunov condition verified for x^T P x <= {c_optimal}.")
    lyznet.tok()

    return c_optimal


def z3_global_lf_verifier(system, net=None, V_sympy=None, c2_P=None, 
                          mode="Poly", a_poly=None, b_poly=None, 
                          LaSalle=False, LaSalle_Steps=10,
                          time_out=60):
    """Global Lyapunov function verification using Z3."""
    
    if V_sympy is None:
        if net is None and (mode != "LogPoly" or (a_poly is None or b_poly is None)):
            print("Either net or V_sympy must be provided, or specify a_poly and b_poly for LogPoly mode!")
            return False, None

        # Select extraction method based on mode
        if mode == "Poly":
            V_net = extract_sympy_PolyNet(system, net)
        elif mode == "DirectPoly":
            V_net = extract_sympy_DirectPoly(system, net)
        elif mode == "LogPoly":
            # Use provided a_poly and b_poly expressions directly
            a_expression = a_poly if a_poly is not None else extract_sympy_PolyNet(system, net.a_poly)
            b_expression = b_poly if b_poly is not None else extract_sympy_PolyNet(system, net.b_poly)
            V_net = sp.log(1 + a_expression) + b_expression
        else:
            raise ValueError(f"Unsupported verification mode: {mode}")
    else:
        V_net = V_sympy

    print('_' * 50)
    print("Verifying global Lyapunov function (with Z3):")    
    print("V = ", sp.simplify(V_net))

    x = sp.Matrix(system.symbolic_vars)
    V_net_matrix = sp.Matrix([V_net])
    
    # Calculate the gradient and Lie derivative based on the mode
    if mode == "LogPoly":
        a_grad = sp.Matrix([sp.diff(a_expression, var) for var in x])
        b_grad = sp.Matrix([sp.diff(b_expression, var) for var in x])
        f = sp.Matrix(system.symbolic_f)
        LfV = sp.Matrix([a_grad.dot(f) / (1 + a_expression) + b_grad.dot(f)])
    else:
        grad_V = V_net_matrix.jacobian(x)
        LfV = grad_V * system.symbolic_f

    # Convert SymPy expressions to Z3-compatible ones
    z3_x = [z3.Real(f"x{i+1}") for i in range(len(system.symbolic_vars))]
    subs = {system.symbolic_vars[i]: z3_x[i] for i in range(len(system.symbolic_vars))}
    
    func_subs = {}
    extra_constraints = []
    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, extra_constraints)
    z3_x_non_zero = z3.Or(*[xi != 0 for xi in z3_x])

    # Create conditions in Z3
    condition = z3_x_non_zero
    if mode == "LogPoly":
        a_z3 = lyznet.utils.sympy_to_z3(a_expression, subs, func_subs, extra_constraints)
        b_z3 = lyznet.utils.sympy_to_z3(b_expression, subs, func_subs, extra_constraints)
        positivity_z3 = z3.Implies(z3_x_non_zero, z3.Or(a_z3 > 0, b_z3 > 0))
        nonnegativity_z3 = z3.And(a_z3 >= 0, b_z3 >= 0)
        lf_condition = z3.And(z3.Implies(condition, LfV_z3 < 0), positivity_z3, nonnegativity_z3)
    else:
        V_net_z3 = lyznet.utils.sympy_to_z3(V_net, subs, func_subs, extra_constraints)
        if c2_P is not None:
            condition = z3.And(condition, V_net_z3 >= c2_P)
        lf_condition = z3.And(z3.Implies(condition, LfV_z3 < 0), z3.Implies(z3_x_non_zero, V_net_z3 > 0))

    # Set up Z3 solver with negation of the conditions
    solver = z3.Solver()
    solver.set("timeout", int(time_out * 1000))
    solver.add(extra_constraints)
    # print("extra_constraints: ", extra_constraints)

    solver.add(z3.Not(lf_condition))

    # Print each constraint in the solver
    # print("All constraints added to the Z3 solver:")
    # for constraint in solver.assertions():
    #     print(constraint)

    # Check and handle verification result
    lyznet.tik()
    if solver.check() == z3.unsat:
        print("Global Lyapunov condition is verified!")
        lyznet.tok()
        return True, None  
    else:
        print("The strict global Lyapunov condition is not valid. Counterexample:")
        model = solver.model()

        # Return counterexamples as floats
        counterexample = [float(model.eval(var).numerator_as_long()) / model.eval(var).denominator_as_long() 
                          for var in z3_x]
        
        # # Return counterexamples in exact rationals
        # counterexample = [
        #     sp.Rational(model.eval(var).numerator_as_long(), model.eval(var).denominator_as_long()) 
        #     for var in z3_x
        # ]

        # Display counterexample
        for i, val in enumerate(counterexample):
            print(f"x{i+1} = {val}")

        lyznet.tok()

        # Continue to LaSalle verification if enabled
        if not LaSalle:
            return False, counterexample

    # LaSalle verification if enabled
    if LaSalle:
        print("\nAttempting LaSalle's invariance condition for global asymptotic stability...")

        lyznet.tik()
        # Set up weak Lyapunov condition for LaSalle verification
        weak_lf_condition = z3.And(z3.Implies(z3_x_non_zero, V_net_z3 > 0), 
                                   LfV_z3 <= 0)

        h = LfV
        lasalle_condition = z3.BoolVal(False)

        for i in range(LaSalle_Steps):
            grad_h = sp.Matrix([sp.diff(h, var) for var in x])
            LhV = grad_h.dot(system.symbolic_f)
            grad_h_z3 = lyznet.utils.sympy_to_z3(LhV, subs, func_subs, extra_constraints)

            # Update LaSalle condition
            # lasalle_condition = z3.Or(lasalle_condition, grad_h_z3 != 0)
            lasalle_condition = grad_h_z3 != 0

            # Set up Z3 solver for LaSalle verification
            solver.reset()
            solver.set("timeout", int(time_out * 1000))
            solver.add(extra_constraints)
            solver.add(z3.Not(z3.And(weak_lf_condition, 
                                     z3.Implies(z3.And(z3_x_non_zero, LfV_z3 == 0), 
                                                lasalle_condition))))

            # solver.add(z3.Not(weak_lf_condition))

            if solver.check() == z3.unsat:
                print(f"LaSalle invariance condition is satisfied at iteration {i+1}.")
                print("Global asymptotic stability is verified!")
                lyznet.tok()
                return True, None
            else:
                print(f"Failed LaSalle verification at iteration {i+1}.")
                h = LhV  # Update for the next iteration

        # If all iterations fail
        print("LaSalle verification failed.")
        lyznet.tok()
        return False, None


def z3_clf_verifier(system, net=None, V_sympy=None, c_max=100, accuracy=1e-2):

    if V_sympy is None:
        if net is None:
            print("Either net or V_sympy must be provided!")
            return False
        V_net = extract_sympy_PolyNet(system, net)
    else:
        V_net = V_sympy

    print('_' * 50)
    print("Verifying control Lyapunov function (with Z3):")    
    print("V = ", sp.simplify(V_net))

    V_net_matrix = sp.Matrix([V_net])
    
    # Symbolic variables and expressions for gradient and dynamics
    x = sp.Matrix(system.symbolic_vars)
    grad_V = V_net_matrix.jacobian(x)  # Compute Jacobian of V_net
    LfV = grad_V * system.symbolic_f
    LgV = grad_V * system.symbolic_g

    # Generate Z3 variables based on the system's symbolic variables
    z3_x = [z3.Real(f"x{i+1}") for i in range(len(system.symbolic_vars))]
    subs = {system.symbolic_vars[i]: z3_x[i] for i in range(len(system.symbolic_vars))}
    
    # Convert the SymPy expressions to Z3
    func_subs = {}
    extra_constraints = []
    V_net_z3 = lyznet.utils.sympy_to_z3(V_net, subs, func_subs, extra_constraints)

    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, extra_constraints)
    LgV_z3_list = [lyznet.utils.sympy_to_z3(expr, subs, func_subs, extra_constraints) for expr in LgV]

    # Construct the conditions
    LgV_zero_z3 = z3.And(*[expr == 0 for expr in LgV_z3_list])
    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x]) 
    condition = z3.And(LgV_zero_z3, z3_x_non_zero)

    V_positive_condition = z3.Implies(z3_x_non_zero, V_net_z3 > 0)
    
    def verify_level_c(c2_P):
        solver = z3.Solver()
        
        # Construct the condition for V(x) <= c2_P
        c_condition = z3.And(V_net_z3 <= c2_P)
        
        # Final condition: LfV < 0 given the conditions for LgV, x, and V(x)
        clf_condition = z3.Implies(z3.And(condition, c_condition), LfV_z3 < 0)

        clf_condition = z3.And(clf_condition, V_positive_condition)
        
        # Add extra constraints for sin/cos variables
        solver.add(extra_constraints)
        solver.add(z3.Not(clf_condition))  # Negation of the CLF condition
        
        # Check for satisfiability
        result = solver.check()
        
        if result == z3.unsat:
            return None
        else:
            return solver.model()

    # Perform bisection to find the largest c2_P that satisfies the CLF condition
    lyznet.tik()
    c_optimal = lyznet.utils.bisection_glb(verify_level_c, 0, c_max, accuracy=accuracy)
    print(f"CLF condition verified for x^T P x <= {c_optimal}.")
    lyznet.tok()

    return c_optimal


def z3_global_clf_verifier(system, net=None, V_sympy=None, c2_P=None):
    """Global CLF verification using Z3."""
    
    if V_sympy is None:
        if net is None:
            print("Either net or V_sympy must be provided!")
            return False
        V_net = extract_sympy_PolyNet(system, net)
    else:
        V_net = V_sympy

    print('_' * 50)
    print("Verifying global control Lyapunov function (with Z3):")    
    print("V = ", sp.simplify(V_net))

    # Symbolic variables and expressions for gradient and dynamics

    V_net_matrix = sp.Matrix([V_net])
    
    # Symbolic variables and expressions for gradient and dynamics
    x = sp.Matrix(system.symbolic_vars)
    grad_V = V_net_matrix.jacobian(x)  # Compute Jacobian of V_net
    LfV = grad_V * system.symbolic_f
    LgV = grad_V * system.symbolic_g

    # Generate Z3 variables based on the system's symbolic variables
    z3_x = [z3.Real(f"x{i+1}") for i in range(len(system.symbolic_vars))]
    subs = {system.symbolic_vars[i]: z3_x[i] for i in range(len(system.symbolic_vars))}
    
    # Convert the SymPy expressions to Z3
    func_subs = {}
    extra_constraints = []
    V_net_z3 = lyznet.utils.sympy_to_z3(V_net, subs, func_subs, extra_constraints)
    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, extra_constraints)
    LgV_z3_list = [lyznet.utils.sympy_to_z3(expr, subs, func_subs, extra_constraints) for expr in LgV]

    # Construct the conditions
    LgV_zero_z3 = z3.And(*[expr == 0 for expr in LgV_z3_list])
    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x])  
    condition = z3.And(LgV_zero_z3, z3_x_non_zero)

    if c2_P is not None:
        # Add the constraint V(x) >= c2_P
        condition = z3.And(condition, V_net_z3 >= c2_P)

    clf_condition = z3.Implies(condition, LfV_z3 < 0)

    V_positive_condition = z3.Implies(z3_x_non_zero, V_net_z3 > 0)
    clf_condition = z3.And(clf_condition, V_positive_condition)

    # Initialize the solver
    solver = z3.Solver()
    solver.add(extra_constraints)  
    solver.add(z3.Not(clf_condition))  

    lyznet.tik() 
    if solver.check() == z3.unsat:
        print("Global CLF condition is verified!")
        verified = True

    else:
        print("The global CLF condition is not valid. Counterexample:")
        model = solver.model()
        for i, var in enumerate(z3_x):
            print(f"x{i+1} = {model[var]}")
        for func_expr, z3_var in func_subs.items():
            print(f"{z3_var} = {model[z3_var]} (represents {func_expr})")
    
        # Evaluate the conditions using the counterexample model
        LgV_zero_val = model.eval(LgV_zero_z3, model_completion=True)
        z3_x_non_zero_val = model.eval(z3_x_non_zero, model_completion=True)
        LfV_val = model.eval(LfV_z3 < 0, model_completion=True)
        V_net_val = model.eval(V_net_z3 > 0, model_completion=True)  # Check if V(x) > 0

        print(f"LgV=zero evaluates to: {LgV_zero_val}")
        print(f"x_non_zero evaluates to: {z3_x_non_zero_val}")
        print(f"LfV < 0 evaluates to: {LfV_val}")
        print(f"V > 0 evaluates to: {V_net_val}") 

        verified = False
        
    lyznet.tok() 

    return verified


def z3_global_quadratic_clf_verifier(system, c2_P=None, P=None):
    x = sp.Matrix(system.symbolic_vars)    

    if P is None:
        P = system.P 
        if P is None: 
            print("No P matrix available for quadratic Lyapunov function!")
            return 

    V = x.T * sp.Matrix(P) * x

    grad_V = V.jacobian(x)
    LfV = grad_V * system.symbolic_f
    LgV = grad_V * system.symbolic_g

    print('_' * 50)
    print("Verifying global quadratic control Lyapunov function (with Z3):")
    print("V = ", sp.simplify(V[0]))

    # Create Z3 variables for state variables
    num_vars = len(system.symbolic_vars)
    z3_x = [z3.Real(f"x{i+1}") for i in range(num_vars)]
    subs = {system.symbolic_vars[i]: z3_x[i] for i in range(num_vars)}

    # Dictionaries to hold function substitutions and extra constraints
    func_subs = {}
    extra_constraints = []

    # Convert SymPy expressions to Z3 expressions
    V_z3 = lyznet.utils.sympy_to_z3(V[0], subs, func_subs, extra_constraints)
    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, extra_constraints)
    LgV_z3_list = [lyznet.utils.sympy_to_z3(expr, subs, func_subs, extra_constraints) for expr in LgV]

    # Construct the conditions
    LgV_zero_z3 = z3.And(*[expr == 0 for expr in LgV_z3_list])
    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x])  # At least one variable is non-zero
    condition = z3.And(LgV_zero_z3, z3_x_non_zero)

    if c2_P is not None:
        # Add the constraint V(x) >= c2_P
        condition = z3.And(condition, V_z3 >= c2_P)
    clf_condition = z3.Implies(condition, LfV_z3 < 0)

    # Initialize the solver
    solver = z3.Solver()
    solver.add(extra_constraints)  # Add constraints for sin and cos variables
    # print("extra_constraints: ", extra_constraints)

    solver.add(z3.Not(clf_condition))  # Add the negation of the CLF condition
    # print("clf_condition: ", clf_condition)

    lyznet.tik()
    # Check for satisfiability
    if solver.check() == z3.unsat:
        print("Global CLF condition is verified!")
        verified = True
    else:
        print("The global CLF condition is not valid. Counterexample:")
        model = solver.model()
        for i, var in enumerate(z3_x):
            print(f"x{i+1} = {model[var]}")
        # Print the values of the function variables
        for func_expr, z3_var in func_subs.items():
            print(f"{z3_var} = {model[z3_var]} (represents {func_expr})")
        
        # Evaluate the conditions using the counterexample model
        LgV_zero_val = model.eval(LgV_zero_z3, model_completion=True)
        z3_x_non_zero_val = model.eval(z3_x_non_zero, model_completion=True)
        LfV_val = model.eval(LfV_z3 < 0, model_completion=True)
        V_val = model.eval(V_z3 > 0, model_completion=True)  # Check if V(x) > 0

        print(f"LgV=zero evaluates to: {LgV_zero_val}")
        print(f"x_non_zero evaluates to: {z3_x_non_zero_val}")
        print(f"LfV < 0 evaluates to: {LfV_val}")
        print(f"V > 0 evaluates to: {V_val}") 

        verified = False

    lyznet.tok()

    return verified


def z3_quadratic_clf_verifier(system, c_max=100, accuracy=1e-3):
    x = sp.Matrix(system.symbolic_vars)    
    V = x.T * sp.Matrix(system.P) * x
    grad_V = V.jacobian(x)    
    LfV = grad_V * system.symbolic_f
    LgV = grad_V * system.symbolic_g
    
    print('_' * 50)
    print("Verifying quadratic control Lyapunov function (with Z3):")
    print("V = ", sp.simplify(V[0]))

    # Create Z3 variables for state variables
    num_vars = len(system.symbolic_vars)
    z3_x = [z3.Real(f"x{i+1}") for i in range(num_vars)]
    subs = {system.symbolic_vars[i]: z3_x[i] for i in range(num_vars)}

    # Dictionaries to hold function substitutions and extra constraints
    func_subs = {}
    extra_constraints = []

    # Convert SymPy expressions to Z3 expressions
    V_z3 = lyznet.utils.sympy_to_z3(V[0], subs, func_subs, extra_constraints)
    LfV_z3 = lyznet.utils.sympy_to_z3(LfV[0], subs, func_subs, extra_constraints)
    LgV_z3_list = [lyznet.utils.sympy_to_z3(expr, subs, func_subs, extra_constraints) for expr in LgV]

    # Construct the conditions
    LgV_zero_z3 = z3.And(*[expr == 0 for expr in LgV_z3_list])
    z3_x_non_zero = z3.Or(*[x != 0 for x in z3_x])  
    condition = z3.And(LgV_zero_z3, z3_x_non_zero)

    def verify_level_c(c2_P):
        solver = z3.Solver()
        c_condition = z3.And(V_z3 <= c2_P)
        clf_condition = z3.Implies(z3.And(condition, c_condition), LfV_z3 < 0)        
        # Add extra constraints for sin/cos variables
        solver.add(extra_constraints)
        solver.add(z3.Not(clf_condition))  # Negation of the CLF condition
        result = solver.check()
        
        if result == z3.unsat:
            return None
        else:
            return solver.model()

    lyznet.tik()
    c_optimal = lyznet.utils.bisection_glb(verify_level_c, 0, c_max, accuracy=accuracy)
    print(f"CLF condition verified for x^T P x <= {c_optimal}.")
    lyznet.tok()

    return c_optimal
