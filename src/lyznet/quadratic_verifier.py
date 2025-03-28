import time 
import dreal
import lyznet

def reach_verifier_dreal_quad(system, x, V, f, c1, V_f_x=None, c_max=100, 
                         tol=1e-4, accuracy=1e-4, number_of_jobs=32, h=None):
    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain
    epsilon = tol

    if system.system_type == "DiscreteDynamicalSystem":
        if V_f_x is None:
            f = [lyznet.utils.simplify_dreal_expression(expr, x) for expr in f]
            V_f_x = lyznet.utils.compose_dreal_expressions(V, x, f, x)
        lyap_condition = V_f_x - V 
        
    else:        
        lie_derivative_of_V = dreal.Expression(0)
        for i in range(len(x)):
            lie_derivative_of_V += f[i] * V.Differentiate(x[i])
        lyap_condition = lie_derivative_of_V

    # add safety constraints
    if h is not None:
        h_conditions = [
            lyznet.utils.sympy_to_dreal(hi, dict(zip(system.symbolic_vars, x)))
            for hi in h
        ]
        safe_set = dreal.logical_and(*[hi < 1 for hi in h_conditions])  

    def Check_reachability(c2, c1=c1):    
        x_bound = lyznet.utils.get_bound(x, xlim, V, c1_V=c1, c2_V=c2)
        reach = dreal.logical_imply(x_bound, lyap_condition <= -epsilon)
        # condition = dreal.logical_and(reach, dynamics_inclusion)

        if h is not None:
            omega = V <= c2
            safety = dreal.logical_imply(omega, safe_set)
            reach = dreal.logical_and(reach, safety)
        
        return dreal.CheckSatisfiability(dreal.logical_not(reach), config)

    # c_best = lyznet.utils.bisection_glb(Check_reachability, 
    #                                     c1, c_max, accuracy)
    # return c_best

    # avoid repeating the verification from c1-level set all the time
    best_c1 = c1
    if Check_reachability(c_max, best_c1) is None:
        return c_max

    while c_max - best_c1 > accuracy:
        mid = (best_c1 + c_max) / 2
        if Check_reachability(mid, best_c1) is None:
            best_c1 = mid  
        else:
            c_max = mid  

    return best_c1

def reach_verifier_dreal(system, x, V, f, c1, V_f_x=None, c_max=100, 
                         tol=1e-4, accuracy=1e-4, number_of_jobs=32, h=None):
    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain
    epsilon = tol

    if system.system_type == "DiscreteDynamicalSystem":
        if V_f_x is None:
            f = [lyznet.utils.simplify_dreal_expression(expr, x) for expr in f]
            V_f_x = lyznet.utils.compose_dreal_expressions(V, x, f, x)
        lyap_condition = V_f_x - V 
        

    else:        
        lie_derivative_of_V = dreal.Expression(0)
        for i in range(len(x)):
            lie_derivative_of_V += f[i] * V.Differentiate(x[i])
        lyap_condition = lie_derivative_of_V

    # add safety constraints
    if h is not None:
        h_conditions = [
            lyznet.utils.sympy_to_dreal(hi, dict(zip(system.symbolic_vars, x)))
            for hi in h
        ]
        safe_set = dreal.logical_and(*[hi < 1 for hi in h_conditions])  

    def Check_reachability(c2, c1=c1):    
        x_bound = lyznet.utils.get_bound(x, xlim, V, c1_V=c1, c2_V=c2)
        
        if system.system_type == "DiscreteDynamicalSystem":
            dynamics = dreal.logical_and(f[0] >= xlim[0][0], f[0] <= xlim[0][1])
            for i in range(1, len(x)):
                dynamics = dreal.logical_and(f[i] >= xlim[i][0], dynamics)
                dynamics = dreal.logical_and(f[i] <= xlim[i][1], dynamics)
            dynamics_inclusion = dreal.logical_imply(x_bound, dynamics)
            # print("dynamics_inclusion: ", dynamics_inclusion)
            lyap = dreal.logical_imply(x_bound, lyap_condition <= -epsilon)
            reach = dreal.logical_and(lyap, dynamics_inclusion)
        else: 
            reach = dreal.logical_imply(x_bound, lyap_condition <= -epsilon)
        # condition = dreal.logical_and(reach, dynamics_inclusion)
        if h is not None:
            omega = V <= c2
            safety = dreal.logical_imply(omega, safe_set)
            reach = dreal.logical_and(reach, safety)
        # return dreal.CheckSatisfiability(dreal.logical_not(condition), config) if with h
        return dreal.CheckSatisfiability(dreal.logical_not(reach), config)

    # c_best = lyznet.utils.bisection_glb(Check_reachability, 
    #                                     c1, c_max, accuracy)
    # return c_best

    # avoid repeating the verification from c1-level set all the time
    best_c1 = c1
    if Check_reachability(c_max, best_c1) is None:
        return c_max

    while c_max - best_c1 > accuracy:
        mid = (best_c1 + c_max) / 2
        if Check_reachability(mid, best_c1) is None:
            best_c1 = mid  
        else:
            c_max = mid  

    return best_c1


def clf_reach_verifier_dreal(system, x, V, c1, c_max=100, tol=1e-4, 
                             accuracy=1e-4, number_of_jobs=32):
    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    g = [
        [
            lyznet.utils.sympy_to_dreal(expr, dict(zip(system.symbolic_vars, x)))
            for expr in row
        ]
        for row in system.symbolic_g.tolist()
    ]

    LfV = dreal.Expression(0)
    for i in range(len(x)):
        LfV += f[i] * V.Differentiate(x[i])
    grad_V = [[V.Differentiate(x[i]) for i in range(len(x))]]
    LgV = lyznet.utils.matrix_multiply_dreal(grad_V, g)
    LgV_zero = dreal.And(*(expr == dreal.Expression(0) for row in LgV 
                           for expr in row))

    def Check_clf_condition(c2):    
        x_bound = lyznet.utils.get_bound(x, xlim, V, c1_V=c1, c2_V=c2)
        x_boundary = dreal.logical_or(x[0] == xlim[0][0], x[0] == xlim[0][1])
        for i in range(1, len(x)):
            x_boundary = dreal.logical_or(x[i] == xlim[i][0], x_boundary)
            x_boundary = dreal.logical_or(x[i] == xlim[i][1], x_boundary)
        set_inclusion = dreal.logical_imply(
            x_bound, dreal.logical_not(x_boundary)
            )
        clf_condition = dreal.logical_imply(dreal.And(x_bound, LgV_zero), 
                                            LfV < 0)
        condition = dreal.logical_and(clf_condition, set_inclusion)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
    
    c_best = lyznet.utils.bisection_glb(Check_clf_condition, 
                                        c1, c_max, accuracy)

    return c_best


def quadratic_reach_verifier(system, c1_P, tol=1e-4, accuracy=1e-4,
                             number_of_jobs=32, c_max=100, V_sympy=None, 
                             h=None):
    # Create dReal variables based on the number of symbolic variables
    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]

    if V_sympy is None:
        # Construct quadratic Lyapunov function x^T P x
        V = dreal.Expression(0)
        for i in range(len(x)):
            for j in range(len(x)):
                V += x[i] * system.P[i][j] * x[j]
    else:
        # If V_sympy is provided, convert it to a dReal expression
        V = lyznet.utils.sympy_to_dreal(V_sympy, dict(zip(system.symbolic_vars, x)))

    print('_' * 50)
    print("Verifying ROA using quadratic Lyapunov function:")
    print("x^TPx = ", V)

    # Create dReal expressions for f based on the symbolic expressions
    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    # for discrete-time systems, V(f(x)) is passed to the verifier
    V_f_x = None
    if system.system_type == "DiscreteDynamicalSystem":
        V_f_x = dreal.Expression(0)
        for i in range(len(f)):
            for j in range(len(f)):
                V_f_x += f[i] * system.P[i][j] * f[j]
        V_f_x = lyznet.utils.simplify_dreal_expression(V_f_x, x)
        # print("Simplified V_f_x: ", V_f_x)

    if c1_P is None:
        c1_P = 1e-3
        print(f"Target level set not specificed. Set it to be x^TPx<={c1_P}.")        

    start_time = time.time()
    c2_P = reach_verifier_dreal_quad(system, x, V, f, c1_P, V_f_x, c_max=c_max, 
                                tol=tol, accuracy=accuracy,
                                number_of_jobs=number_of_jobs, h=h)

    if c2_P is None:
        c2_P = c1_P

    end_time = time.time()
    if c2_P > c1_P:     
        print(f"Largest level set x^T*P*x <= {c2_P} verified by reach & stay.")
    else:
        print(f"Largest level set x^T*P*x <= {c2_P} remains the same.")
    print(f"Time taken for verification: {end_time - start_time} seconds.\n")
    return c2_P


def quadratic_CM_verifier(system, c1_P, c2_P, tol=1e-4, accuracy=1e-4,
                          number_of_jobs=32):
    print('_' * 50)
    print("Verifying quadratic contraction metric...")

    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain
    d = len(system.symbolic_vars)

    # Create dReal variables based on the number of symbolic variables
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]
    V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * system.P[i][j] * x[j]

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    Df_x = [[fi.Differentiate(xi) for xi in x] for fi in f]
    Df_x_T = list(map(list, zip(*Df_x)))  # transpose of Df_x

    M_prod1 = lyznet.utils.matrix_multiply_dreal(Df_x_T, system.P) 
    M_prod2 = list(map(list, zip(*M_prod1))) 

    CM_derivative = [[M_prod1[i][j] + M_prod2[i][j] 
                      for j in range(d)] for i in range(d)]

    # verifying negative definiteness of CM_derivative
    constraints = []
    for n in range(1, d + 1):
        sub_matrix = [[-CM_derivative[i][j] for j in range(n)] for i in range(n)]
        det_sub_matrix = lyznet.utils.compute_determinant_dreal(sub_matrix)
        constraints.append(det_sub_matrix >= tol)

    negative_definiteness = dreal.And(*constraints)

    def Check_contraction(c2):    
        x_bound = lyznet.utils.get_bound(x, xlim, V, c2_V=c2)
        condition = dreal.logical_imply(x_bound, negative_definiteness)    
        result = dreal.CheckSatisfiability(
            dreal.logical_not(condition), config
            )
        return result

    start_time = time.time()
    c_best = lyznet.utils.bisection_glb(Check_contraction, 
                                        c1_P, c2_P, accuracy)
    end_time = time.time()

    print(f"Largest level set x^T*P*x <= {c_best} verified " 
          "for P to be a contraction metric.")
    print(f"Time taken for verification: " 
          f"{end_time - start_time} seconds.\n")

    return c_best


def quadratic_clf_verifier(system, c1_P, c_max=100, tol=1e-4, accuracy=1e-5,
                           number_of_jobs=32):
    if system.P is None:
        return

    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]

    V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * system.P[i][j] * x[j]
    print('_' * 50)
    print("Verifying quadratic control Lyapunov function:")
    print("x^TPx = ", V)

    start_time = time.time()
    c2_P = clf_reach_verifier_dreal(system, x, V, c1_P, c_max=c_max, 
                                    tol=tol, accuracy=accuracy,
                                    number_of_jobs=number_of_jobs)    
    end_time = time.time()

    if c2_P is None:
        c2_P = c1_P
    if c2_P > c1_P:     
        print(f"Largest level set x^T*P*x <= {c2_P} verified for CLF condition.")
    else:
        print(f"Largest level set x^T*P*x <= {c2_P} remains the same.")
    print(f"Time taken for verification: {end_time - start_time} seconds.\n")
    
    return c2_P
