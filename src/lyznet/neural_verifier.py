import time 

import numpy as np
import dreal 

import lyznet
from itertools import product
from sympy import factorial

import sympy


def extract_dreal_Net(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net


def extract_dreal_CMNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    CM_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)
    return CM_net


def extract_dreal_UNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    U_net = [np.dot(h, final_layer_weight[i]) + final_layer_bias[i] 
             for i in range(final_layer_weight.shape[0])]
    # U_net = np.dot(h, final_layer_weight.T) + final_layer_bias
    return U_net


def extract_dreal_PolyNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]
    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [z[j]**2 for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net


def extract_dreal_RidgePoly(model, x):
    num_ridges = model.num_ridges
    ridge_directions = model.ridge_directions.data.cpu().numpy()
    coefficients = model.coefficients.data.cpu().numpy()
    degree = model.degree

    # Compute ridge outputs
    ridge_outputs = np.dot(x, ridge_directions.T)**degree
    
    # Compute final output
    V_net = np.dot(ridge_outputs, coefficients)
    return V_net


def extract_dreal_HomoNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    norm = dreal.sqrt(sum(xi * xi for xi in x))
    h = [xi / norm for xi in x]
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    
    input_layer_weight_norm = np.linalg.norm(weights[0])

    return V_net * (norm ** model.deg), input_layer_weight_norm


def extract_dreal_HomoUNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    # Compute the norm of the input and normalize the inputs
    norm = dreal.sqrt(sum(xi * xi for xi in x))
    h = [xi / norm for xi in x]

    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    # Calculate U_net for each output neuron
    U_net = [(np.dot(h, final_layer_weight[i].T) + final_layer_bias[i]) * (norm ** model.deg)
             for i in range(final_layer_weight.shape[0])]
    
    input_layer_weight_norm = np.linalg.norm(weights[0])

    return U_net, input_layer_weight_norm


def extract_dreal_HomoPolyNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    norm = dreal.sqrt(sum(xi * xi for xi in x))
    h = [xi / norm for xi in x]
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [z[j]**2 for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net * (norm ** model.deg)


def extract_dreal_SimpleNet(model, x):
    d = len(model.initial_layers)    
    weights = [layer.weight.data.cpu().numpy() 
               for layer in model.initial_layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.initial_layers]
    
    h = []
    for i in range(d):
        xi = x[i]  
        z = xi * weights[i][0, 0] + biases[i][0]  
        h_i = dreal.tanh(z) 
        h.append(h_i)
    
    final_output = sum([h_i * h_i for h_i in h])    
    return final_output


def neural_CM_verifier(system, model, V_net=None, c2_V=None, 
                       tol=1e-4, number_of_jobs=32):
    print('_' * 50)
    
    if V_net is not None and c2_V is not None: 
        print(f"Verifying neural contraction metric on V<={c2_V}:")
    else:
        print(f"Verifying neural contraction metric on domain {system.domain}:")

    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    # Create dReal variables based on the number of symbolic variables
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    if V_net is not None and c2_V is not None: 
        V_learn = extract_dreal_Net(V_net, x)
        x_bound = lyznet.utils.get_bound(x, xlim, V_learn, c2_V=c2_V)
    else: 
        x_bound = lyznet.utils.get_x_bound(x, xlim)

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]
    CM_learn = extract_dreal_CMNet(model, x)
    # print("CM = ", CM_learn[0].Expand())

    d = len(system.symbolic_vars)
    M = [[dreal.Expression(0) for _ in range(d)] for _ in range(d)]

    k = 0
    for i in range(d):
        for j in range(i, d):
            M[i][j] = CM_learn[k]
            M[j][i] = CM_learn[k]  # Fill in symmetric terms
            k += 1

    print("Verifying positive definiteness...")

    constraints = []
    for n in range(1, d + 1):
        sub_matrix = [[M[i][j] for j in range(n)] for i in range(n)]
        det_sub_matrix = lyznet.utils.compute_determinant_dreal(sub_matrix)
        constraints.append(det_sub_matrix >= tol)

    positive_definiteness = dreal.And(*constraints)
    condition = dreal.logical_imply(x_bound, positive_definiteness)    
    start_time = time.time()
    result = dreal.CheckSatisfiability(
        dreal.logical_not(condition), config
        )
    if result is None:
        print("Positive definiteness verified!")
    else:
        print(result)
        print("Positive definiteness cannot be verified!")
    end_time = time.time()
    print(f"Time taken for verifying positive definiteness: " 
          f"{end_time - start_time} seconds.\n")

    print("Verifying contraction...")
    Df_x = [[fi.Differentiate(xi) for xi in x] for fi in f]
    Df_x_T = list(map(list, zip(*Df_x)))  # transpose of Df_x

    M_dot = [[dreal.Expression(0) for _ in range(d)] for _ in range(d)]

    for i in range(d):
        for j in range(d):
            lie_derivative_of_Mij = dreal.Expression(0)
            for k in range(len(x)):
                lie_derivative_of_Mij += f[k] * M[i][j].Differentiate(x[k])
            M_dot[i][j] = lie_derivative_of_Mij

    # print("M: ", M)

    M_prod1 = lyznet.utils.matrix_multiply_dreal(Df_x_T, M) 
    M_prod2 = list(map(list, zip(*M_prod1))) 
    # The following is supposed to be equivalent but some weird bugs exist with dReal
    # M_prod2 = lyznet.utils.matrix_multiply_dreal(M, Df_x)

    CM_derivative = [[M_prod1[i][j] + M_prod2[i][j] + M_dot[i][j] 
                      for j in range(d)] for i in range(d)]

    # verifying negative definiteness of CM_derivative
    constraints = []
    for n in range(1, d + 1):
        sub_matrix = [[-CM_derivative[i][j] for j in range(n)] for i in range(n)]
        det_sub_matrix = lyznet.utils.compute_determinant_dreal(sub_matrix)
        constraints.append(det_sub_matrix >= tol)

    negative_definiteness = dreal.And(*constraints)
    condition = dreal.logical_imply(x_bound, negative_definiteness)    
    start_time = time.time()
    result = dreal.CheckSatisfiability(
        dreal.logical_not(condition), config
        )
    if result is None:
        print("Negative definiteness verified!")
    else:
        print("Negative definiteness cannot be verified! Counterexample found: ")
        print(result)
    end_time = time.time()
    print(f"Time taken for verifying negative definiteness: " 
          f"{end_time - start_time} seconds.\n")

    return result


def check_homo_positive(f, x, config):
    norm = dreal.sqrt(sum(xi * xi for xi in x))
    unit_sphere = (norm == 1)
    condition = dreal.logical_imply(unit_sphere, f > 0)
    result = dreal.CheckSatisfiability(dreal.logical_not(condition), config)
    return result


def check_lambda_bound(H_k, x, k, config, lambda_value):
    norm_1 = sum(abs(xi) for xi in x)
    bound_poly = H_k - lambda_value * (norm_1**k)
    norm_2 = dreal.sqrt(sum(xi**2 for xi in x))
    unit_sphere = (norm_2 == 1)
    condition = dreal.logical_imply(unit_sphere, bound_poly >= 0)
    result = dreal.CheckSatisfiability(dreal.logical_not(condition), config)
    return result


def verify_remainder_bound(f, v_k, x, xlim, k, lambda_value, config,
                           symbolic_evaluation):
    # verify using Talor expansion that v can be dominatied by its homogenous
    # part v_k, where k is the degree, on the set V_c={v_k<=c}. 
    # v_k is homogeneous so the set V_c is star-shaped to simplify the condition

    c_upper = 1
    c_final = 0
    lambda_value = lambda_value - 1e-8
    multiindices = [alpha for alpha in product(range(k + 1), repeat=len(x)) if sum(alpha) == k]
    for alpha in multiindices:
        def Check_bound(c_value):
            derivative = f
            for i, v in enumerate(x):
                for _ in range(alpha[i]):
                    derivative = derivative.Differentiate(v)
                    # print("derivative: ", derivative)

            origin = {xi: dreal.Expression(0) for xi in x}
            if symbolic_evaluation:
                derivative_at_origin = derivative.Substitute(origin)
            else:
                derivative_at_origin = derivative.Substitute(origin).Evaluate()
            bound_expr = abs(derivative - derivative_at_origin) - lambda_value * dreal.Expression(factorial(k))
            x_bound = lyznet.utils.get_bound(x, xlim, v_k, c2_V=c_value)
            condition = dreal.logical_imply(x_bound, bound_expr <= 0)
            result = dreal.CheckSatisfiability(dreal.logical_not(condition), config)
            return result 
        c_current = lyznet.utils.bisection_glb(Check_bound, low=0, high=c_upper)
        c_upper = c_current
        c_final = c_current
    return c_final


def strict_local_verifier(x, xlim, f, V, c2, config, symbolic_evaluation):
    print('_' * 25)
    print("Exact verification of local stability using neural Lyapunov function:")
    print(f"Symbolic evaluation of derivatives enabled? {symbolic_evaluation}")

    start_time = time.time()
    # V = V.Expand()

    k = 2  
    homogeneous_terms = []  

    V_0, _ = lyznet.utils.extract_homo_polynomial_dreal(
        V, x, 0, symbolic_evaluation=symbolic_evaluation)
    V_1, _ = lyznet.utils.extract_homo_polynomial_dreal(
        V, x, 1, symbolic_evaluation=symbolic_evaluation)
    homogeneous_terms.extend([V_0, V_1])

    while True:
        V_k, _ = lyznet.utils.extract_homo_polynomial_dreal(
            V, x, k, symbolic_evaluation=symbolic_evaluation)
        # print(f"V_{k}: ", V_k)
        if k % 2 == 0:
            print(f"Checking degree {k} homogeneous part of V: ", V_k)
            if check_homo_positive(V_k, x, config) is None: 
                print(f"The degree {k} homogeneous part V_{k} is positive definite!")            
                break
            print(f"The degree {k} homogeneous part is not positive definite.")
        homogeneous_terms.append(V_k)
        k += 1

    # lie derivative of the Lyapunov function
    LV = dreal.Expression(0)
    for i in range(len(x)):
        LV += f[i] * V.Differentiate(x[i])

    j = 2
    while True:
        LV_j, _ = lyznet.utils.extract_homo_polynomial_dreal(
            LV, x, j, symbolic_evaluation=symbolic_evaluation)
        print(f"Checking degree {j} homogeneous part of LV=DV*f: ", LV_j)

        if check_homo_positive(-LV_j, x, config) is None:
            print(f"The degree {j} homogeneous part LV_{j} is negative definite!")
            break

        print(f"The degree {j} homogeneous part is not negative definite.")
        j += 2 

    def Check_bound(lambda_value):
        return check_lambda_bound(-LV_j, x, j, config, lambda_value)

    lyznet.tik()
    lambda_max = lyznet.utils.bisection_glb(Check_bound, low=0, high=1)
    print(f"Largest lambda such that LV_{j} <= -lambda*|x|_1^{j}: {lambda_max}")
    lyznet.tok()

    lyznet.tik()
    lambda_value = lambda_max - 1e-8
    c_h = verify_remainder_bound(
        -LV, V_k, x, xlim, j, lambda_value, config, symbolic_evaluation
        )
    print(f"Largest c such that V_{k}<=c implies LV <= -epsilon* |x|_1^{j}: {c_h}")
    lyznet.tok()

    def verify_inclusion(c1):
        # verify that {V<=c1} is contained within the domain and {V_k<=c_h}
        x_bound = lyznet.utils.get_bound(x, xlim, V, c2_V=c1)
        condition = dreal.logical_imply(x_bound, V_k <= c_h)

        x_boundary = dreal.logical_or(x[0] == xlim[0][0], x[0] == xlim[0][1])
        for i in range(1, len(x)):
            x_boundary = dreal.logical_or(x[i] == xlim[i][0], x_boundary)
            x_boundary = dreal.logical_or(x[i] == xlim[i][1], x_boundary)
        set_inclusion = dreal.logical_imply(
            x_bound, dreal.logical_not(x_boundary)
            )
        condition = dreal.logical_and(condition, set_inclusion)

        result = dreal.CheckSatisfiability(dreal.logical_not(condition), config)
        return result

    lyznet.tik()
    c1_V = lyznet.utils.bisection_glb(verify_inclusion, low=0, high=1)
    print(f"Lyapunov condition for local stability verified  on V <= {c1_V}.")
    lyznet.tok()

    end_time = time.time()

    print(f"Time taken for verification: {end_time - start_time} seconds.\n")

    # strictly verifiable Lyapunov function with lower degree term removed
    V = V - sum(homogeneous_terms)  
    # check values of V(0) and DV(0)
    origin = {xi: dreal.Expression(0) for xi in x}
    V_at_origin = V.Substitute(origin).Evaluate()
    grad_V = [V.Differentiate(xi) for xi in x]
    grad_V_at_origin = [derivative.Substitute(origin).Evaluate()
                        for derivative in grad_V]    
    print("V(0): ", V_at_origin)
    print("DV(0): ", grad_V_at_origin)
    return c1_V, V


def neural_verifier(system, model, c2_P=None, c1_V=0.1, c2_V=1, 
                    tol=1e-4, accuracy=1e-2, 
                    net_type=None, number_of_jobs=32, verifier=None,
                    h=None, homo_net=None, c_h=None, 
                    symbolic_evaluation=False):
    # {x^TPx<=c2_P}: target quadratic-Lyapunov level set 
    # c1_V: target Lyapunov level set if c2_P is not specified
    # c2_V: maximal level to be verified

    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    # Create dReal variables based on the number of symbolic variables
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    print('_' * 50)
    print("Verifying neural Lyapunov function:")

    if net_type == "Simple":
        V_learn = extract_dreal_SimpleNet(model, x)
    elif net_type == "Homo": 
        V_learn, norm_W = extract_dreal_HomoNet(model, x)        
    elif net_type == "Poly":
        V_learn = extract_dreal_PolyNet(model, x)
    elif net_type == "RidgePoly":
        V_learn = extract_dreal_RidgePoly(model, x)
    elif net_type == "HomoPoly":
        V_learn = extract_dreal_HomoPolyNet(model, x)
    else:
        V_learn = extract_dreal_Net(model, x)
        
    # print("V = ", V_learn.Expand())

    lie_derivative_of_V = dreal.Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V_learn.Differentiate(x[i])

    # If homogeneous verifier is called, do the following: 
    if verifier == "Homo": 
        return lyznet.homo_global_verifier(V_learn, lie_derivative_of_V, x, 
                                           config)

    if verifier == "Strict":
        # Strictly verify NN is a local Lyapunov function
        c1_V, V = strict_local_verifier(x, xlim, f, V_learn, c2_V, config,
                                        symbolic_evaluation)
        V_learn = V

    if c2_P is None and verifier == "Homo_local":
        # Quadratic LF is unavailable
        # Use a homogeneous LF to do local stability analysis 
        V_h, _ = extract_dreal_HomoNet(homo_net, x)
        target = V_h <= c_h

    if c2_P is not None:
        quad_V = dreal.Expression(0)
        for i in range(len(x)):
            for j in range(len(x)):
                quad_V += x[i] * system.P[i][j] * x[j]    
        target = quad_V <= c2_P

    start_time = time.time()

    def Check_inclusion(c1):
        x_bound = lyznet.utils.get_bound(x, xlim, V_learn, c2_V=c1)
        condition = dreal.logical_imply(x_bound, target)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
 
    if c2_P is not None:
        c1_V = lyznet.utils.bisection_glb(Check_inclusion, 0, 1, accuracy)
        c1_V_time = time.time()
        print(f"Time taken for verifying inclusion is: {c1_V_time - start_time} seconds.\n")
        print(f"Verified V<={c1_V} is contained in x^TPx<={c2_P}.")
    elif verifier == "Homo_local":
        c1_V = lyznet.utils.bisection_glb(Check_inclusion, 0, 1, accuracy)
        print(f"Verified V<={c1_V} is contained in V_h<={c_h}.")
    elif verifier == "Strict":
        print(f"Strict verifier enabled. Target level set to be V<={c1_V}.")        
    else:
        print(f"Target level set not specificed. Set it to be V<={c1_V}.")        

    c2_V = lyznet.reach_verifier_dreal(system, x, V_learn, f, c1_V, c_max=c2_V, 
                                       tol=tol, accuracy=accuracy,
                                       number_of_jobs=number_of_jobs, h=h)
    print(f"Verified V<={c2_V} will reach V<={c1_V}.")
    end_time = time.time()
    print(f"Time taken for verifying Lyapunov function of {system.name}: " 
          f"{end_time - start_time} seconds.\n")

    return c1_V, c2_V


def neural_clf_verifier(system, model, c2_P=None, c1_V=0.1, c2_V=1, 
                        tol=1e-4, accuracy=1e-2, number_of_jobs=32):

    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    V_learn = extract_dreal_Net(model, x)
    print("V = ", V_learn.Expand())

    quad_V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            quad_V += x[i] * system.P[i][j] * x[j]
    
    if c2_P is not None:
        target = quad_V <= c2_P

    start_time = time.time()

    def Check_inclusion(c1):
        x_bound = lyznet.utils.get_bound(x, xlim, V_learn, c2_V=c1)
        condition = dreal.logical_imply(x_bound, target)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
 
    print('_' * 50)
    print("Verifying neural Lyapunov function:")

    if c2_P is not None:
        c1_V = lyznet.utils.bisection_glb(Check_inclusion, 0, 1, accuracy)
        print(f"Verified V<={c1_V} is contained in x^TPx<={c2_P}.")
    else:
        print(f"Target level set not specificed. Set it to be V<={c1_V}.")        

    c2_V = lyznet.clf_reach_verifier_dreal(system, x, V_learn, c1_V, 
                                           c_max=c2_V, tol=tol, 
                                           accuracy=accuracy,
                                           number_of_jobs=number_of_jobs)    

    print(f"Verified V<={c2_V} will reach V<={c1_V}.")
    end_time = time.time()
    print(f"Time taken for verifying control Lyapunov function of {system.name}: " 
          f"{end_time - start_time} seconds.\n")

    return c1_V, c2_V
