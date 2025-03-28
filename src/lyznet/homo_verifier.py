import lyznet
import dreal
import sympy
import time 
from itertools import product
from sympy import factorial


def homo_global_verifier(V_learn, lie_derivative_of_V, x, config, plot=False):
    print('_' * 50)
    print("Verifying global stability using homogenous Lyapunov function...")

    norm = dreal.sqrt(sum(xi * xi for xi in x))
    unit_sphere = (norm == 1)

    # Define verification conditions
    condition_V = dreal.logical_imply(unit_sphere, V_learn >= 1e-7)
    condition_dV = dreal.logical_imply(
        unit_sphere, lie_derivative_of_V <= -1e-7
    )
    condition = dreal.logical_and(condition_V, condition_dV)

    # Perform verification
    start_time = time.time()
    result = dreal.CheckSatisfiability(
        dreal.logical_not(condition), config
    )
    end_time = time.time()

    # Print results
    if result is None:
        print("Global stability verified for homogeneous vector field!")
        success = True
    else:
        print(result)
        print("Stability cannot be verified for homogeneous vector field!")
        success = False

    print(f"Time taken for verification: {end_time - start_time} seconds.\n")
    return success


def homo_local_verifier(system, tol=1e-4, accuracy=1e-4, c_max=100,
                        number_of_jobs=32, layer=1, width=10, epoch=5, 
                        homo_deg=1, plot=False, symbolic_evaluation=True):
    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    print('_' * 50)
    print("Verifying local stability using homogenous Lyapunov function:")
    print(f"Symbolic evaluation of derivatives enabled? {symbolic_evaluation}")

    start_time = time.time()

    f_h, k = lyznet.utils.extract_lowest_homogeneous_part_dreal(f, x)
    x_sympy = system.symbolic_vars
    # f_sympy = system.symbolic_f
    f_h_sympy = [lyznet.utils.dreal_to_sympy(expr, x_sympy) for expr in f_h]

    domain = system.domain
    sys_name = system.name + "_homo"
    homo_system = lyznet.DynamicalSystem(f_h_sympy, domain, sys_name)

    print("Computing local homogeneous Lyapunov function...")

    net, model_path = lyznet.neural_learner(
        homo_system, lr=0.001, num_colloc_pts=300000, max_epoch=epoch, 
        layer=layer, width=width, 
        loss_mode="Homo_Lyapunov", net_type="Homo", homo_deg=homo_deg)

    V_h, _ = lyznet.extract_dreal_HomoNet(net, x)        
    LV_h = dreal.Expression(0)
    grad_V_h = [V_h.Differentiate(xi) for xi in x]
    for i in range(len(x)):
        LV_h += f_h[i] * grad_V_h[i]

    if not homo_global_verifier(V_h, LV_h, x, config):
        return None

    def compute_lambda():
        norm_1 = sum(abs(xi) for xi in x)
        unit_sphere = (dreal.sqrt(sum(xi * xi for xi in x)) == 1)
        grad_norm = dreal.sqrt(sum(grad**2 for grad in grad_V_h))

        def Check_bound(lambda_value):
            bound_poly = -LV_h - lambda_value * grad_norm * (norm_1**k)
            condition = dreal.logical_imply(unit_sphere, bound_poly > 0)
            return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

        lambda_max = lyznet.utils.bisection_glb(Check_bound, low=0, high=1)
        print(f"Largest lambda such that LV_h <= -lambda*||∇V||*|x|_1^k: {lambda_max}")
        return lambda_max

    def verify_remainder(lambda_value):
        c_upper = 1
        c_final = 0
        lambda_value = lambda_value - 1e-8
        d = len(x)
        multiindices = [alpha for alpha in product(range(k + 1), repeat=d) if sum(alpha) == k]
        for alpha in multiindices:
            for f_i in f:
                def Check_remainder_bound(c_value):
                    derivative = f_i
                    for i, v in enumerate(x):
                        for _ in range(alpha[i]):
                            derivative = derivative.Differentiate(v)
                    origin = {xi: dreal.Expression(0) for xi in x}
                    if symbolic_evaluation:
                        derivative_at_origin = derivative.Substitute(origin)
                    else:
                        derivative_at_origin = derivative.Substitute(origin).Evaluate()
                    bound_expr = abs(derivative - derivative_at_origin) - lambda_value * dreal.Expression(factorial(k)) / dreal.sqrt(d)
                    
                    if c_value is not None:
                        x_bound = lyznet.utils.get_bound(x, xlim, V_h, c2_V=c_value)
                    else:
                        print("Bisection accuracy exchausted without result!")
                        return None 
                    condition = dreal.logical_imply(x_bound, bound_expr <= 0)
                    result = dreal.CheckSatisfiability(dreal.logical_not(condition), config)
                    return result
                
                c_current = lyznet.utils.bisection_glb(Check_remainder_bound, 
                                                       low=0, high=c_upper, 
                                                       accuracy=accuracy)
                c_upper = c_current
                c_final = c_current
        return c_final

    lyznet.tik()
    lambda_max = compute_lambda()
    lyznet.tok()
    
    lyznet.tik()
    c1_h = verify_remainder(lambda_max)    
    print(f"Local stability verified on V_h<={c1_h}!")
    lyznet.tok()

    end_time = time.time()

    print(f"Time taken for verification: {end_time - start_time} seconds.\n")

    if plot:
        lyznet.plot_V(system, net, model_path, c1_V=c1_h, phase_portrait=True)

    return c1_h, net, model_path


def homo_reach_verifier(system, V_h_net, model_path, c1_h, 
                        accuracy=1e-4, tol=1e-4, 
                        number_of_jobs=32, c_max=100, plot=False):
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    V_h, _ = lyznet.extract_dreal_HomoNet(V_h_net, x)        

    V_h_normalized = V_h / dreal.Expression(c1_h)

    lyznet.tik()

    c2_h_normalized = lyznet.reach_verifier_dreal(system, x, V_h_normalized, f, 1.0, c_max=c_max, 
                                                  tol=tol, accuracy=accuracy,
                                                  number_of_jobs=number_of_jobs)

    c2_h = c2_h_normalized * c1_h

    print(f"Level set V_h<={c2_h} verified by reachability!")

    lyznet.tok()
    
    if plot:
        lyznet.plot_V(system, V_h_net, c1_V=c1_h, c2_V=c2_h, phase_portrait=True)
   
    return c2_h, V_h_net, model_path
