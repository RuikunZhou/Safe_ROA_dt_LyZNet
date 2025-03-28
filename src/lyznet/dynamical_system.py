import numpy as np
import sympy
import scipy
import torch
import lyznet


class DynamicalSystem:
    def __init__(self, f, domain=None, name="", symbolic_vars=None, 
                 Q=None, ep=None):
        self.f = f 
        if domain is None:
            domain = [[-1.0, 1.0]] * len(f)        
        self.domain = domain
        self.name = name
        self.system_type = "DynamicalSystem"

        self.symbolic_f = sympy.Matrix(f) if isinstance(f, list) else f

        if symbolic_vars is not None:
            self.symbolic_vars = list(symbolic_vars)
        else:
            n = len(self.symbolic_f)
            self.symbolic_vars = [sympy.symbols(f'x{i+1}') for i in range(n)]

        self.ep = ep
        if self.ep is not None:
            self.shift_to_origin(self.ep)

        self.A = self.compute_linearization()  
        self.Q = Q if Q is not None else np.eye(len(self.symbolic_f))
        self.P = self.compute_quadratic_lyapunov_function() 

        self.f_numpy = sympy.lambdify(
            self.symbolic_vars, self.symbolic_f, modules=['numpy']
            )

        # self.f_numpy = [sympy.lambdify(self.symbolic_vars, fi, modules='numpy') 
        #                 for fi in self.symbolic_f]

        self.f_torch = [sympy.lambdify(self.symbolic_vars, fi, modules=[torch]) 
                        for fi in self.symbolic_f]

    def f_numpy_vectorized(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.squeeze(np.array([self.f_numpy(*sample) for sample in x]))

    def f_torch_vectorized(self, x):
        result = []
        for func in self.f_torch:
            element = func(*x.T)
            if isinstance(element, (int, float)):
                element = torch.tensor(element).expand(x.shape[0])
            result.append(element)
        return torch.stack(result, dim=1)

    def compute_quadratic_lyapunov_function(self):
        if not np.all(np.real(np.linalg.eigvals(self.A)) < 0):
            print("Skipping solving Lyapunov equation: "
                  "A is not Hurwitz.")
            return None
        if not np.all(np.linalg.eigvals(self.Q) > 0):
            print("Skipping solving Lyapunov equation: "
                  "Q is not positive definite.")
            return None   
        self.P = scipy.linalg.solve_continuous_lyapunov(self.A.T, -self.Q)
        return self.P

    # def _shift_to_origin(self, f, ep):
    #     substitutions = {var: var + val for var, 
    #                      val in zip(self.symbolic_vars, ep)}
    #     print(f"Equilibrium point {self.ep} shifted to the origin.")
    #     return sympy.simplify(f.subs(substitutions))

    def shift_to_origin(self, ep):
        """
        Shift the dynamics f(x) to the origin based on equilibrium point ep.

        Parameters:
        - ep: List of equilibrium points for state variables.
        """
        if len(ep) != len(self.symbolic_vars):
            raise ValueError("Length of ep must match the number of state variables.")

        # Shift state variables
        substitutions = {var: var + val for var, val in zip(self.symbolic_vars, ep)}
        self.symbolic_f = sympy.simplify(self.symbolic_f.subs(substitutions))

        # Adjust domain
        self.domain = [[bound[0] - e, bound[1] - e] for bound, e in zip(self.domain, ep)]

        # Update equilibrium point
        self.ep = ep

        # Recompute linearization and Lyapunov matrix if necessary
        self.A = self.compute_linearization()
        self.P = self.compute_quadratic_lyapunov_function()

        print(f"Equilibrium point {self.ep} shifted to the origin.")

    def compute_linearization(self):
        jacobian_np = np.array(
            lyznet.utils.compute_jacobian_np_dreal(
                self.symbolic_f, self.symbolic_vars
            )
        )
        return jacobian_np


class ControlAffineSystem(DynamicalSystem):
    def __init__(self, f, g, domain=None, name="", Q=None, R=None, 
                 u_func_numpy=None, 
                 u_expr=None, u_domain=None,
                 x_ep=None, u_ep=None):
        super().__init__(f, domain, name, Q=Q)
        self.system_type = "ControlAffineSystem"
        if u_domain is None:
            u_domain = [[-1.0, 1.0]] * g.shape[1]
        self.u_domain = u_domain            
        self.g = g
        self.symbolic_g = sympy.Matrix(g) 
        origin = {var: 0 for var in self.symbolic_vars}
        self.B = np.array(self.symbolic_g.subs(origin)).astype(np.float64)

        self.g_numpy = sympy.lambdify(
            self.symbolic_vars, self.symbolic_g, modules=['numpy']
            )

        self.g_torch = [[sympy.lambdify(self.symbolic_vars, 
                                        self.symbolic_g[i, j], 
                                        modules=[torch]) 
                        for j in range(self.symbolic_g.shape[1])]
                        for i in range(self.symbolic_g.shape[0])]

        self.R = (np.array(R).astype(np.float64) if R is not None else 
                  np.eye(self.B.shape[1]))
        self.Q = (np.array(Q).astype(np.float64) if Q is not None else 
                  np.eye(self.A.shape[1]))

        self.P, self.K = self.compute_lqr_gain()

        # adding expressions/functions for controller (initilized to linear)
        if u_expr is None and self.K is not None:
            self.u_expr = sympy.Matrix(self.K) * sympy.Matrix(self.symbolic_vars)
        else:
            self.u_expr = u_expr

        if u_func_numpy is None:
            self.u_func_numpy = self.default_u_func_numpy
        else:
            self.u_func_numpy = u_func_numpy

        # use vectorized numpy functions to handle batch inputs
        self.closed_loop_f_numpy = lyznet.get_closed_loop_f_numpy(
                                self, self.f_numpy_vectorized, 
                                self.g_numpy_vectorized, 
                                self.u_func_numpy 
                                )

        if x_ep is not None and u_ep is not None:
            self.shift_to_origin(self, x_ep, u_ep)

    def shift_to_origin(self, x_ep, u_ep):
        """
        Shift the dynamics f(x) + g(x)u to the origin based on equilibrium points (x_ep, u_ep).

        Parameters:
        - x_ep: List of equilibrium points for state variables.
        - u_ep: List of equilibrium points for control variables.
        """
        # Validate inputs
        if len(x_ep) != len(self.symbolic_vars):
            raise ValueError("Length of x_ep must match the number of state variables.")
        if len(u_ep) != self.symbolic_g.shape[1]:
            raise ValueError("Length of u_ep must match the number of control variables.")

        # Shift state variables
        x_substitutions = {var: var + val for var, val in zip(self.symbolic_vars, x_ep)}
        shifted_f = self.symbolic_f.subs(x_substitutions)
        shifted_g = self.symbolic_g.subs(x_substitutions)

        # Add control effect to f (f(x+x_ep) + g(x+x_ep)*u_ep)
        u_vars = [sympy.symbols(f'u{i+1}') for i in range(shifted_g.shape[1])]
        control_substitutions = {u_var: u_val for u_var, u_val in zip(u_vars, u_ep)}
        new_f = sympy.simplify(shifted_f + shifted_g * sympy.Matrix(u_vars).subs(control_substitutions))

        # Update g(x+x_ep) (g(x+x_ep)*u)
        new_g = sympy.simplify(shifted_g)

        # Update attributes
        self.symbolic_f = new_f
        self.symbolic_g = new_g

        # Adjust domains
        self.domain = [[bound[0] - e, bound[1] - e] for bound, e in zip(self.domain, x_ep)]
        self.u_domain = [[bound[0] - e, bound[1] - e] for bound, e in zip(self.u_domain, u_ep)]

        # Recompute linearization at the new origin
        self.A = self.compute_linearization()
        origin = {var: 0 for var in self.symbolic_vars}
        self.B = np.array(self.symbolic_g.subs(origin)).astype(np.float64)

        # Recompute P and K
        self.P, self.K = self.compute_lqr_gain()

        print(f"Equilibrium points x_ep: {x_ep}, u_ep: {u_ep} shifted to origin.")

    def g_torch_vectorized(self, x):
        batch_size = x.shape[0]

        def process_element(func):
            result = func(*x.T)
            # workaround for issue with lamdify for torch mode when g has 
            # constant terms ; slower than an explicitly defined g_torch func
            if not isinstance(result, torch.Tensor):
                return torch.tensor(result, dtype=torch.float32).expand(batch_size, 1)
            elif result.ndim == 1:
                return result.unsqueeze(1).float()

        result = [torch.cat([process_element(func) for func in row], dim=1) 
                  for row in self.g_torch]
        return torch.stack(result, dim=1).float()

    def default_u_func_numpy(self, x):
        u_expr_numpy = sympy.lambdify(self.symbolic_vars, self.u_expr, 
                                      modules=['numpy'])
        x = np.atleast_2d(x)
        u_value_transposed = np.transpose(u_expr_numpy(*x.T))
        u_value = np.transpose(u_value_transposed, (0, 2, 1))
        output = np.squeeze(u_value, axis=-1)
        return output

    def g_numpy_vectorized(self, samples):
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        return np.array([self.g_numpy(*sample) for sample in samples])

    def compute_lqr_gain(self):
        if not np.all(np.linalg.eigvals(self.R) > 0):
            print("Skipping solving Lyapunov equation: "
                  "R is not positive definite.")
            return None, None   

        if lyznet.utils.is_controllable(self.A, self.B):
            print("The system is controllable.")
        elif lyznet.utils.is_stabilizable(self.A, self.B):
            print("The system is not controllable, but stabilizable.")
        else:
            print("The system is not stabilizable. Skipping lqr computation.")
            return None, None

        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = - np.linalg.inv(self.R) @ (self.B.T @ P)
        return P, K


class DiscreteDynamicalSystem(DynamicalSystem):
    def __init__(self, f, domain, name, symbolic_vars=None, Q=None):
        super().__init__(f, domain, name, symbolic_vars=symbolic_vars, Q=Q)
        self.system_type = "DiscreteDynamicalSystem"

    def compute_quadratic_lyapunov_function(self):
        if not np.all(np.abs(np.linalg.eigvals(self.A)) < 1):
            print("Skipping solving Lyapunov equation: "
                  "A is not Schur stable.")
            return None
        if not np.all(np.linalg.eigvals(self.Q) > 0):
            print("Skipping solving Lyapunov equation: "
                  "Q is not positive definite.")
            return None   
        self.P = scipy.linalg.solve_discrete_lyapunov(self.A.T, self.Q)
        return self.P
