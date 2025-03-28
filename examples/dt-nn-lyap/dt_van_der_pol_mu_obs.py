import sympy 
import lyznet
import numpy as np
import time

lyznet.utils.set_random_seed()

# Define dynamics
mu = 1.0
x1, x2 = sympy.symbols('x1 x2')
dt = 0.1
v_max = 2000
f_vdp = [x1 - dt*x2, x2 + dt*(x1 - mu * (1 - x1**2) * x2)] 
domain = [[-2.5, 2.5], [-3.5, 3.5]]
sys_name = f"dts_van_der_pol_obs_v_max_{v_max}.py"
vdp_system = lyznet.DiscreteDynamicalSystem(f_vdp, domain, sys_name)


def h(x):
    # h < 1 defines the safe set, avoiding three obstacles
    obstacle = 1 + 1/4 - ((x[0] - 1)**2 + (x[1] - 1)**2) / 0.5**2  
    return obstacle


def h_torch(x):
    # h < 1 defines the safe set, avoiding an obstacle
    obstacle = 1 + 1/4 - ((x[:, 0] - 1)**2 + (x[:, 1] - 1)**2) / 0.5**2  
    return obstacle


h_sympy = [
    1 + 1/4 - ((x1 - 1)**2 + (x2 - 1)**2) / 0.5**2, 
]

print("System dynamics: x+ = ", vdp_system.symbolic_f)
print("Domain: ", vdp_system.domain)

# compute c_max for c2_P
P_inv = np.linalg.inv(vdp_system.P)

c_values = []
for i in range(len(domain)):
    lower_bound = domain[i][0]
    upper_bound = domain[i][1]
    min_val = min(-lower_bound, upper_bound)
    c_i = (min_val ** 2) / P_inv[i, i]
    c_values.append(c_i)

c2_max = min(c_values)

print("c2_max: ", c2_max)


# # Generate data (needed for data-augmented learner)
data = lyznet.generate_dts_data(vdp_system, n_samples=3000, v_max=v_max, h=h)

# # Call the neural lyapunov learner
net, model_path = lyznet.neural_learner(
    vdp_system, data=data, lr=0.001, layer=2, width=30, num_colloc_pts=300000, 
    max_epoch=20, loss_mode="DTS_Zubov", v_max=v_max, h=h_torch,
    # net_type="ReLU"
    )


c1_P = 1.7784
c2_P = lyznet.quadratic_reach_verifier(vdp_system, c1_P, c_max=c2_max, h=h_sympy)

# lyznet.plot_V(vdp_system, c1_P=c1_P, c2_P=c2_P, h=h)

# Call the neural lyapunov verifier
c1_V, c2_V = lyznet.neural_verifier(vdp_system, net, c2_P, c2_V=10, h=h_sympy)

# c2_V = 0.728

lyznet.plot_V(vdp_system, net, model_path, c2_P=c2_P, c2_V=c2_V, h=h)