import sympy as sp
import numpy as np
import lyznet

lyznet.utils.set_random_seed()


x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
symbolic_vars = sp.Matrix([x1, x2, x3, x4])
dt = 0.05
v_max = 20000

# Parameters
alpha_1, alpha_2 = 1, 1
beta_1, beta_2 = 0.5, 0.5
D_1, D_2 = 0.4, 0.5

# Define the system equations
dx1 = x2
dx2 = -alpha_1 * sp.sin(x1) - beta_1 * sp.sin(x1 - x3) - D_1 * x2
dx3 = x4
dx4 = -alpha_2 * sp.sin(x3) - beta_2 * sp.sin(x3 - x1) - D_2 * x4

# Discretize the system
f_dt = [
    x1 + dt * dx1,
    x2 + dt * dx2,
    x3 + dt * dx3,
    x4 + dt * dx4
]

print("f_dt: ", f_dt)

xlim = 3.5
domain = [[-xlim, xlim]] * 4
sys_name = f"power_system_v_max_{v_max}_{domain[0]}_dt_{dt}"
power_system = lyznet.DiscreteDynamicalSystem(f_dt, domain, sys_name)

print("System dynamics: x+ = ", power_system.symbolic_f)
print("Domain: ", power_system.domain)

# compute c_max for c2_P
P_inv = np.linalg.inv(power_system.P)

c_values = []
for i in range(len(domain)):
    lower_bound = domain[i][0]
    upper_bound = domain[i][1]
    min_val = min(-lower_bound, upper_bound)
    c_i = (min_val ** 2) / P_inv[i, i]
    c_values.append(c_i)

c2_max = min(c_values)

# Generate data (needed for data-augmented learner)
data = lyznet.generate_dts_data(power_system, n_samples=20000, v_max=v_max)

# Call the neural lyapunov learner
net, model_path = lyznet.neural_learner(
    power_system, data=data, lr=0.001, layer=2, width=50, num_colloc_pts=1000000, batch_size=1000,
    max_epoch=20, loss_mode="DTS_Zubov", v_max=v_max)

c1_P = 1.3375
print("c1_P computed is: ", c1_P)

c2_P = lyznet.quadratic_reach_verifier(power_system, c1_P, c_max=c2_max, accuracy=1)

# # c2_V = 0.4736

# lyznet.plot_V(power_system, net, model_path, c2_P=c2_P, c2_V=c2_V)
