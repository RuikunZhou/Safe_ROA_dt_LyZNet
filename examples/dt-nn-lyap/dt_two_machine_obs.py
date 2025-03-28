import sympy 
import lyznet
import numpy as np
import time
import torch

lyznet.utils.set_random_seed()

x1, x2 = sympy.symbols('x1 x2')

delta = 3.141592653589793/3
dt = 0.1
v_max = 2000

f = [x1 + dt*x2, x2 + dt*(-0.5*x2 - (sympy.sin(x1+delta)-sympy.sin(delta)))]
# domain = [[-2.0, 3.0], [-3.0, 1.5]]
domain = [[-1.0, 1.5], [-1.5, 1.0]]


def h(x):
    # h < 1 defines the safe set, avoiding two obstacles
    obstacle1 = 1 + 1/8**2 - ((x[0] - 0.25)**2 + (x[1] - 0.25)**2) 
    obstacle2 = 1 + 1/8**2 - ((x[0] - 0.25)**2 + (x[1] + 0.25)**2) 
    return max(obstacle1, obstacle2)


def h_torch(x):
    # h < 1 defines the safe set, avoiding two obstacles
    obstacle1 = 1 + 1/8**2 - ((x[:, 0] - 0.25)**2 + (x[:, 1] - 0.25)**2) 
    obstacle2 = 1 + 1/8**2 - ((x[:, 0] - 0.25)**2 + (x[:, 1] + 0.25)**2) 
    return torch.max(torch.stack([obstacle1, obstacle2], dim=1), dim=1).values


h_sympy = [
    1 + 1/8**2 - ((x1 - 0.25)**2 + (x2 - 0.25)**2),  
    1 + 1/8**2 - ((x1 - 0.25)**2 + (x2 + 0.25)**2),  
]

sys_name = f"dt_two_machine_power_obs_v_max_{v_max}"

system = lyznet.DiscreteDynamicalSystem(f, domain, sys_name)

# compute c_max for c2_P
P_inv = np.linalg.inv(system.P)

c_values = []
for i in range(len(domain)):
    lower_bound = domain[i][0]
    upper_bound = domain[i][1]
    min_val = min(-lower_bound, upper_bound)
    c_i = (min_val ** 2) / P_inv[i, i]
    c_values.append(c_i)

c2_max = min(c_values)

print("c2_max: ", c2_max)

data = lyznet.generate_dts_data(system, n_samples=3000, v_max=v_max, h=h)

net, model_path = lyznet.neural_learner(
    system, data=data, lr=0.001, layer=2, width=30, num_colloc_pts=300000, 
    max_epoch=20, loss_mode="DTS_Zubov", v_max=v_max, 
    # net_type="ReLU",
    h=h_torch
    )

time1 = time.time()
c1_P = 0.2117
c2_P = lyznet.quadratic_reach_verifier(system, c1_P, c_max=c2_max, h=h_sympy)

print("c1_P: ", c1_P)
print("c2_P: ", c2_P)

lyznet.plot_V(system, c1_P=c1_P, c2_P=c2_P, h=h)
c1_V, c2_V = lyznet.neural_verifier(system, net, c2_P, h=h_sympy)

# # c2_V = 0.4736

lyznet.plot_V(system, net, model_path, c2_P=c2_P, c2_V=c2_V, h=h)
