# This code corresponds to the experiments in Section J.

import jax
from jax import numpy as jnp
from jax import grad, jit
import numpy as np
import matplotlib.pyplot as plt

def gamma_arrays(gamma, p, T):
  """Generate gamma array ~ Bin(T, p)."""

  return gamma * np.random.binomial(1, p, size=(T,))


def lambda_fn(t, i, H, gammas):
  """Calculate lambda."""

  if i > 0 and i <= t:
    return np.prod([1-gammas[t-1-j] for j in range(1,i)]) * gammas[t-i-1]
  elif i > 0:
    return 0
  return 1 - np.sum([lambda_fn(t, j, H, gammas) for j in range(1, H+1)])

def lambda_bar_fn(t, i, gammas):
  """Calculate lambda bar."""

  return np.prod([1-gammas[t-1-j] for j in range(1, i+1)])

def get_action(step, p, M, H, gammas, wlist):

  lambda_0 = lambda_fn(step, 0, H, gammas)
  lambdas = jnp.array([lambda_fn(step, i, H, gammas) for i in range(1, H+1)])
  weighted_w = jnp.array([lambdas[i-1] * wlist[step-i] for i in range(1, H+1)])
  mv_prod = jnp.einsum('hij,hj->i', M, weighted_w)
  u = lambda_0 * p + mv_prod
  return u

def get_x(p, M, x_init, T, system, params, wlist):

  gammas, H, _ = params
  x_cur = x_init
  for t in range(1, T):
    u = get_action(T, p, M, H, gammas, wlist)
    new_x = (1-gammas[t-1]) * system(x_cur, u) + gammas[t-1] * wlist[t-1]
    x_cur = new_x
  return x_cur

def surrogate_loss(p, M, x_init, step, system, params, cost_fn, wlist):

  x = get_x(p, M, x_init, step, system, params, wlist)
  gammas, _, _ = params
  u = get_action(step, p, M, H, gammas, wlist)
  return cost_fn(x, u)

def omd_step(eta, state, grad):
  """One step omd update."""

  p, M = state
  grad_p, grad_M = grad
  new_p = p * jnp.exp(-eta * grad_p)
  new_M = M * jnp.exp(-eta * grad_M)
  new_p /= jnp.sum(new_p)
  new_M = jax.vmap(jax.vmap(simplex_proj, in_axes=0), in_axes=2, out_axes=2)(new_M)
  return new_p, new_M

def random_noise_vec(d):

  vec = np.zeros(d)
  sum = 0
  for i in range(0,d):
    vec[i] = np.random.rand()
    sum += vec[i]
  vec /= sum
  return vec

def w_generator(d,T):

  wlist=[]
  for i in range(T):
    wlist.append(random_noise_vec(d))
  return wlist


def training(cost_fns, system, params, inits, wlist, T):

  gammas, H, lr = params
  p, M, x = inits
  x_init = x

  costs = []
  xs = []

  u_gpc = np.ones(3) / 3

  for t in range(1, T+1):
    w = wlist[t-1]
    u = get_action(t, p, M, H, gammas, wlist)
    print(p,u)
    xs.append(x)
    cost = cost_fns[t](x, u)
    costs.append(cost)

    gamma = gammas[t-1]
    x = (1 - gamma) * system(x,u) + gamma * w

    loss_wrapper = lambda p, M: surrogate_loss(p, M, x_init, t, system, (gammas, H, lr), cost_fns[t], wlist)
    grad_fn_p = grad(loss_wrapper, argnums=0)
    grad_fn_M = grad(loss_wrapper, argnums=1)
    g_p, g_M = grad_fn_p(p, M), grad_fn_M(p, M)
    g = (g_p, g_M)
    p, M = omd_step(lr, (p, M), g)

  return costs, xs

def evaluate_policy(cost_fns, system, x_init, wlist, gammas, T, policy):

  x = x_init
  costs = []
  xs = [x]
  for i in range(1, T+1):
    w = wlist[i-1]
    gamma = gammas[i-1]
    u = policy(cost_fns[i-1], system, x)
    cost = cost_fns[i](x,u)
    costs.append(cost)
    xs.append(x)
    x = (1 - gamma) * system(x,u) + gamma * w
  return costs, xs

def null_action(cost_fn, system, x):

  return jnp.array([1.,1.,1.])/3

def best_response(cost_fn, system, x):

  bestu = jnp.array([1.,1.,1.])/3
  bestu_cost = cost_fn(system(x,bestu),bestu)
  for i in np.linspace(0, 1, 10):
    for j in np.linspace(0, 1-i, 10):
      u = jnp.array([i,j,1-i-j])
      cost = cost_fn(system(x,u),u)
      if cost < bestu_cost:
        bestu = u
        bestu_cost = cost

  return bestu

def plot_pop(xs, title):

  plt.plot([x[0] for x in xs], label='Rock (x[1])',color='red')
  plt.plot([x[1] for x in xs], label='Scissors (x[2])',color='black')
  plt.plot([x[2] for x in xs], label='Paper (x[3])',color='blue')
  plt.legend(loc='upper left', fontsize='xx-small', markerscale=0.6)
  plt.title(title)
  plt.ylim(-0.1, 1.1)
  plt.xlabel('Time Step')
  plt.ylabel('Proportion')
  plt.grid(True)
  plt.show()

def plot_costs(costs, cost_ncs, title):

  plt.plot(costs, label='Control')
  plt.plot(cost_ncs, label='Baseline')
  plt.legend(loc='upper left', fontsize='xx-small', markerscale=0.6)
  plt.title(title)
  plt.ylim(-0.1, 0.1 + max(max(costs),max(cost_ncs)))
  plt.xlabel('Time Step')
  plt.ylabel('Cost')
  plt.grid(True)
  plt.show()

def plot_all_costs(costs, best_response_costs, null_costs, title):
  
  plt.plot(costs, label='GPC-Simplex costs')
  plt.plot(best_response_costs, label='Best-response costs',linestyle='dashed')
  plt.plot(null_costs, label="Default control costs")
  plt.legend(loc='upper right', fontsize='xx-small', markerscale=0.6)
  plt.title(title)
  plt.xlabel('Time Step')
  plt.ylabel('Cost')
  plt.grid(True)
  plt.show()

def replicator_evolve(x, M, eta):
  fitness = M @ x - x @ M @ x
  x_new = jnp.multiply(x, 1 + eta * fitness)
  return x_new

# Controlled RPS with random cost function

# Initialization
T, H = 200, 5
d_control = 3

# replicator dynamics
x_init = np.array([1.0, 1.0, 1.0]) / 3
p = np.array([1.0, 1.0, 1.0])/3
M = jnp.ones((H, d_control, 3)) / d_control
lr = 5 * np.sqrt(3 * H * np.log(H)) / (10 * np.sqrt(T))
inits = (p, M, x_init)

cost_fn0 = lambda x, u: x[0]*x[0]
cost_fn1 = lambda x, u: x[0]*x[0] + u[2]*u[2]
cost_fns = [cost_fn0]
for i in range(1,T+1):
  if np.random.randint(0,2) == 0:
    cost_fns.append(cost_fn0)
  else:
    cost_fns.append(cost_fn1)

def system(x,u):
  controlled_rps = jnp.array([[0,u[0],-u[2]],[-u[0],0,u[1]],[u[2],-u[1],0]])
  return replicator_evolve(x, controlled_rps, 0.25)

wlist = w_generator(3, T)
gammas = [0] * T
params = (gammas, H, lr)
costs_gpc_rng, xs_gpc_rng = training(cost_fns, system, params, inits, wlist, T)

costs_br_rng, xs_br_rng = evaluate_policy(cost_fns, system, x_init, wlist, gammas, T, best_response)
costs_default_rng, xs_default_rng = evaluate_policy(cost_fns, system, x_init, wlist, gammas, T, null_action)

# Controlled RPS with fixed cost function

# Initialization
T, H = 100, 5
d_control = 3

# replicator dynamics
x_init = np.array([1.0, 1.0, 1.0]) / 3
p = np.array([1.0,1.0,1.0])/3
M = jnp.ones((H, d_control, 3)) / d_control
lr = 5 * np.sqrt(3 * H * np.log(H)) / (10 * np.sqrt(T))

inits = (p, M, x_init)

cost_fn0 = lambda x, u: x[0]*x[0]
cost_fns = [cost_fn0]
for i in range(1,T+1):
  cost_fns.append(cost_fn0)

def system(x,u):
  controlled_rps = jnp.array([[0,u[0],-u[2]],[-u[0],0,u[1]],[u[2],-u[1],0]])
  return replicator_evolve(x, controlled_rps, 0.25)

wlist = w_generator(3, T)
gammas = [0] * T
params = (gammas, H, lr)
costs_gpc, xs_gpc = training(cost_fns, system, params, inits, wlist, T)

costs_br_fixed, xs_br_fixed = evaluate_policy(cost_fns, system, x_init, wlist, gammas, T, best_response)
costs_default_fixed, xs_default_fixed = evaluate_policy(cost_fns, system, x_init, wlist, gammas, T, null_action)

# Fixed cost plots
plot_all_costs(costs_gpc, costs_br_fixed, costs_default_fixed, "Comparison of losses over time")
plot_pop(xs_gpc, "Population evolution with GPC-Simplex")

def window_sum(v,wlen):
  sums = np.cumsum(v)
  window_sums = []
  for i in range(len(v)):
    val = sums[i]
    if i >= wlen:
      val -= sums[i-wlen]
    val /= min(i+1, wlen)
    window_sums.append(val)
  return window_sums

gps_ws = window_sum(costs_gpc_rng, 15)
br_ws = window_sum(costs_br_rng, 15)
default_ws = window_sum(costs_default_rng, 15)

# Random cost plots
plot_all_costs(gps_ws, br_ws, default_ws, "Comparison of sliding-window losses over time")
plot_pop(xs_gpc_rng, "Population evolution with GPC-Simplex")

