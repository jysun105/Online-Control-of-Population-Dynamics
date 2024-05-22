import jax
from jax import numpy as jnp
from jax import grad, jit
import numpy as np
import matplotlib.pyplot as plt


def gamma_arrays(gamma, p, T):
  """Generate gamma arrays with Bin(T, p)."""

  return gamma * np.random.binomial(1, p, size=(T,))


def random_noise_vec(d):

  vec = np.zeros(d)
  sum = 0
  for i in range(0, d):
    vec[i] = np.random.rand()
    sum += vec[i]
  vec = vec / sum
  return vec

def w_generator(d, T):

  wlist = []
  for i in range(T):
    wlist.append(random_noise_vec(d))
  return wlist


def lambda_fn(t, i, H, gammas):
  """Calculate lambdas."""

  if i > 0 and i <= t:
    return np.prod([1-gammas[t-1-j] for j in range(1,i)]) * gammas[t-i-1]
  elif i > 0:
    return 0
  return 1 - np.sum([lambda_fn(t, j, H, gammas) for j in range(1, H+1)])


def lambda_bar_fn(t, i, gammas):
  """Calculate lambda bar."""

  return np.prod([1-gammas[t-1-j] for j in range(1, i+1)])


def get_action(step, p, M, H, gammas, wlist):
  """Compute control."""

  lambda_0 = lambda_fn(step, 0, H, gammas)
  lambdas = jnp.array([lambda_fn(step, i, H, gammas) for i in range(1, H+1)])
  weighted_w = jnp.array([lambdas[i-1] * wlist[step-i] for i in range(1, H+1)])
  mv_prod = jnp.einsum('hij,hj->i', M, weighted_w)
  u = lambda_0 * p + mv_prod
  return u


def get_state(step, p, M, x_init, B, params, wlist):

  gammas, H, sir_params = params
  x_cur = x_init
  for t in range(1, step):
    w = wlist[t-1]
    gamma = gammas[t-1]
    A = get_A(x_cur, sir_params)
    u = get_action(t, p, M, H, gammas, wlist)
    new_x = one_step(x_cur, p, w, gamma, sir_params, A, B)
    x_cur = new_x
  return x_cur


def surrogate_loss(p, M, x_init, step, B, params, cost_fn, wlist):

  gammas, _, _ = params
  x = get_state(step, p, M, x_init, B, params, wlist)
  u = get_action(step, p, M, H, gammas, wlist)
  return cost_fn(x, u)


def omd_step(eta, state, grad):
  """One step omd update."""

  p, M = state
  grad_p, grad_M = grad
  new_p = p * jnp.exp(-eta * grad_p) + 0.001
  new_M = M * jnp.exp(-eta * grad_M)
  new_p /= jnp.sum(new_p)
  new_M = jax.vmap(jax.vmap(simplex_proj, in_axes=0), in_axes=2, out_axes=2)(new_M)
  return new_p, new_M


def one_step(x, u, w, gamma, sir_params, A, B):
  """One step state evolution."""

  beta, _, _ = sir_params
  delta = beta * x[0] * x[1]
  x = (1 - gamma) * (A @ x + delta * B @ u) + gamma * w
  return x


def training(x_init, p_init, B, params, cost_fn, wlist):
  """Controller wrapper."""

  gammas, H, sir_params = params
  beta, q, pi = sir_params
  T = len(wlist)
  A_init = get_A(x_init, sir_params)
  x, x_q, x_nq = x_init, x_init, x_init
  A, A_q, A_nq = A_init, A_init, A_init

  u_q = np.array([1, 0])
  u_nq = np.array([0, 1])

  p = p_init
  M = jnp.ones((H, 2, 3)) / 2
  lr = 30 * np.sqrt(3 * H * np.log(H)) / (10 * np.sqrt(T))

  costs, costs_q, costs_nq = [], [], []
  us = []

  for t in tqdm(range(1, T+1)):
    w = wlist[t-1]
    u = get_action(t, p, M, H, gammas, wlist)
    us.append(u)
    cost = cost_fn(x, u)
    costs.append(cost)
    costs_q.append(cost_fn(x_q, u_q))
    costs_nq.append(cost_fn(x_nq, u_nq))
    gamma = gammas[t-1]
    x = one_step(x, u, w, gamma, sir_params, A, B)
    x_q = one_step(x_q, u_q, w, gamma, sir_params, A_q, B)
    x_nq = one_step(x_nq, u_nq, w, gamma, sir_params, A_nq, B)

    psuedo_cost = surrogate_loss(p, M, x_init, t, B, params, cost_fn, wlist)
    loss_wrapper = lambda p, M: surrogate_loss(p, M, x_init, t, B, params, cost_fn, wlist)
    grad_fn_p, grad_fn_M = grad(loss_wrapper, argnums=0), grad(loss_wrapper, argnums=1)
    g_p, g_M = grad_fn_p(p, M), grad_fn_M(p, M)
    p, M = omd_step(lr, (p, M), (g_p, g_M))
    A, A_q, A_nq = get_A(x, sir_params), get_A(x_q, sir_params), get_A(x_nq, sir_params)

  return costs, costs_q, costs_nq, us


# Plotting functions.


def plot_cost(costs, costs_q, costs_nq, sir_params, cost_params):
  beta, q, pi = sir_params
  c2, c3 = cost_params
  plt.plot(costs, label='costs')
  plt.plot(costs_q, label='costs (full preventions)')
  plt.plot(costs_nq, label='costs (no preventions)')
  plt.title(r'Cost Over Time, $\beta$=%.2f, $\theta$=%.2f, $\xi$=%.3f, $c_2$=%d, $c_3$=%d'%(beta, q, pi, c2, c3))
  plt.legend(loc='upper right', fontsize='small', markerscale=0.6)
  plt.xlabel('Time Step')
  plt.ylabel('Cost')
  plt.grid(True)
  plt.show()


def plot_control(us, sir_params, cost_params):
  beta, q, pi = sir_params
  c2, c3 = cost_params
  plt.plot([u[1] for u in us])
  plt.title(r'$u(2)$ Over Time, $\beta$=%.2f, $\theta$=%.2f, $\xi$=%.3f, $c_2$=%d, $c_3$=%d'%(beta, q, pi, c2, c3))
  plt.xlabel('Time Step')
  plt.ylabel(r'$u(2)$')
  plt.grid(True)
  plt.show()


def plot_cumcost(costs, costs_q, costs_nq, sir_params, cost_params):
  beta, q, pi = sir_params
  c2, c3 = cost_params
  plt.plot(np.cumsum(costs), label='cumulative costs')
  plt.plot(np.cumsum(costs_q), label='cumulative costs (full preventions)')
  plt.plot(np.cumsum(costs_nq), label='cumulative costs (no preventions)')
  plt.title(r'Cumulative Cost Over Time, $\beta$=%.2f, $\theta$=%.2f, $\xi$=%.3f, $c_2$=%d, $c_3$=%d'%(beta, q, pi, c2, c3))
  plt.legend(loc='upper left', fontsize='small', markerscale=0.6)
  plt.xlabel('Time Step')
  plt.ylabel('Cumulative cost')
  plt.grid(True)
  plt.show()


x_init = np.array([0.9, 0.1, 0])
p_init = np.array([0.1, 0.9])
B = np.array([[1, 0], [0, 1], [0, 0]])
T, H = 200, 5


# No noise: Fig. 1, 5, 6
gammas = gamma_arrays(0, 0, T)
wlist = w_generator(3, T)
sir_params_ls = [(0.5, 0.03, 0.005), (0.3, 0.05, 0.001)]
cost_params_ls = [(1, 1), (1, 5), (1, 10), (1, 20)]
for (sir_params, cost_params) in itertools.product(sir_params_ls, cost_params_ls):
  params = (gammas, delta, H, sir_params)
  c2, c3 = cost_params
  cost_fn = lambda x, u: c3 * (x.T @ np.array([0, 1, 0])) ** 2 + c2 * x[0] * (u.T @ np.array([1, 0]))
  costs0, costs_q0, costs_nq0, us0 = training(x_init, p_init, B, params, cost_fn, wlist)
  plot_cost(costs0, costs_q0, costs_nq0, sir_params, cost_params)
  plot_control(us0, sir_params, cost_params)
  plot_cumcost(costs0, costs_q0, costs_nq0, sir_params, cost_params)


# With noise: Fig. 4

sir_params = (0.5, 0.03, 0.005)
cost_params = (1, 5)
gammas = gamma_arrays(0.01, 0.2, T)
params = (gammas, delta, H, sir_params)
cost_fn = lambda x, u: 5 * (x.T @ np.array([0, 1, 0])) ** 2 + x[0] * (u.T @ np.array([1, 0]))
wlist = [np.array([0, 1, 0])] * T
costs, costs_q, costs_nq, us = training(x_init, p_init, B, params, cost_fn, wlist)
plot_cost(costs, costs_q, costs_nq, sir_params, cost_params)
plot_control(us, sir_params, cost_params)
plot_cumcost(costs, costs_q, costs_nq, sir_params, cost_params)

gammas = [0.01] * T
params = (gammas, delta, H, sir_params)
cost_fn = lambda x, u: 5 * (x.T @ np.array([0, 1, 0])) ** 2 + x[0] * (u.T @ np.array([1, 0]))
wlist = w_generator(3, T)
costs, costs_q, costs_nq, us = training(x_init, p_init, B, params, cost_fn, wlist)
plot_cost(costs, costs_q, costs_nq, sir_params, cost_params)
plot_control(us, sir_params, cost_params)
plot_cumcost(costs, costs_q, costs_nq, sir_params, cost_params)


