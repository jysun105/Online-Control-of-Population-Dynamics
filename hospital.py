# This script contains experiment code for Section 4.1: 
# controlling hospital flow, reproducing a study by Ketcheson (2021)

import jax
from jax import numpy as jnp
from jax import grad
import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm
from tqdm import tqdm
from functools import partial
import itertools
from scipy.integrate import solve_ivp
from scipy.special import expit

# Import code from Ketcheson 2021 github repository
!wget https://raw.githubusercontent.com/ketch/SIR-control-code/main/code/SIR_control.py -O SIR_control.py
from SIR_control import solve_pmp, plot_timeline, SIR_forward

# Important control and LDS related functions

def get_A(x, sir_params):
  beta, q, pi = sir_params
  return jnp.array([[1-beta*x[1], 0, pi], [0, 1-q, 0], [0, q, 1-pi]])

def get_state(step, p, x_init, params):
  H, sir_params = params
  x_cur = x_init
  for t in range(1, step):
    A = get_A(x_cur, sir_params)
    new_x = one_step(x_cur, p, sir_params, A)
    x_cur = new_x
  return x_cur

def surrogate_loss(p, x_init, step, params, cost_fn):
  x = get_state(step, p, x_init, params)
  return cost_fn(x, p)

def SIR_forward(beta=0.3, gamma=0.1, x0=0.99, y0=0.01, T=100):
    """ Model the current outbreak using the SIR model."""

    du = np.zeros(3)
    u0 = np.zeros(3)

    def f(t,u):
        du[0] = (-beta*u[1]*u[0]).item()
        du[1] = (beta*u[1]*u[0] - gamma*u[1]).item()
        return du

    # Initial values
    u0[1] = y0 # Initial infected
    u0[0] = x0

    times = np.linspace(0,T,10000)
    solution = solve_ivp(f,[0,T],u0,t_eval=times,method='RK23',max_step=0.1)
    x = solution.y[0,:]
    y = solution.y[1,:]
    t = solution.t

    return x, y, t

# Optimization and state transition steps

def omd_step(eta, p, grad, regularizer=0.001):
  """Mirro descent step."""

  new_p = p * jnp.exp(-eta * grad) + regularizer
  new_p /= jnp.sum(new_p)
  return new_p

def one_step(x, u, sir_params, A):
  """One step evolution of the system."""	

  beta, _, _ = sir_params
  delta = beta * x[0] * x[1]
  x = A @ x + delta *  np.array([[1, 0], [0, 1], [0, 0]]) @ u
  return x



def training(x_init, p_init, threshold_params, params, cost_fn, T, lr, 
	regularizer=0.01, lr_decay=0.0, regularizer_decay=0.0):
  """Controller wrapper."""

  H, sir_params = params
  beta, q, pi = sir_params
  y_max, threshold, magnitude = threshold_params
  A_init = get_A(x_init, sir_params)
  x, x_nq = x_init, x_init
  A, A_nq = A_init, A_init
  u_nq = np.array([0, 1])

  p = p_init

  costs, us, xs, x_nqs = [], [], [], []

  for t in tqdm(range(1, T+1)):
    us.append(p)
    cost = cost_fn(x, p)
    costs.append(cost)
    xs.append(x)
    x_nqs.append(x_nq)
    x, x_nq = one_step(x, p, sir_params, A), one_step(x_nq, u_nq, sir_params, A_nq)
    loss_wrapper = lambda p: surrogate_loss(p, x_init, t, params, cost_fn)
    g_p = grad(loss_wrapper)(p)
    p = omd_step(lr * (1 + np.maximum(0.0, magnitude * (x[1] - threshold * y_max))) * np.exp(-lr_decay * t), p, g_p, regularizer=regularizer * np.exp(-regularizer_decay * t))
    A, A_nq = get_A(x, sir_params), get_A(x_nq, sir_params)

  return costs, xs, x_nqs, us


# lambert w function implementation in jax (WO branch)

def _real_lambertw_recursion(w: jax.Array, x: jax.Array) -> jax.Array:
    return w / (1+w) * (1+jnp.log(x / w))

@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _lambertwk0(x, max_steps=5):
    w_0 = jax.lax.select(
        x > jnp.e,
        jnp.log(x) - jnp.log(jnp.log(x)),
        x / jnp.e
    )
    w_0 = jax.lax.select(
        x > 0,
        w_0,
        jnp.e * x / (1 + jnp.e * x + jnp.sqrt(1 + jnp.e * x)) * jnp.log(
            1 + jnp.sqrt(1 + jnp.e * x))
    )

    w, _ = jax.lax.scan(
        lambda carry, _: (_real_lambertw_recursion(carry, x),) * 2,
        w_0,
        xs=None, length=max_steps
    )

    w = jax.lax.select(
        jnp.isclose(x, 0.0),
        0.0,
        w
    )
    return w

@_lambertwk0.defjvp
def _lambertw_jvp(max_steps, primals, tangents):
    x, = primals
    t, = tangents

    y = _lambertwk0(x, max_steps)
    dydx = 1 / (x + jnp.exp(y))

    jvp = jax.lax.select(
        jnp.isclose(x, -1/jnp.e),
        jnp.nan,
        dydx * t
    )
    return y, jvp

@jnp.vectorize
def lambertw(x, k=0, max_steps=5):
    if k != 0:
        raise NotImplementedError()
    return _lambertwk0(x, max_steps=max_steps)


def plot_sol(gpc_outputs, sols, cost_params):
	"""Plot solution and state over time with comparison to no control."""

    xs, x_nqs, us = gpc_outputs
    x, y, t, control, y2, t2 = sols
    c2, c3 = cost_params

    plt.plot([s[0] for s in xs], label=r'$x(1)$ (GPC-simplex)', color='steelblue')
    plt.plot(t, x, label=r'$x(1)$', color='steelblue', linestyle='--')
    plt.plot([s[1] for s in xs], label=r'$x(2)$ (GPC-simplex)', color='orange')
    plt.plot(t, y, label=r'$x(2)$', color='orange', linestyle='--')
    plt.plot(t2, y2, label=r'$x(2)$ (no intervention)', color='red', linestyle='--')
    plt.plot([y_max] * len(xs), label=r'$y_{max}$', color='purple', linestyle='--')
    plt.title('State vs. time, c2=%.2f, c3=%d'%(c2, c3))
    plt.legend(loc='upper right', fontsize='small', markerscale=0.6)
    plt.xlabel('Time Step')
    plt.grid(True)
    plt.show()

    plt.plot([u[1] for u in us], label=r'$u(1)$ (GPC-simplex)', color='green')
    plt.plot(t, control, label=r'$u(1)$', color='green', linestyle='--')
    plt.title('Control vs. time, c2=%.2f, c3=%d'%(c2, c3))
    plt.xlabel('Time Step')
    plt.ylabel(r'$u_t$(2)')
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.show()

# Control algorithm and SIR problem parameter setting and initialization

x_init, p_init = np.array([0.9, 0.01, 0.09]), np.array([0.01, 0.99])
T, H = 100, 5
beta, q, pi = 0.3, 0.1, 0.0 # infection, recovery, loss of immune
params = (H, (beta, q, pi))
y_max, sigma_0 = 0.1, 3.0


def cost(y_max, sigma_0, c2, c3, final=1.0):
  """Cost function, analogous to minimizehospital flow cost used in
  (Section 4.2 in Ketcheson (2021))."""

  return lambda x, u: c2 * (u[0] ** 2) 
  + c3 * (x[1] - y_max) / (1 + jnp.exp(-100 * (x[1] - y_max))) 
  + final * jnp.real(lambertw(-sigma_0 * x[0] * jnp.exp(-sigma_0 * (1 - x[2])))) / sigma_0

def compute_sol(c2, c3, y_max, x_init, T, beta, q):
  """Compute closed-form solution."""

  x, y, sigma, t, newguess, J = solve_pmp(
  	c2=5e-2, c3=c3, ymax=y_max, T=T, guess=None, x0=x_init[0], y0=x_init[1]
  	)
  x, y, sigma, t, newguess, J = solve_pmp(
  	c2=c2, c3=c3, ymax=y_max, T=T, guess=newguess, x0=x_init[0], y0=x_init[1]
  	)
  x2, y2, t2 = SIR_forward(beta=beta, gamma=q, x0=x_init[0], y0=x_init[1], T=T)

  return x, y, sigma, t, x2, y2, t2, J


# Set SIR parameters
c2, c3 = 0.01, 100
cost_fn = cost(y_max, sigma_0, c2, c3)
x, y, sigma, t, x2, y2, t2, J = compute_sol(c2, c3, y_max, x_init, T, beta, q)
sigma_ratios = sigma / sigma_0
w = np.stack([x, y, 1-x-y], axis=1)
v = np.stack([1 - sigma_ratios, sigma_ratios], axis=1)
J = jax.vmap(cost_fn)(w, v)

# Learning parameters

lr, decay, threshold, magitude = 0.5, 1e-3, 0.9, 1.0

costs, xs, x_nqs, us = training(
	x_init, p_init, (y_max, threshold, magnitude), params, cost_fn, T, lr, lr_decay=decay
)

plot_sol((xs, x_nqs, us), (x, y, t, sigma/sigma_0, y2, t2), (c2, c3))
