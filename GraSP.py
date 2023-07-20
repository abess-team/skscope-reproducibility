import numpy as np
import jax.numpy as jnp
import math
import time

from skscope import GraspSolver, OMPSolver

def data_generator(n, p, s, rho, random_state=None):
    np.random.seed(random_state)
    # beta
    beta = np.zeros(p)
    true_support_set = np.random.choice(p, s, replace=False)
    beta[true_support_set] = np.random.normal(0, 1, s)
    intercept = np.random.normal(0, 1)
    # X
    X = np.empty((n, p))
    X[:, 0] = np.random.normal(0, 1, n)
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + np.sqrt(1-rho**2) * np.random.normal(0, 1, n)
    # y
    xbeta = np.clip(X @ beta + intercept, -30, 30)
    p = 1 / (1 + np.exp(-xbeta))
    y = np.random.binomial(1, p)

    return X, y, np.append(beta, intercept), true_support_set

def logistic_loss(params, X, y):
    xbeta = jnp.clip(X @ params[:-1] + params[-1], -30, 30)
    return jnp.mean(jnp.log(1 + jnp.exp(xbeta)) - y * xbeta)

def task(n, rho, seed):
    p = 1000
    s = 10
    lambd = math.sqrt(math.log(p) / n)  / 10
    X, y, true_params, true_support_set = data_generator(n, p, s, rho, seed)
    def loss(params):
        return logistic_loss(params, X, y)
    def loss_L2(params):
        return logistic_loss(params, X, y) + lambd * jnp.square(params[:-1]).sum()

    results = [{"method": "True", "time": 0.0, "accuracy": 1.0, "loss": loss(true_params), "error": 0.0}]
    methods = ["OMP", "GraSP_L2", "GraSP"]
    solvers = [OMPSolver(p+1,s,preselect=[p]), GraspSolver(p+1,s,preselect=[p]), GraspSolver(p+1,s,preselect=[p])]
    loss_fns = [loss, loss_L2, loss]

    for solver, loss_fn, method in zip(solvers, loss_fns, methods):
        result = {"method": method}
        start = time.perf_counter()
        solver.solve(loss_fn, jit=True)
        result["time"] = time.perf_counter() - start
        result["accuracy"] = len(set(solver.get_support()) & set(true_support_set)) / s
        result["loss"] = logistic_loss(solver.get_estimated_params(), X, y).item()
        result["error"] = np.linalg.norm(solver.get_estimated_params()[:-1] - true_params[:-1]) / np.linalg.norm(true_params[:-1])
        results.append(result)

    return results

if __name__ == "__main__":
    # {"n": np.arange(50, 1001, 50), "rho": [0.0, 1/3, 0.5, math.sqrt(2)/2]}, repeat 200 times
    print(task(n=1000, rho=0.5, seed=0))

              

