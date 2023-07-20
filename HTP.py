import numpy as np
from jax import numpy as jnp
from skscope import IHTSolver, HTPSolver

p = 1000
n = 200
tolSuccess  = 1e-4; # Tolerance on the relative error for a recovered vector to be consider a success


def data_generator(s, type: str, seed):
    np.random.seed(seed)

    beta = np.zeros(p)
    position_nonzero = np.random.choice(p, s, replace=False)
    if type == "flat":
        beta[position_nonzero] = 1.0
    elif type == "gaussian":
        beta[position_nonzero] = np.random.randn(s)
    elif type == "linear":
        beta[position_nonzero] = np.arange(1, s + 1, dtype=float) / s

    X = np.random.randn(n, p) / np.sqrt(n)
    y = np.matmul(X, beta)

    return beta, X, y


def task(s, type: str, seed):
    true_params, X, y = data_generator(s, type, seed)
    results = []

    def linear_loss(params):
        return jnp.sum(jnp.square(y - jnp.matmul(X, params)))

    solvers = {
        "IHT" : IHTSolver(p,s, step_size=1.0),
        "HTP" : HTPSolver(p,s, step_size=1.0),
        "HTP'" : HTPSolver(p,2 * s, step_size=1.0),
    }

    for name, solver in solvers.items():
        solver.solve(linear_loss, jit=True)
        results.append(
            {
                "method": name,
                "success": int(np.linalg.norm(true_params - solver.params) / np.linalg.norm(true_params) < tolSuccess),
            }
        )

    return results

if __name__ == "__main__":  
    # {"s": np.arange(1, 100), "type": ["flat", "gaussian", "linear"]}, repeat 100 times
    print(task(s=10, type="flat", seed=46455))
 