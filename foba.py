import numpy as np
import jax.numpy as jnp
import time
from scope import FobaSolver, ForwardSolver, OMPSolver
import parallel_experiment_util as util

n, p = 100, 500


def data_generator(s, random_state=0):
    np.random.seed(random_state)
    # beta
    beta_nonzero = np.random.uniform(0, 1, s)
    beta_nonzero /= np.linalg.norm(beta_nonzero)
    beta_nonzero *= 5
    position_nonzero = np.random.choice(p, s, replace=False)
    beta = np.zeros(p)
    beta[position_nonzero] = beta_nonzero
    # sample
    X = np.random.multivariate_normal(beta, np.eye(p), n)
    negative_index = np.random.choice(n, int(n / 2), replace=False)
    X[negative_index] *= -1
    # response
    Xbeta = np.clip(X @ beta, -30, 30)
    y = np.random.binomial(1, 1 / (1 + np.exp(-Xbeta)))
    y = 2 * y - 1
    return X, y, beta, position_nonzero


def logistic_loss(params, X, y):
    Xbeta = jnp.clip(jnp.matmul(X, params), -30, 30)
    return jnp.mean(jnp.log(1 + jnp.exp(-y * Xbeta))) + 0.005 * jnp.square(params).sum()

def logistic_loss_grad(params, X, y):
    Xbeta = jnp.clip(jnp.matmul(X, params), -30, 30)
    return jnp.mean(-(y / (1 + jnp.exp(y * Xbeta)))[:,jnp.newaxis] * X, axis=0) + 0.01 * params

def F_score(support_set1, support_set2):
    set1 = set(support_set1)
    set2 = set(support_set2)
    return 2 * len(set1 & set2) / (len(set1) + len(set2))


def estimation_error(true_params, est_params):
    return np.linalg.norm(true_params - est_params) / np.linalg.norm(true_params)


def task(s, seed):
    X, y, true_params, true_support_set = data_generator(s, seed)
    X = jnp.array(X)
    y = jnp.array(y)

    def objective(params, data_indices):
        return logistic_loss(
            params,
            X[data_indices, :],
            y[data_indices],
        )
    def gradient(params, data_indices):
        return logistic_loss_grad(
            params,
            X[data_indices, :],
            y[data_indices],
        )
    cv_fold_id = [(5*i) // n for i in range(n)]
    results = []
    solvers = {
        #"foba_gdt" : FobaSolver(p, np.arange(20), n, use_gradient=True, cv=5, cv_fold_id=cv_fold_id),
        #"omp" : OmpSolver(p, np.arange(20), n, cv=5, cv_fold_id=cv_fold_id),
        "foba" : FobaSolver(p, np.arange(20), n, cv=5, cv_fold_id=cv_fold_id),
        "forward" : ForwardSolver(p, np.arange(20), n, cv=5, cv_fold_id=cv_fold_id),
    }

    for method, solver in solvers.items():
        result = {"method": method}
        start = time.perf_counter()
        solver.solve(objective, gradient, jit=True)
        result["time"] = time.perf_counter() - start
        result["F_score"] = F_score(true_support_set, solver.support_set)
        result["estimation_error"] = estimation_error(true_params, solver.params)
        result["loss"] = solver.value_of_objective
        result["sparsity"] = solver.support_set.size
        print(result)
        results.append(result)

    return results


if __name__ == "__main__":
    in_key = ["s", "seed"]
    out_key = ["method", "time", "F_score", "loss", "estimation_error", "sparsity"]

    experiment = util.ParallelExperiment(
        task, in_key, out_key, processes=20, name="Foba-obj", memory_limit=0.8
    )

    if True:
        experiment.check(s=5, seed=0)
    else:
        """
        parameters = util.para_generator(
            {"s": np.arange(5, 15)},
            repeat=50,
            seed=1,
        )
        """
        parameters = [
            {"s": s, "seed": 50 * (s-5) + i} for s in range(5, 15) for i in [1]
        ]
        experiment.run(parameters)
        experiment.save()
