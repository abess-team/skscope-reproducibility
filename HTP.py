import numpy as np
from scope import IHTSolver, quadratic_objective, GrahtpSolver, OmpSolver
import parallel_experiment_util as util

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

    model: dict = quadratic_objective(np.matmul(X.T, X), -np.matmul(X.T, y))

    return model, beta


def task(s, type: str, seed):
    model, true_params = data_generator(s, type, seed)
    results = []

    solvers = {
        #"IHT-1/3" : IHTSolver(p,s, step_size=1/3), 
        #"IHT" : IHTSolver(p,s, step_size=1.0),
        "HTP" : GrahtpSolver(p,s, step_size=1.0),
        #"HTP-1.6" : GrahtpSolver(p,s, step_size=1.6),
        #"HTP-0.71": GrahtpSolver(p,s, step_size=0.71),
        "HTP'" : GrahtpSolver(p,2 * s, step_size=1.0),
    }

    for name, solver in solvers.items():
        solver.solve(**model)
        results.append(
            {
                "method": name,
                "success": int(np.linalg.norm(true_params - solver.params) / np.linalg.norm(true_params) < tolSuccess),
            }
        )

    return results

if __name__ == "__main__":  
    in_key = ["s", "type", "seed"]
    out_key = ["method", "success"]

    experiment = util.ParallelExperiment(
        task, in_key, out_key, processes=20, name="HTP", memory_limit=0.8
    )
    if False:
        experiment.check(s=10, type="flat", seed=46455)
    else:
        parameters = util.para_generator(
            {"s": np.arange(1, 100), "type": ["flat", "gaussian", "linear"]},
            #{"s": np.arange(1, 66), "type": ["flat"]},
            repeat=100,
            seed=0,
        )
        experiment.run(parameters)
        experiment.save()