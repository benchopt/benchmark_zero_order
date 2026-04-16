from benchopt import BaseSolver

import numpy as np
from scipy.optimize import minimize


class Solver(BaseSolver):
    """Scipy optimizers."""

    name = "scipy"

    install_cmd = "conda"
    requirements = ["numpy", "scipy"]
    parameters = {
        "solver": ["Nelder-Mead", "Powell", "BFGS"],
    }

    sampling_strategy = 'callback'

    def set_objective(self, function, dimension, bounds):
        self.function = function
        self.dimension = dimension
        self.bounds = bounds

    def run(self, cb):
        f = self.function

        seed = self.get_seed(
            use_repetition=True, use_dataset=True, use_solver=True
        )
        rng = np.random.RandomState(seed)  # fix seed
        x0 = rng.uniform(size=self.dimension,
                         low=self.bounds[0],
                         high=self.bounds[1])
        self.xopt = x0
        best_val = np.inf

        class _StopScipy(Exception):
            pass

        def objective(x):
            nonlocal best_val
            value = f(x)
            if value < best_val:
                best_val = value
                self.xopt = np.asarray(x).flatten()
            return value

        def scipy_callback(xk):
            if not cb():
                raise _StopScipy()

        options = {}
        if self.solver in ("Nelder-Mead", "Powell"):
            options.update({"xatol": 1e-20, "fatol": 1e-20})

        try:
            minimize(
                objective,
                x0=x0,
                method=self.solver,
                callback=scipy_callback,
                options=options,
            )
        except _StopScipy:
            pass

    def get_result(self):
        return dict(x=self.xopt)
