from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
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

    def set_objective(self, function, dimension):
        self.function = function
        self.dimension = dimension

    def run(self, n_iter):
        f = self.function
        x0 = np.ones(self.dimension) / 2.0
        if n_iter == 0:
            self.xopt = x0
            return
        result = minimize(
            f,
            x0=x0,
            method=self.solver,
            options={"maxiter": n_iter, "xatol": 1e-20, "fatol": 1e-20},
        )
        self.xopt = result.x

    def get_result(self):
        return self.xopt.flatten()
