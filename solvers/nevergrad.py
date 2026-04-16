from benchopt import BaseSolver

import numpy as np
import nevergrad as ng


class Solver(BaseSolver):
    """nevergrad"""

    name = "nevergrad"

    install_cmd = "conda"
    requirements = ["pip::nevergrad"]
    parameters = {
        "solver": ["NGOpt", "RandomSearch", "ScrHammersleySearch",
                   "TwoPointsDE", "CMA", "PSO"],
    }

    sampling_strategy = 'callback'

    def set_objective(self, function, dimension, bounds):
        self.function = function
        self.dimension = dimension
        self.bounds = bounds

    def run(self, cb):
        f = self.function
        self.xopt = None

        # Get a seed that varies across repetitions, datasets and solvers,
        # to ensure a good coverage of the search space, while still being
        # reproducible.
        seed = self.get_seed(
            use_repetition=True, use_dataset=True, use_solver=True
        )
        rng = np.random.RandomState(seed)

        parametrization = ng.p.Array(shape=(self.dimension,))
        parametrization.set_bounds(self.bounds[0], self.bounds[1])
        parametrization.random_state = rng  # fix seed
        optimizer = ng.optimizers.registry[self.solver](
            budget=1000, parametrization=parametrization, num_workers=1
        )

        def stop_criterion(optimizer):
            if optimizer.num_tell == 0:
                return False

            recommendation = optimizer.provide_recommendation()
            if recommendation is not None:
                self.xopt = np.asarray(recommendation.value).flatten()
            return not cb()

        optimizer.register_callback("ask", ng.callbacks.EarlyStopping(
            stop_criterion
        ))

        recommendation = optimizer.minimize(f)
        self.xopt = np.asarray(recommendation.value).flatten()

    def get_result(self):
        return dict(x=self.xopt)
