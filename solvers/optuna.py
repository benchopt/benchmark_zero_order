from benchopt import BaseSolver

import numpy as np
import optuna
from optuna import samplers
# Check that cmaes is installed
import cmaes  # noqa: F401


class Solver(BaseSolver):
    """optuna"""

    name = "optuna"

    install_cmd = "conda"
    requirements = ["optuna", "cmaes"]
    parameters = {
        "solver": ["cmaes", "TPE", "RandomSearch"]
    }

    sampling_strategy = 'callback'

    def set_objective(self, function, dimension, bounds):
        self.function = function
        self.dimension = dimension
        self.bounds = bounds

    def run(self, cb):
        def objective(trial):
            x = np.array([
                trial.suggest_float(f'x_{k}', self.bounds[0], self.bounds[1])
                for k in range(self.dimension)
            ])
            return self.function(x)

        class StopCallback:

            def __call__(self, study, trial):
                # Call the callback after each function evaluation, and stop
                # the optimization if the callback returns False.
                if not cb():
                    study.stop()

        # Get a seed that varies across repetitions, datasets and solvers,
        # to ensure a good coverage of the search space, while still being
        # reproducible.
        seed = self.get_seed(
            use_repetition=True, use_dataset=True, use_solver=True
        )
        if self.solver == "TPE":
            sampler = samplers.TPESampler(seed=seed, n_startup_trials=10)
        elif self.solver == "RandomSearch":
            sampler = samplers.RandomSampler(seed=seed)
        elif self.solver == "cmaes":
            sampler = samplers.CmaEsSampler(seed=seed)
        else:
            raise NotImplementedError(f"Solver {self.solver} not implemented")
        self.study_ = optuna.create_study(
            sampler=sampler, direction='minimize'
        )
        optuna.logging.disable_default_handler()  # limit verbosity
        self.study_.optimize(
            objective, n_trials=1000, callbacks=[StopCallback()]
        )
        self.xopt = self.study_.best_trial.params

    def get_result(self):
        best_param = self.study_.best_trial.params
        xopt = np.array([best_param[f'x_{k}'] for k in range(self.dimension)])
        return dict(x=xopt)
