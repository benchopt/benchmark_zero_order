from benchopt import BaseObjective

import numpy as np


class Objective(BaseObjective):
    min_benchopt_version = "1.9"
    name = "Zero-order test functions"

    def get_one_result(self):
        # Return one result for testing purpose.
        return dict(x=np.zeros(self.dimension))

    def set_data(self, function, dimension, bounds):
        self.function = function
        self.dimension = dimension
        self.bounds = bounds

    def evaluate_result(self, x):
        return self.function(x)

    def get_objective(self):
        return dict(function=self.function,
                    dimension=self.dimension,
                    bounds=self.bounds)
