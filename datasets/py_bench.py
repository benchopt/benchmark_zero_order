from benchopt import BaseDataset

from PyBenchFCN import SingleObjectiveProblem as SOP


class Dataset(BaseDataset):

    name = "FCN"

    install_cmd = "conda"
    requirements = ["pip::PyBenchFCN"]

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "function": ["ackley", "rastrigin", "rosenbrock"],
        "dimension": [2, 10],
    }

    def __init__(self, function="rosenbrock", dimension=2):
        self.function = function
        self.dimension = dimension

    def get_data(self):
        # Bounds are taken from:
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        # Or
        # https://opytimark.readthedocs.io/en/latest/api/opytimark.markers.n_dimensional.html
        if self.function == "ackley":
            problem = SOP.ackleyfcn(self.dimension)  # Ackley problem
            bounds = [-32, 32]
        elif self.function == "rosenbrock":
            problem = SOP.rosenbrockfcn(self.dimension)  # Rosenbrock problem
            bounds = [-30, 30]
        elif self.function == "rastrigin":
            problem = SOP.rastriginfcn(self.dimension)  # Rastrigin problem
            bounds = [-5.12, 5.12]
        elif self.function == "schwefel":
            problem = SOP.schwefel220fcn(self.dimension)  # Schwefel problem
            bounds = [-100, 100]
        else:
            raise NotImplementedError(
                f"Function {self.function} not implemented"
            )
        return dict(function=problem.f, dimension=problem.n_var,
                    bounds=bounds)
