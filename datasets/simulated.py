from benchopt import BaseDataset

import numpy as np


class Dataset(BaseDataset):

    name = "simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "dimension": [2, 10],
    }

    def get_data(self):
        return dict(
            function=lambda x: np.linalg.norm(x, 2) ** 2,
            dimension=self.dimension, bounds=(-3, 3)
        )
