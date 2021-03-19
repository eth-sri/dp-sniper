from typing import List

from dpsniper.mechanisms.abstract import Mechanism
from dpsniper.mechanisms.laplace import LaplaceMechanism
from dpsniper.mechanisms.sparse_vector_technique import *

import numpy as np


class ParallelMechanism(Mechanism):
    """
    A wrapper class running multiple mechanisms in parallel and returning all outputs.
    """

    def __init__(self, mechanisms: List[Mechanism]):
        self.mechanisms = mechanisms

    def m(self, a, n_samples: int =1):
        b_tup = ()
        for mech in self.mechanisms:
            b = mech.m(a, n_samples)
            if len(b.shape) == 1:
                b = np.atleast_2d(b).T
            b_tup = b_tup + (b, )
        all_b = np.column_stack(b_tup)
        return all_b

    def get_n_parallel(self):
        return len(self.mechanisms)


class LaplaceParallel(ParallelMechanism):
    """
    Running multiple instances of LaplaceMechanism in parallel.
    """

    def __init__(self, n_parallel: int, eps: float = 0.1):
        mechanisms = []
        for i in range(0, n_parallel):
            mechanisms.append(LaplaceMechanism(eps=eps))
        super().__init__(mechanisms)


class SVT34Parallel(ParallelMechanism):
    """
    Running SparseVectorTechnique3 and SparseVectorTechnique4 in parallel.
    """

    def __init__(self, eps=0.1, c=2, t=1):
        mechanisms = [
            SparseVectorTechnique3(eps=eps, c=c, t=t),
            SparseVectorTechnique4(eps=eps, c=c, t=t)
        ]
        super().__init__(mechanisms)
