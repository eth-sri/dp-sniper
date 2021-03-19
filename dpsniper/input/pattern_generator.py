# Modification of StatDP pattern implementation (statdp.generators.generate_database) by Yuxing Wang
#   MIT License, Copyright (c) 2018-2019 Yuxin Wang
#   https://github.com/cmla-psu/statdp

import numpy as np

from dpsniper.input.input_domain import InputDomain, InputBaseType
from dpsniper.input.input_pair_generator import InputPairGenerator


class PatternGenerator(InputPairGenerator):
    """
    Generates the patterns from StatDP [1].

    [1] Ding, Zeyu, Yuxin Wang, Guanhong Wang, Danfeng Zhang, and Daniel Kifer.
        "Detecting Violations of Differential Privacy." In Proceedings of the 2018
        ACM SIGSAC Conference on Computer and Communications Security  - CCS ’18.
        https://doi.org/10.1145/3243734.3243818.

    This is a modification of statdp.generators.generate_database with the following changes:
        - Clips to output range
        - Converts to float if domain is of base type FLOAT
    """

    def __init__(self, domain: InputDomain, component_wise_neighborhood: bool):
        """
        Creates a new pattern generator.

        Args:
            domain: the domain to sample from
            component_wise_neighborhood: whether to use component-wise difference for neighborhood (true),
                                         or single-entry difference (false)
        """
        self.domain = domain
        self.component_wise_neighborhood = component_wise_neighborhood

    def get_input_pairs(self):
        """
        Generates pattern samples in a round-robin fashion.
        """
        for (a1, a2) in self._get_raw_inputs():
            a1 = self._clip(a1)
            a2 = self._clip(a2)
            if self.domain.base_type == InputBaseType.FLOAT:
                a1 = a1.astype(float)
                a2 = a2.astype(float)
            yield a1, a2
            yield a2, a1    # also use swapped

    def _get_raw_inputs(self):
        a1 = np.array([1] * self.domain.dimensions)

        # one above
        a1 = a1.copy()
        a2 = np.array([0]+[1]*(self.domain.dimensions-1))
        yield a1, a2

        # one below
        a1 = a1.copy()
        a2 = np.array([2] + [1] * (self.domain.dimensions - 1))
        yield a1, a2

        if self.component_wise_neighborhood:
            # one above rest below
            a1 = a1.copy()
            a2 = np.array([2]+[0]*(self.domain.dimensions-1))
            yield a1, a2

            # one below rest above
            a1 = a1.copy()
            a2 = np.array([0]+[2]*(self.domain.dimensions-1))
            yield a1, a2

            # half half
            a1 = a1.copy()
            a2 = np.array([2]*int(self.domain.dimensions / 2)
                          + [0]*(self.domain.dimensions - int(self.domain.dimensions / 2)))
            yield a1, a2

            # all above
            a1 = a1.copy()
            a2 = np.array([2]*self.domain.dimensions)
            yield a1, a2

            # all below
            a1 = a1.copy()
            a2 = np.array([0]*self.domain.dimensions)
            yield a1, a2

            # X shape
            a1 = np.array([1] * int(self.domain.dimensions / 2)
                          + [0] * (self.domain.dimensions - int(self.domain.dimensions / 2)))
            a2 = np.array([0] * int(self.domain.dimensions / 2)
                          + [1] * (self.domain.dimensions - int(self.domain.dimensions / 2)))
            yield a1, a2

    def _clip(self, x):
        return np.maximum(self.domain.range[0], np.minimum(self.domain.range[1], x))
