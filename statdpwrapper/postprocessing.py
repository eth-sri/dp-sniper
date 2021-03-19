from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from statdp.algorithms import _hamming_distance
from dpsniper.utils.my_logging import log
from dpsniper.utils.zero import ZeroNoisePrng


class Postprocessing(ABC):
    """
    Base class for StatDP postprocessing.
    """

    def __init__(self):
        # whether this postprocessing needs to be passed the noisefree reference value
        self.requires_noisefree_reference = False

    @abstractmethod
    def process(self, b: List, noisefree_reference=None) -> Tuple:
        """
        Performs postprocessing on the output b.

        Args:
            b: list of (potentially mixed) algorithm outputs
            noisefree_reference: w0 reference output when running algorithm without noise

        Returns:
            tuple containing the postprocessed outputs
        """
        pass

    @abstractmethod
    def n_output_dimensions(self):
        """
        Returns the number of output dimensions.
        """
        pass


class HammingDistancePP(Postprocessing):
    """
    Hamming distance between output and noisefree reference output.
    """

    def __init__(self):
        super().__init__()
        self.requires_noisefree_reference = True

    def process(self, b, noisefree_reference=None):
        if noisefree_reference is None:
            raise ValueError("process(...) of HammingDistancePP requires noisefree_reference")
        return _hamming_distance(b, noisefree_reference)

    def n_output_dimensions(self):
        return 1

    def __str__(self):
        return "HammingDistancePP"


class CountPP(Postprocessing):
    """
    Count the number of occurrences of a specific value in the output.
    """

    def __init__(self, val):
        super().__init__()
        self.val = val

    def process(self, b, noisefree_reference=None):
        return b.count(self.val)

    def n_output_dimensions(self):
        return 1

    def __str__(self):
        return "CountPP({})".format(self.val)


class LengthPP(Postprocessing):
    """
    The length of the output list.
    """

    def __init__(self):
        super().__init__()

    def process(self, b, noisefree_reference=None):
        return len(b)

    def n_output_dimensions(self):
        return 1

    def __str__(self):
        return "LengthPP"


class EntryPP(Postprocessing):
    """
    The entry at a specific position in the output list.
    """

    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def process(self, b, noisefree_reference=None):
        if self.index < len(b):
            x = b[self.index]
            if x is None:
                return 0    # StatDP input must be numerical
            return float(x)
        # out of bounds may happen if output has variable dimensions and is shorter than max_dimensions
        return 0

    def n_output_dimensions(self):
        return 1

    def __str__(self):
        return "EntryPP({})".format(self.index)


class AvgPP(Postprocessing):
    """
    The average of the output list.
    """

    def __init__(self):
        super().__init__()

    def process(self, b, noisefree_reference=None):
        cnt = 0
        sum = 0
        for i in b:
            if (isinstance(i, int) or isinstance(i, float)) and type(i) != bool:
                # only incorporate numerical values
                cnt += 1
                sum += i
            elif type(i) != bool and i is not None:
                log.warn("Unknown type %s", str(type(i)))
        if cnt > 0:
            return sum / cnt
        return 0

    def n_output_dimensions(self):
        return 1

    def __str__(self):
        return "AvgPP"


class MinPP(Postprocessing):
    """
    The minimum value in the output list.
    """

    def __init__(self):
        super().__init__()

    def process(self, b, noisefree_reference=None):
        min = float("infinity")
        for i in b:
            if (isinstance(i, int) or isinstance(i, float)) and type(i) != bool:
                # only incorporate numerical values
                if i < min:
                    min = i
            elif type(i) != bool and i is not None:
                log.warn("Unknown type %s", str(type(i)))
        return min

    def n_output_dimensions(self):
        return 1

    def __str__(self):
        return "MinPP"


class MaxPP(Postprocessing):
    """
    The maximum value in the output list.
    """

    def __init__(self):
        super().__init__()

    def process(self, b, noisefree_reference=None):
        max = float("-infinity")
        for i in b:
            if (isinstance(i, int) or isinstance(i, float)) and type(i) != bool:
                # only incorporate numerical values
                if i > max:
                    max = i
            elif type(i) != bool and i is not None:
                log.warn("Unknown type %s", str(type(i)))
        return max

    def n_output_dimensions(self):
        return 1

    def __str__(self):
        return "MaxPP"


class CombinedPP(Postprocessing):
    """
    Combining the results of a numerical and a categorical postprocessing to a tuple.
    """

    def __init__(self, categorical_pp: Postprocessing, numerical_pp: Postprocessing):
        super().__init__()
        self.categorical_pp = categorical_pp
        self.numerical_pp = numerical_pp
        self.requires_noisefree_reference = categorical_pp.requires_noisefree_reference or \
                                            numerical_pp.requires_noisefree_reference

    def process(self, b, noisefree_reference=None):
        return self.categorical_pp.process(b, noisefree_reference), self.numerical_pp.process(b, noisefree_reference)

    def n_output_dimensions(self):
        return self.categorical_pp.n_output_dimensions() + self.numerical_pp.n_output_dimensions()

    def __str__(self):
        return "CombinedPP({}, {})".format(str(self.categorical_pp), str(self.numerical_pp))


class IdentityPP(Postprocessing):
    """
    Postprocessing that does nothing except padding to the given maximum length
    """

    def __init__(self, max_length: int):
        super().__init__()
        self.max_length = max_length

    def process(self, b, noisefree_reference=None):
        for i in range(len(b), self.max_length):
            b.append(-1)
        return tuple(b)

    def n_output_dimensions(self):
        return self.max_length

    def __str__(self):
        return "IdentityPP"


class PostprocessingConfig:
    """
    Configuration object specifying the applicable postprocessings.
    """

    def __init__(self,
                 is_numerical: bool,
                 is_categorical: bool,
                 has_variable_dimensions: bool,
                 max_dimensions: int,
                 categories=[],
                 disable_pp=False):
        self.is_numerical = is_numerical
        self.is_categorical = is_categorical
        self.has_variable_dimensions = has_variable_dimensions
        self.max_dimensions = max_dimensions
        self.categories = categories
        self.disable_pp = disable_pp


class PostprocessedAlgorithm:
    """
    Wrapper for an algorithm with postprocessing.
    """

    def __init__(self, algorithm, postprocessing: Postprocessing):
        self.algorithm = algorithm
        self.postprocessing = postprocessing
        self.__name__ = str(postprocessing)

    def __str__(self):
        return "{}_{}".format(str(self.algorithm), str(self.postprocessing))


the_zero_noise_prng = ZeroNoisePrng()


def get_postprocessing_search_space(pp_config: PostprocessingConfig):
    """
    Return the list of applicable postprocessings for a given configuration.
    """
    if pp_config.disable_pp:
        # disable postprocessing by using the identity postprocessing
        return [IdentityPP(pp_config.max_dimensions)]

    if pp_config.max_dimensions == 1:
        # postprocessing irrelevant if only one dimension, simply return the value itself
        return [EntryPP(0)]

    pps_numerical = []
    if pp_config.is_numerical:
        for i in range(0, pp_config.max_dimensions):
            pps_numerical.append(EntryPP(i))
        pps_numerical.append(AvgPP())
        pps_numerical.append(MinPP())
        pps_numerical.append(MaxPP())

    pps_categorical = []
    if pp_config.is_categorical:
        for c in pp_config.categories:
            pps_categorical.append(CountPP(c))
        if not pp_config.is_numerical:
            # hamming distance not defined for mixed outputs because length of categorical parts may differ
            pps_categorical.append(HammingDistancePP())

    pps_combined = []
    if pp_config.is_numerical and pp_config.is_categorical:
        # also include all combinations
        for ppnum in pps_numerical:
            for ppcat in pps_categorical:
                pps_combined.append(CombinedPP(ppcat, ppnum))
        if pp_config.has_variable_dimensions:
            for ppnum in pps_numerical:
                pps_combined.append(CombinedPP(LengthPP(), ppnum))
            for ppcat in pps_categorical:
                pps_combined.append(CombinedPP(ppcat, LengthPP()))

    all_pps = pps_categorical + pps_numerical + pps_combined
    if pp_config.has_variable_dimensions:
        all_pps.append(LengthPP())
    return all_pps


def get_postprocessed_algorithms(algorithm, pp_config: PostprocessingConfig):
    """
    Construct all postprocessed algorithms for a given algorithm and configuration.
    """
    algs = []
    pps = get_postprocessing_search_space(pp_config)
    for pp in pps:
        algs.append(PostprocessedAlgorithm(algorithm, pp))
    return algs


def compose_postprocessing(prng, queries, epsilon, alg, **kwargs):
    """
    The composed algorithm with postprocessing actually tested by StatP.
    """
    algorithm = alg.algorithm
    postprocessing = alg.postprocessing
    new_kwargs = kwargs.copy()
    del new_kwargs['_d1']
    b = algorithm(prng, queries, epsilon, **new_kwargs)
    if postprocessing.requires_noisefree_reference:
        a0 = kwargs['_d1']  # get reference input
        w0 = algorithm(the_zero_noise_prng, a0, epsilon, **new_kwargs)
        return postprocessing.process(b, noisefree_reference=w0)
    return postprocessing.process(b)
