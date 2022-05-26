from abc import ABC, abstractmethod

import numpy as np
import struct


class FeatureTransformer(ABC):
    """
    A features transformer preparing features for a classifier.
    """

    @abstractmethod
    def transform(self, x):
        """
        Transforms a given feature to a different representation.

        Args:
            x: ndarray of shape (n_samples, original_dimensions) containing original features

        Returns:
            ndarray of shape (n_samples, new_dimensions) containing new features
        """
        pass


class BitPatternFeatureTransformer(FeatureTransformer):
    """
    Use the bit-representation of a 1d 64-bit floating point number as feature.
    """

    def transform(self, x):
        assert(x.shape[1] == 1)         # must be 1-d features
        assert(x.dtype == np.float64)   # must be 64 bit floats (double precision)
        # Use numpy arrays as structs
        x_int = x.view( dtype=np.uint64 ).reshape(-1)
        # Use format to convert to binary string
        int_to_bin = lambda x: [ c == '1' for c in '{:0>64b}'.format(x) ]
        x_bin = np.array( list( map( int_to_bin, x_int ) ) )
        x_bin = x_bin.reshape( ( x.shape[0], 64 ) )
        return x_bin


class FlagsFeatureTransformer(FeatureTransformer):
    """
    Encodes special values as boolean flags in additional dimensions.

    For example, for special value -1, it transforms
    [[-3, -1, 4], [-3, 2, -3]] to [[-3, 0, 4, 0, 1, 0], [-3, 2, -3, 0, 0, 0]]
    """

    def __init__(self, special_values: list):
        """
        Args:
            special_values: The list of special values to be encoded.
                            A separate flag dimension is introduced for each of these values.
        """
        self.special_values = special_values

    def transform(self, x):
        input_dim = x.shape[1]
        y = np.zeros((x.shape[0], input_dim*(len(self.special_values) + 1)))
        y[:, :input_dim] = x
        for i in range(0, len(self.special_values)):
            val = self.special_values[i]
            where = x == val
            y[:, :input_dim][where] = 0
            y[:, input_dim*(i+1):input_dim*(i+2)] = where
        return y
