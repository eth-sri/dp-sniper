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

        # unfortunately, cannot vectorize this
        res = np.zeros((x.shape[0], 64), dtype=np.bool)
        for row_idx in range(0, x.shape[0]):
            v = x[row_idx, 0]

            # extract floating point bit pattern according to
            # https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
            packed = struct.pack("!d", v)
            str_bytes = [bin(c) for c in packed]

            byte_offset = 1
            for byte in str_bytes:
                for bit_idx in range(0, len(byte)):
                    bit = byte[len(byte) - bit_idx - 1]
                    if bit == 'b':
                        break
                    res[row_idx, 8*byte_offset - bit_idx - 1] = bit
                    bit_idx += 1
                byte_offset += 1
        return res


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
