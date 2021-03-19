from enum import Enum
from typing import List


class InputBaseType(Enum):
    FLOAT = 1   # floating point
    INT = 2     # integer
    BOOL = 3    # integer values 0 and 1


class InputDomain:
    """
    A representation of the input domain for mechanisms.
    """

    def __init__(self, dimensions: int, base_type: InputBaseType, range: List[int]):
        """
        Creates an input domain.

        Args:
            dimensions: the dimensionality of the input
            base_type: the InputBaseType to be used for all dimensions
            range: list with two entries [min, max] representing the range for all dimensions
        """
        self.dimensions = dimensions
        self.base_type = base_type

        # ensure correct format and data type of range
        assert(len(range) == 2)
        if self.base_type == InputBaseType.FLOAT:
            self.range = [float(range[0]), float(range[1])]
        elif self.base_type == InputBaseType.INT:
            self.range = [int(range[0]), int(range[1])]
        elif self.base_type == InputBaseType.BOOL:
            self.range = [0, 1]
