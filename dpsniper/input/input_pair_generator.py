from abc import ABC, abstractmethod


class InputPairGenerator(ABC):
    """
    A generator for mechanism input pairs in a neighborhood.
    """

    @abstractmethod
    def get_input_pairs(self):
        """
        Generates input pairs in a neighborhood for a mechanism.

        Returns:
            a generator for tuples of two 1d arrays (number of components depends on input dimension of mechanism)
        """
        pass
