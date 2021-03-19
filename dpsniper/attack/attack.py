from abc import ABC, abstractmethod


class Attack(ABC):
    """
    A probabilistic set containment check for attacking differential privacy.
    """

    @abstractmethod
    def check(self, b):
        """
        Computes the probabilities whether given vectorized outputs b lie in the attack set

        Args:
            b:  1d array of shape (n_samples,) if mechanism output is 1-dimensional;
                nd array of shape (n_samples, d) if mechanism output is d-dimensional

        Returns:
            float 1d array of shape (n_samples,) containing probabilities in [0.0, 1.0]
        """
        pass
