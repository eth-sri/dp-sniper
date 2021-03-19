from abc import ABC, abstractmethod
from typing import Optional

from dpsniper.classifiers.feature_transformer import FeatureTransformer
from dpsniper.utils.my_logging import log

from sklearn import preprocessing


class StableClassifier(ABC):
    """
    A classifier for two classes 0 and 1. The classifier is stable w.r.t. numerical rounding noise
    and provides functionality for feature transformation and normalization.
    """

    def __init__(self, feature_transform: Optional[FeatureTransformer] = None, normalize_input: bool = False):
        """
        Creates an abstract stable classifier.

        Args:
            feature_transform: an optional feature transformer to be applied to the input features
            normalize_input: whether to perform normalization of input features (after feature transformation)
        """
        self.feature_transform = feature_transform

        self.normalize_input = normalize_input
        self.normalizer = None  # remember data normalizer, needs to be re-used at prediction time

    def _transform(self, x):
        """
        Perform feature transformation.
        """
        if self.feature_transform is not None:
            log.debug("Transforming features...")
            x = self.feature_transform.transform(x)
            log.debug("Done transforming features")
        return x

    def train(self, training_batch_generator):
        """
        Trains the classifier.

        Args:
            training_batch_generator: generator for batches containing training data.

        Note:
            Each batch returned by the generator must be a tuple (x, y) where
            x: nd array of shape (n_samples, feature_dimensions) containing features;
            y: 1d array of shape (n_samples, ) containing labels in {0, 1}
        """
        def generate_transformed_normalized():
            first = True
            for (x, y) in training_batch_generator:
                x = self._transform(x)
                if self.normalize_input:
                    if first:
                        # fit normalizer on first batch
                        self.normalizer = preprocessing.StandardScaler().fit(x)
                        first = False
                    # normalize the data
                    x = self.normalizer.transform(x)
                yield x, y

        self._train(generate_transformed_normalized())

    def get_normalizer_str(self):
        if self.normalize_input:
            return f'StandardScaler(scale={self.normalizer.scale_}, mean={self.normalizer.mean_})'
        else:
            return None

    def predict_probabilities(self, x):
        """
        Computes the probabilities p(y = 0 | x) for a vector x based on the trained classifier.

        Args:
            x: nd array of shape (n_samples, feature_dimensions)

        Returns:
            1d array of shape (n_samples, )
        """
        x = self._transform(x)
        if self.normalize_input:
            x = self.normalizer.transform(x)
        return self._predict_probabilities(x)

    @abstractmethod
    def _train(self, training_batch_generator):
        """
        Trains the classifier on feature-transformed and normalized data.

        Args:
            training_batch_generator: generator for batches containing training data.

        Note:
            Each batch returned by the generator must be a tuple (x, y) where
            x: nd array of shape (n_samples, feature_dimensions) containing features;
            y: 1d array of shape (n_samples, ) containing labels in {0, 1}
        """
        pass

    @abstractmethod
    def _predict_probabilities(self, x):
        """
        Computes the probabilities p(y = 0 | x) for a (feature-transformed and normalized)
        vector x based on the trained classifier.

        Args:
            x: nd array of shape (n_samples, feature_dimensions)

        Returns:
            1d array of shape (n_samples, )
        """
        pass
