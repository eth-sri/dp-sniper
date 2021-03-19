from dpsniper.classifiers.stable_classifier import StableClassifier
from dpsniper.classifiers.multi_layer_perceptron import MultiLayerPerceptron
from dpsniper.classifiers.logistic_regression import LogisticRegression


class ClassifierFactory:
    """
    A factory constructing classifiers.
    """

    def __init__(self, clazz: type, **args):
        """
        Construct a factory which creates instances of a fixed type using fixed arguments.

        Args:
            clazz: the type of the classifier
            **args: the arguments to be passed when constructing the classifier
        """
        self.clazz = clazz
        self.args = args

    def create(self) -> StableClassifier:
        """
        Returns:
            a new StableClassifier
        """
        return self.clazz(**self.args)


class LogisticRegressionFactory(ClassifierFactory):
    """
    A factory constructing logistic regression classifiers.
    """

    def __init__(self, **args):
        """
        Args:
            **args: arguments to be passed when constructing the classifier (see LogisticRegression.__init__)
        """
        super().__init__(LogisticRegression, **args)


class MultiLayerPerceptronFactory(ClassifierFactory):
    """
    A factory constructing feedforward neural network classifiers.
    """

    def __init__(self, **args):
        """
        Args:
            **args: arguments to be passed when constructing the classifier (see MultilayerPerceptron.__init__)
        """
        super().__init__(MultiLayerPerceptron, **args)
