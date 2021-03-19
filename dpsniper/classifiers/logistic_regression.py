from dpsniper.classifiers.multi_layer_perceptron import MultiLayerPerceptron


class LogisticRegression(MultiLayerPerceptron):
    """
    A logistic regression classifier.
    """

    def __init__(self, in_dimensions: int, optimizer_factory: 'TorchOptimizerFactory', **args):
        """
        Creates a logistic regression classifier.

        Args:
            in_dimensions: number of input dimensions for the classifier (dimensionality of features)
            optimizer_factory: a factory constructing the optimizer to be used for training
            **args: any additional arguments for the classifier (see MultiLayerPerceptron.__init__)
        """
        super().__init__(
            in_dimensions=in_dimensions,
            optimizer_factory=optimizer_factory,
            hidden_sizes=(),
            **args
        )
