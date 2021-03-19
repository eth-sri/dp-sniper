import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tempfile import NamedTemporaryFile
from datetime import datetime
import random
import string
import numpy as np

from dpsniper.classifiers.feature_transformer import FeatureTransformer
from dpsniper.classifiers.stable_classifier import StableClassifier
from dpsniper.classifiers.torch_optimizer_factory import TorchOptimizerFactory
from dpsniper.utils.my_logging import log
from dpsniper.utils.paths import get_output_directory


class MultiLayerPerceptron(StableClassifier):
    """
    A feedforward neural network classifier.
    """

    def __init__(self,
                 in_dimensions: int,
                 optimizer_factory: TorchOptimizerFactory,
                 feature_transform: Optional[FeatureTransformer] = None,
                 normalize_input: bool = True,
                 n_test_batches: int = 0,
                 hidden_sizes: Tuple = (10, 5),
                 epochs: int = 10,
                 regularization_weight: float = 0.001,
                 label: Optional[str] = None):
        """
        Creates a feedforward neural network classifier.

        Args:
            in_dimensions: number of input dimensions for the classifier (dimensionality of features)
            optimizer_factory: a factory constructing the optimizer to be used for training
            feature_transform: an optional feature transformer to be applied to the input features
            normalize_input: whether to perform normalization of input features (after feature transformation)
            n_test_batches: number of batches reserved for the test set (non-zero allows to track test accuracy during training)
            hidden_sizes: a tuple (x_1, ..., x_n) of integers defining the number x_i of hidden neurons in the i-th hidden layer
            epochs: number of epochs for training
            regularization_weight: regularization coefficient in [0, 1]
            label: an optional string label for the classifier (used for tensorboard log file names)
        """
        super().__init__(feature_transform=feature_transform, normalize_input=normalize_input)
        self.in_dimensions = in_dimensions
        self.optimizer_factory = optimizer_factory
        self.n_test_batches = n_test_batches
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.label = label
        self.regularization_weight = regularization_weight

        self.state_dict_file = None  # representation of model for pickling
        self.tensorboard_dir = get_output_directory('training', 'tensorboard')

        # initialize torch-specific fields
        self.model = None
        self.criterion = None
        self._init_torch()

    def _regularized(self, unregularized):
        """
        Computes the regularized loss.
        """
        l2_reg = torch.tensor(0., requires_grad=True)
        for p in self.model.parameters():
            l2_reg = l2_reg + (p*p).sum()   # NOTE: computes squared L2 norm (which is differentiable at 0)
        return unregularized + self.regularization_weight*l2_reg

    def _init_torch(self):
        """
        Initializes the pytorch model.
        """
        self.model = self._get_perceptron_model()

        if self.state_dict_file is not None:
            # load state of model (needed after serialization)
            state_dict = torch.load(self.state_dict_file)
            self.model.load_state_dict(state_dict)

        # binary cross entropy loss
        self.criterion = nn.BCELoss()

    def _get_perceptron_model(self):
        """
        Creates the pytorch model.
        """
        # list of layers
        model = []

        # add layers
        previous_size = self.in_dimensions
        for size in self.hidden_sizes:
            model.append(nn.Linear(previous_size, size))
            model.append(nn.ReLU())
            previous_size = size

        # output layer (size 1)
        model.append(nn.Linear(previous_size, 1))
        model.append(nn.Sigmoid())

        # create end-to-end model
        model = nn.Sequential(*model)

        return model

    def _get_tensorboard_log_dir(self):
        now = datetime.now()
        t = now.strftime("%Y-%m-%d_%H-%M-%S")
        letters = string.ascii_letters
        rand = ''.join(random.choice(letters) for _ in range(10))
        details = [t, rand, self.label]
        details = '_'.join(str(d) for d in details)
        d = os.path.join(self.tensorboard_dir, details)
        log.debug('Tensorboard directory at ' + d)
        return d

    @staticmethod
    def _get_test_set(batch_generator, n_test_batches):
        if n_test_batches > 0:
            test_x_list = []
            test_y_list = []
            for i in range(0, n_test_batches):
                x, y = next(batch_generator)
                test_x_list.append(x)
                test_y_list.append(y)
            x_test = np.vstack(test_x_list)
            y_test = np.vstack(test_y_list)
            return torch.Tensor(x_test), torch.Tensor(y_test)
        else:
            return None, None

    def _train(self, training_batch_generator):
        # get test set from first n_test_batches training batches
        x_test, y_test = MultiLayerPerceptron._get_test_set(training_batch_generator, self.n_test_batches)

        # get optimizer
        optimizer, scheduler = self.optimizer_factory.create_optimizer_with_scheduler(self.model.parameters())

        # logging
        log_dir = self._get_tensorboard_log_dir()
        writer = SummaryWriter(log_dir=log_dir)

        # training
        self.model.train()
        batch_idx = 0
        for x_train, y_train in training_batch_generator:
            batch_idx += 1

            # convert to tensors
            x_train = torch.Tensor(x_train)
            y_train = torch.Tensor(y_train)

            # run epochs on this batch
            self._train_one_batch(batch_idx, x_train, y_train, x_test, y_test, writer, optimizer, scheduler)

        writer.close()

    def _train_one_batch(self, batch_idx: int, x_train, y_train, x_test, y_test, writer, optimizer, scheduler):
        for epoch in range(self.epochs):
            # not really an "epoch" as it does not loop over the whole data set, but only over one batch

            # closure function required for optimizers such as LBFGS that need to compute
            # the gradient of the loss themselves
            def closure():
                # initialize gradients
                optimizer.zero_grad()

                # compute loss (forward pass)
                y_pred = self.model(x_train).squeeze()
                loss = self._regularized(self.criterion(y_pred, y_train))

                # backward pass
                loss.backward()
                return loss

            optimizer.step(closure)
            if scheduler is not None:
                scheduler.step()

            # recompute loss for logging
            loss = closure()

            # logging
            step_idx = (batch_idx-1)*self.epochs + epoch
            writer.add_scalar('Loss/train', loss.item(), step_idx)
            log.debug('Batch {}, epoch {} (global step {}): train loss: {}'.format(batch_idx, epoch, step_idx, loss.item()))

            if x_test is not None:
                # compute test loss
                y_pred_test = self.model(x_test).squeeze()
                loss_test = self.criterion(y_pred_test, y_test.squeeze())
                writer.add_scalar('Loss/test', loss_test.item(), step_idx)

                # compute testing accuracy
                accuracy_test = 1 - torch.mean(torch.abs(torch.round(y_pred_test) - y_test))
                writer.add_scalar('Accuracy/test', accuracy_test.item(), step_idx)

    def _run(self, features):
        """
        Run inference on the trained model.
        """
        features = torch.Tensor(features)
        self.model.eval()
        y_pred = self.model(features)
        return y_pred

    def _predict_probabilities(self, features):
        """
        Compute the probability of class 0 by performing inference on the trained model.
        """
        y_pred = self._run(features)
        y_pred = y_pred.data.numpy()
        # want to return probability of class 0 -> must compute the opposite probability
        probs = 1-y_pred.T[0]
        return np.around(probs, decimals=3)     # round to 3 decimals for numerical stability

    def __str__(self):
        return "MultiLayerPerceptron[normalizer={}, model_structure={}, params={}]"\
            .format(self.get_normalizer_str(),self.model, self.model.state_dict())

    def __getstate__(self):
        # store torch model to a file (for pickling)
        self.state_dict_file = NamedTemporaryFile(
            dir=get_output_directory('training', 'models'),
            prefix='MultiLayerPerceptron_',
            suffix='.model',
            delete=False).name
        state_dict = self.model.state_dict()
        torch.save(state_dict, self.state_dict_file)

        # capture what is normally pickled
        state = self.__dict__.copy()

        # clear torch-specific objects
        state['model'] = None
        state['criterion'] = None

        return state

    def __setstate__(self, new_state):
        # re-instantiate our state from a pickled state
        self.__dict__.update(new_state)
        # move information to model
        self._init_torch()
