import os

import numpy as np
from sklearn.datasets import make_blobs
import tempfile
import pickle

from dpsniper.classifiers.multi_layer_perceptron import MultiLayerPerceptron
from dpsniper.classifiers.torch_optimizer_factory import AdamOptimizerFactory, LBFGSOptimizerFactory, \
	SGDOptimizerFactory
from dpsniper.utils.paths import get_output_directory
from tests.my_test_case import MyTestCase


def blob_label(y, label, loc):
	"""
	reassign all labels in loc to label
	"""
	target = np.copy(y)
	for l in loc:
		target[y == l] = label
	return target


def training_batch_generator():
	for i in range(4):
		# x_train: features, y_train: labels
		x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True)
		# re-assign class 0->0 and classes 1,2,3->1
		y_train = blob_label(y_train, 0, [0])
		y_train = blob_label(y_train, 1, [1, 2, 3])
		yield x_train, y_train


x_test, y_test = make_blobs(n_samples=10, n_features=2, cluster_std=1.5, shuffle=True)
# re-assign class 0->0 and classes 1,2,3->1
y_test = blob_label(y_test, 0, [0])
y_test = blob_label(y_test, 1, [1, 2, 3])


class TestMLP(MyTestCase):

	def test_train_adam(self):
		# generate classifier
		model = MultiLayerPerceptron(in_dimensions=2, optimizer_factory=AdamOptimizerFactory(), hidden_sizes=(50, 50))

		# train
		model.train(training_batch_generator())

		# get probabilities
		probs = model.predict_probabilities(x_test)
		self.assertEqual(len(probs.shape), 1)
		self.assertEqual(probs.shape[0], 10)

		# to string
		s = str(model)
		self.assertIn('MultiLayerPerceptron', s)

	def test_train_lbfgs(self):
		# generate classifier
		model = MultiLayerPerceptron(in_dimensions=2, optimizer_factory=LBFGSOptimizerFactory(), hidden_sizes=(50, 50))

		# train
		model.train(training_batch_generator())

		# get probabilities
		probs = model.predict_probabilities(x_test)
		self.assertEqual(len(probs.shape), 1)
		self.assertEqual(probs.shape[0], 10)

		# to string
		s = str(model)
		self.assertIn('MultiLayerPerceptron', s)

	def test_train_sgd(self):
		# generate classifier
		model = MultiLayerPerceptron(in_dimensions=2, optimizer_factory=SGDOptimizerFactory(), hidden_sizes=(50, 50))

		# train
		model.train(training_batch_generator())

		# get probabilities
		probs = model.predict_probabilities(x_test)
		self.assertEqual(len(probs.shape), 1)
		self.assertEqual(probs.shape[0], 10)

		# to string
		s = str(model)
		self.assertIn('MultiLayerPerceptron', s)

	def test_pickle(self):
		model = MultiLayerPerceptron(in_dimensions=2, optimizer_factory=LBFGSOptimizerFactory(), hidden_sizes=(50, 50))
		model.train(training_batch_generator())
		probs1 = model.predict_probabilities(x_test)
		filename = tempfile.mktemp(dir=get_output_directory("tmp"))
		with open(filename, "wb") as f:
			pickle.dump(model, f)
		with open(filename, "rb") as f:
			obj = pickle.load(f)
		os.remove(filename)
		assert(isinstance(obj, MultiLayerPerceptron))
		probs2 = obj.predict_probabilities(x_test)
		np.testing.assert_array_equal(probs1, probs2)
