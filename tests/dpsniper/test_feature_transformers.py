import unittest

from dpsniper.classifiers.feature_transformer import *
from tests.my_test_case import MyTestCase


class TestFeatureTransformers(MyTestCase):

    def test_1_flag_transformer(self):
        trans = FlagsFeatureTransformer([-1])
        x = np.array([[-3, -1, 4], [-3, 2, -3]])
        y = trans.transform(x)
        np.testing.assert_array_equal(y, np.array([[-3, 0, 4, 0, 1, 0],
                                                   [-3, 2, -3, 0, 0, 0]]))

    def test_2_flags_transformer(self):
        trans = FlagsFeatureTransformer([-1, -3])
        x = np.array([[-3, -1, 4], [-3, 2, -3]])
        y = trans.transform(x)
        np.testing.assert_array_equal(y, np.array([[0, 0, 4, 0, 1, 0, 1, 0, 0],
                                                   [0, 2, 0, 0, 0, 0, 1, 0, 1]]))
