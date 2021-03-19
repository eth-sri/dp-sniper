import os
import unittest

from dpsniper.utils.paths import set_output_directory
from dpsniper.utils.torch import torch_initialize

here = os.path.dirname(os.path.realpath(__file__))


class MyTestCase(unittest.TestCase):
    """
    Generic wrapper class that initializes torch and sets the output directory.
    """

    def setUp(self):
        set_output_directory(os.path.join(here, "..", "out"))
        torch_initialize()
