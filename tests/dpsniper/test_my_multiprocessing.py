import unittest
from collections.abc import Iterable

from dpsniper.utils.my_multiprocessing import MyParallelExecutor
from dpsniper.utils.my_multiprocessing import split_by_batch_size, split_into_parts
from tests.my_test_case import MyTestCase


def f(i):
    return i*i


class TestMyMultiprocessing(MyTestCase):

    def setUp(self):
        self.e = MyParallelExecutor()
        self.e.initialize(1)

    def test_parallel_executor(self):
        inputs = range(10)
        results = self.e.execute(f, inputs)
        expected = [f(i) for i in inputs]
        self.assertEqual(results, expected)

    def check_split_into_batches(self, n, batch_size):
        with self.subTest(n=n, batch_size=batch_size):
            sizes = split_by_batch_size(n, batch_size)
            self.assertIsInstance(sizes, Iterable)
            self.assertEqual(sum(sizes), n)
            for size in sizes:
                self.assertGreaterEqual(size, 0)

    def test_split_into_batches(self):
        for n in [0, 1, 9, 100]:
            for batch_size in [1, 9, 100]:
                self.check_split_into_batches(n, batch_size)

    def check_split_into_parts(self, n, n_parts):
        with self.subTest(n=n, n_parts=n_parts):
            sizes = split_into_parts(n, n_parts)
            self.assertIsInstance(sizes, Iterable)
            self.assertEqual(sum(sizes), n)
            for size in sizes:
                self.assertGreaterEqual(size, 0)

    def test_split_into_parts(self):
        for n in [0, 1, 9, 100]:
            for n_parts in [1, 9, 100]:
                self.check_split_into_parts(n, n_parts)
