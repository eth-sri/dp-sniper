from dpsniper.mechanisms.geometric import TruncatedGeometricMechanism
from dpsniper.mechanisms.noisy_hist import *
from dpsniper.mechanisms.report_noisy_max import *
from dpsniper.mechanisms.rappor import *
from dpsniper.mechanisms.parallel import *
from dpsniper.mechanisms.prefix_sum import PrefixSum
from dpsniper.mechanisms.laplace import *
from tests.my_test_case import MyTestCase


class TestMechanisms(MyTestCase):

	def test_laplace(self):
		m = LaplaceMechanism()
		bs = m.m(np.array([10.3]), 1000)
		self.assertEqual(bs.shape, (1000,))

	def test_geometric(self):
		m = TruncatedGeometricMechanism()
		bs = m.m(np.array([3]), 1000)
		self.assertEqual(bs.shape, (1000,))

	def test_svt1(self):
		m = SparseVectorTechnique1()
		bs = m.m(np.array([2.4, 3.6, 2.1, 1.0]), 1000)
		self.assertEqual(bs.shape, (1000, 4))

	def test_svt2(self):
		m = SparseVectorTechnique2()
		bs = m.m(np.array([2.4, 3.6, 2.1, 1.0]), 1000)
		self.assertEqual(bs.shape, (1000, 4))

	def test_svt3(self):
		m = SparseVectorTechnique3()
		bs = m.m(np.array([2.4, 3.6, 2.1, 1.0]), 1000)
		self.assertEqual(bs.shape, (1000, 4))

	def test_svt4(self):
		m = SparseVectorTechnique4()
		bs = m.m(np.array([2.4, 3.6, 2.1, 1.0]), 1000)
		self.assertEqual(bs.shape, (1000, 4))

	def test_svt5(self):
		m = SparseVectorTechnique5()
		bs = m.m(np.array([2.4, 3.6, 2.1, 1.0]), 1000)
		self.assertEqual(bs.shape, (1000, 4))

	def test_svt6(self):
		m = SparseVectorTechnique6()
		bs = m.m(np.array([2.4, 3.6, 2.1, 1.0]), 1000)
		self.assertEqual(bs.shape, (1000, 4))

	def test_noisy_hist1(self):
		m = NoisyHist1()
		bs = m.m(np.array([3, 6, 1, 0]), 1000)
		self.assertEqual(bs.shape, (1000, 4))

	def test_noisy_hist2(self):
		m = NoisyHist2()
		bs = m.m(np.array([3, 6, 1, 0]), 1000)
		self.assertEqual(bs.shape, (1000, 4))

	def test_report_noisy_max1(self):
		m = ReportNoisyMax1()
		bs = m.m(np.array([2.4, 3.6, 2.1, 1.0]), 1000)
		self.assertEqual(bs.shape, (1000, ))

	def test_report_noisy_max2(self):
		m = ReportNoisyMax2()
		bs = m.m(np.array([2.4, 3.6, 2.1, 1.0]), 1000)
		self.assertEqual(bs.shape, (1000, ))

	def test_report_noisy_max3(self):
		m = ReportNoisyMax3()
		bs = m.m(np.array([2.4, 3.6, 2.1, 1.0]), 1000)
		self.assertEqual(bs.shape, (1000, ))

	def test_report_noisy_max4(self):
		m = ReportNoisyMax4()
		bs = m.m(np.array([2.4, 3.6, 2.1, 1.0]), 1000)
		self.assertEqual(bs.shape, (1000, ))

	def test_rappor(self):
		m = Rappor()
		bs = m.m(np.array([3.4]), 1000)
		self.assertEqual(bs.shape, (1000, 20))

	def test_one_time_rappor(self):
		m = OneTimeRappor()
		bs = m.m(np.array([3.4]), 1000)
		self.assertEqual(bs.shape, (1000, 20))

	def test_parallel_1(self):
		m1 = LaplaceMechanism()
		m2 = TruncatedGeometricMechanism()
		m = ParallelMechanism([m1, m2])
		bs = m.m(np.array([3]), 1000)
		self.assertEqual(bs.shape, (1000, 2))

	def test_parallel_2(self):
		m1 = SparseVectorTechnique1()
		m2 = SparseVectorTechnique2()
		m = ParallelMechanism([m1, m2])
		bs = m.m(np.array([3, 4]), 1000)
		self.assertEqual(bs.shape, (1000, 4))

	def test_laplace_parallel(self):
		m = LaplaceParallel(10)
		bs = m.m(np.array([3]), 1000)
		self.assertEqual(bs.shape, (1000, 10))

	def test_svt_parallel(self):
		m = SVT34Parallel()
		bs = m.m(np.array([3, 4]), 1000)
		self.assertEqual(bs.shape, (1000, 4))

	def test_prefix_sum(self):
		m = PrefixSum()
		bs = m.m(np.array([2.3, 5.4, 2.3]), 10)
		self.assertEqual(bs.shape, (10, 3))

	def test_numerical_svt(self):
		m = NumericalSVT(eps=10)
		bs = m.m(np.array([0.8, 1.3, 4.0, 0.3]), 10)
		self.assertEqual(bs.shape, (10, 4))

	def test_laplace_fixed(self):
		m = LaplaceFixed()
		bs = m.m(np.array([3.2]), 1000)
		self.assertEqual(bs.shape, (1000,))
