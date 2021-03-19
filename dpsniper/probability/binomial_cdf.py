from scipy.stats import beta


def lcb(n: int, k: int, alpha: float):
	"""
	Computes a lower confidence bound on the probability parameter p of a binomial CDF.

	Returns:
		The largest p such that Pr[Binom[n,p] >= k] <= alpha
	"""
	if k == 0:
		return 0
	else:
		# Inspired by https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
		return beta.ppf(alpha, k, n-k+1)


def ucb(n: int, k: int, alpha: float):
	"""
	Computes an upper confidence bound on the probability parameter p of a binomial CDF.

	Returns:
		The smallest p such that Pr[Binom[n,p] <= k] <= alpha
	"""
	if k == n:
		return 1
	else:
		return beta.ppf(1-alpha, k+1, n-k)
