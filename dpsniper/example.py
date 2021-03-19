from dpsniper.attack.dpsniper import DPSniper
from dpsniper.classifiers.classifier_factory import LogisticRegressionFactory
from dpsniper.classifiers.torch_optimizer_factory import SGDOptimizerFactory
from dpsniper.search.ddconfig import DDConfig
from dpsniper.search.ddsearch import DDSearch
from dpsniper.input.input_domain import InputDomain, InputBaseType
from dpsniper.input.pattern_generator import PatternGenerator
from dpsniper.mechanisms.laplace import LaplaceMechanism
from dpsniper.utils.initialization import initialize_dpsniper

if __name__ == "__main__":
    # create mechanism
    mechanism = LaplaceMechanism()

    # configuration
	# use Logistic regression with stochastic gradient descent optimization as the underlying machine-learning algorithm
    classifier_factory = LogisticRegressionFactory(in_dimensions=1, optimizer_factory=SGDOptimizerFactory())
	# consider 1-dimensional floating point input pairs from the range [-10, 10] with maximum distance of 1
    input_generator = PatternGenerator(InputDomain(1, InputBaseType.FLOAT, [-10, 10]), False)
	# adapt the number of processes to fit your machine
    config = DDConfig(n_processes=2)

    with initialize_dpsniper(config, out_dir="example_outputs"):
        # run DD-Search to find the witness
        witness = DDSearch(mechanism, DPSniper(mechanism, classifier_factory, config), input_generator, config).run()

		# re-compute the power of the witness using high precision for a tighter lower confidence bound
        witness.compute_eps_high_precision(mechanism, config)

    print("eps_lcb = {}".format(witness.lower_bound))
    print("witness = ({}, {}, {})".format(witness.a1, witness.a2, witness.attack))
