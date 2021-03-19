from dpsniper.classifiers.torch_optimizer_factory import AdamOptimizerFactory
from dpsniper.input.input_domain import InputDomain, InputBaseType
from dpsniper.mechanisms.laplace import LaplaceMechanism, LaplaceFixed
from dpsniper.experiments.experiment_runner import ExperimentRunner
from dpsniper.experiments.my_experiment import MyExperiment
from dpsniper.input.pattern_generator import PatternGenerator
from dpsniper.classifiers.classifier_factory import MultiLayerPerceptronFactory
from dpsniper.classifiers.feature_transformer import BitPatternFeatureTransformer


def run_exp_floating_point(output_path: str, config: 'DDConfig'):
    runner = ExperimentRunner(output_path, "floating_point", log_debug=True)

    mechanism = LaplaceMechanism(eps=0.1)
    domain = InputDomain(1, InputBaseType.FLOAT, [-10, 10])
    sampler = PatternGenerator(domain, False)
    factory = MultiLayerPerceptronFactory(in_dimensions=64,
                                          hidden_sizes=(10, 5),
                                          optimizer_factory=AdamOptimizerFactory(learning_rate=0.1,
                                                                                 step_size=500),
                                          feature_transform=BitPatternFeatureTransformer(),
                                          regularization_weight=0.0001,
                                          epochs=10,
                                          label='Floating')
    exp = MyExperiment("floating_point", mechanism, sampler, factory, config)
    runner.experiments.append(exp)
    runner.run_all(config.n_processes)


def run_exp_floating_fixed(output_path: str, config: 'DDConfig'):
    runner = ExperimentRunner(output_path, "floating_fixed", log_debug=True)

    mechanism = LaplaceFixed(eps=0.1)
    domain = InputDomain(1, InputBaseType.FLOAT, [-10, 10])
    sampler = PatternGenerator(domain, False)
    factory = MultiLayerPerceptronFactory(in_dimensions=64,
                                          hidden_sizes=(10, 5),
                                          optimizer_factory=AdamOptimizerFactory(learning_rate=0.1,
                                                                                 step_size=500),
                                          feature_transform=BitPatternFeatureTransformer(),
                                          regularization_weight=0.0001,
                                          epochs=10,
                                          label='Floating_Fixed')
    exp = MyExperiment("floating_fixed", mechanism, sampler, factory, config)
    runner.experiments.append(exp)
    runner.run_all(config.n_processes)
