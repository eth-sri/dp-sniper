from dpsniper.classifiers.feature_transformer import FlagsFeatureTransformer
from dpsniper.classifiers.torch_optimizer_factory import SGDOptimizerFactory, \
    AdamOptimizerFactory
from dpsniper.classifiers.classifier_factory import *
from dpsniper.input.input_domain import InputDomain, InputBaseType
from dpsniper.input.pattern_generator import PatternGenerator
from dpsniper.mechanisms.parallel import *
from dpsniper.mechanisms.noisy_hist import *
from dpsniper.mechanisms.rappor import *
from dpsniper.mechanisms.prefix_sum import *
from dpsniper.mechanisms.report_noisy_max import *
from dpsniper.mechanisms.geometric import TruncatedGeometricMechanism
from dpsniper.experiments.experiment_runner import ExperimentRunner
from dpsniper.experiments.my_experiment import MyExperiment


def class_name(obj):
    return type(obj).__name__.split(".")[-1]


def run_exp_default(series_name: str, use_mlp: bool, output_path: str, config: 'DDConfig'):
    runner = ExperimentRunner(output_path, series_name, log_debug=True)

    def run_dpsniper(mechanism, input_generator: PatternGenerator, output_size, feature_transform=None):
        if use_mlp:
            factory = MultiLayerPerceptronFactory(
                in_dimensions=output_size,
                hidden_sizes=(10, 5),
                optimizer_factory=AdamOptimizerFactory(learning_rate=0.1,
                                                       step_size=500),
                feature_transform=feature_transform,
                regularization_weight=0.0001,
                epochs=10,
                label=class_name(mechanism))
        else:
            factory = LogisticRegressionFactory(
                feature_transform=feature_transform,
                in_dimensions=output_size,
                optimizer_factory=SGDOptimizerFactory(learning_rate=0.3, momentum=0.3, step_size=500),
                regularization_weight=0.001,
                epochs=10,
                label=class_name(mechanism))
        runner.experiments.append(MyExperiment(class_name(mechanism), mechanism, input_generator, factory, config))

    domain = InputDomain(1, InputBaseType.FLOAT, [-10, 10])
    run_dpsniper(LaplaceMechanism(), PatternGenerator(domain, False), 1)

    domain = InputDomain(1, InputBaseType.INT, [0, 5])
    run_dpsniper(TruncatedGeometricMechanism(), PatternGenerator(domain, False), 1)

    domain = InputDomain(5, InputBaseType.INT, [0, 10])
    run_dpsniper(NoisyHist1(), PatternGenerator(domain, False), 5)
    run_dpsniper(NoisyHist2(), PatternGenerator(domain, False), 5)

    domain = InputDomain(5, InputBaseType.FLOAT, [-10, 10])
    run_dpsniper(ReportNoisyMax1(), PatternGenerator(domain, True), 1)
    run_dpsniper(ReportNoisyMax2(), PatternGenerator(domain, True), 1)
    run_dpsniper(ReportNoisyMax3(), PatternGenerator(domain, True), 1)
    run_dpsniper(ReportNoisyMax4(), PatternGenerator(domain, True), 1)

    domain = InputDomain(10, InputBaseType.FLOAT, [-10, 10])
    run_dpsniper(SparseVectorTechnique1(c=1, t=0.5), PatternGenerator(domain, True), 20,
                 feature_transform=FlagsFeatureTransformer([-1]))
    run_dpsniper(SparseVectorTechnique2(c=1, t=1), PatternGenerator(domain, True), 20,
                 feature_transform=FlagsFeatureTransformer([-1]))
    run_dpsniper(SparseVectorTechnique3(c=1, t=1), PatternGenerator(domain, True), 30,
                 feature_transform=FlagsFeatureTransformer([-1000.0, -2000.0]))
    run_dpsniper(SparseVectorTechnique4(c=1, t=1), PatternGenerator(domain, True), 20,
                 feature_transform=FlagsFeatureTransformer([-1]))
    run_dpsniper(SparseVectorTechnique5(c=1, t=1), PatternGenerator(domain, True), 10)
    run_dpsniper(SparseVectorTechnique6(c=1, t=1), PatternGenerator(domain, True), 10)

    domain = InputDomain(1, InputBaseType.INT, [-10, 10])
    r = Rappor()
    run_dpsniper(r, PatternGenerator(domain, False), r.filter_size)
    otr = OneTimeRappor()
    run_dpsniper(otr, PatternGenerator(domain, False), otr.filter_size)

    domain = InputDomain(1, InputBaseType.FLOAT, [-10, 10])
    lp = LaplaceParallel(eps=0.005, n_parallel=20)
    run_dpsniper(lp, PatternGenerator(domain, False), lp.get_n_parallel())

    domain = InputDomain(10, InputBaseType.FLOAT, [-10, 10])
    run_dpsniper(SVT34Parallel(), PatternGenerator(domain, True), 80,
                 feature_transform=FlagsFeatureTransformer([-1, -1000.0, -2000.0]))
    run_dpsniper(PrefixSum(), PatternGenerator(domain, True), 10)
    run_dpsniper(NumericalSVT(), PatternGenerator(domain, True), 10)

    runner.run_all(config.n_processes)
