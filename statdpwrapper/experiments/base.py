from dpsniper.search.ddconfig import DDConfig
from dpsniper.probability.estimators import EpsEstimator
from dpsniper.utils.my_logging import log, log_context, time_measure
from statdpwrapper.experiments.binary_search import BinarySearch
from statdpwrapper.postprocessing import PostprocessingConfig, the_zero_noise_prng
from statdpwrapper.verification import StatDPAttack, StatDPPrEstimator


config = DDConfig()
p_value = 1 - config.confidence
precision = 0.001
n_samples_detector = 500000


def run_statdp(name: str, algorithm, pp_config: PostprocessingConfig, num_input: tuple, sensitivity, default_kwargs):
    with log_context(name):
        try:
            log.info("Running StatDP binary search...")
            with time_measure("statdp_time"):
                eps, a1, a2, event, postprocessing, a0 = BinarySearch(
                        algorithm,
                        num_input,
                        sensitivity,
                        n_samples_detector,
                        default_kwargs,
                        pp_config).find(p_value, precision)
            log.info("StatDP result: [eps=%f, a1=%s, a2=%s, event=%s, postprocessing=%s, a0=%s]", eps, a1, a2,
                     str(event), str(postprocessing), a0)

            log.info("Verifying epsilon using %d samples...", config.n_final)
            with time_measure("statdp_verification_time"):
                attack = StatDPAttack(event, postprocessing)
                if postprocessing.requires_noisefree_reference:
                    noisefree_reference = algorithm(the_zero_noise_prng, a0, **default_kwargs)
                    attack.set_noisefree_reference(noisefree_reference)
                pr_estimator = StatDPPrEstimator(algorithm,
                                                 config.n_final,
                                                 config,
                                                 use_parallel_executor=True,
                                                 **default_kwargs)
                eps_verified, eps_lcb = EpsEstimator(pr_estimator, allow_swap=True)\
                    .compute_eps_estimate(a1, a2, attack)
            log.info("Verified eps=%f (lcb=%f)", eps_verified, eps_lcb)
            log.data("statdp_result", {"eps": eps_verified,
                                       "eps_lcb": eps_lcb,
                                       "eps_preliminary": eps,
                                       "a1": a1,
                                       "a2": a2,
                                       "event": event,
                                       "postprocessing": str(postprocessing)})
        except Exception:
            log.error("Exception while running StatDP on %s", name, exc_info=True)
