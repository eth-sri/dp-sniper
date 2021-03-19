# DP-Sniper

A machine-learning-based tool for discovering differential privacy violations in black-box algorithms.

This is an implementation of the approach presented in the following research paper:

> B. Bichsel, S. Steffen, I. Bogunovic and M. Vechev. 2021.
> DP-Sniper: Black-Box Discovery of Differential Privacy Violations using Classifiers.
> In IEEE Symposium on Security and Privacy (SP 2021).

## Install

Install DP-Sniper by running its installation script [install.sh](./install.sh):

```bash
bash ./install.sh
```

You can ignore the warning `ResourceWarning: unclosed running multiprocessing
pool`.

_Note: The above steps are sufficient to use the main package `dpsniper`. If you would like to run the experiments from the SP 2021 research paper, you have to follow additional installation steps as described [below](README.md#sp-2021-evaluation)._

## Basic Usage

The main algorithms DD-Search and DP-Sniper from the publication can be found in [dpsniper/search/ddsearch.py](dpsniper/search/ddsearch.py) and [dpsniper/attack/dpsniper.py](dpsniper/attack/dpsniper.py), respectively.

### Example: Testing the Laplace Mechanism

Below, we provide a minimal example snippet testing differential privacy for the Laplace mechanism (see [dpsniper/example.py](dpsniper/example.py) for the complete code). The example uses Logistic regression with stochastic gradient descent optimization as the underlying machine-learning algorithm. It considers 1-dimensional floating point input pairs from the range [-10, 10] with maximum distance of 1. The DD-Search algorithm wraps the core DP-Sniper algorithm and is used to find the witness. The last line re-computes the power of the witness using high precision for a tighter lower confidence bound. When executing the example, temporary outputs and log files will be stored to the folder `example_outputs` of the current working directory.

```python
# create mechanism
mechanism = LaplaceMechanism()

# configuration
classifier_factory = LogisticRegressionFactory(in_dimensions=1, optimizer_factory=SGDOptimizerFactory())
input_generator = PatternGenerator(InputDomain(1, InputBaseType.FLOAT, [-10, 10]), False)
config = DDConfig(n_processes=2)

with initialize_dpsniper(config, out_dir="example_outputs"):
    # run DD-Search
    witness = DDSearch(mechanism, DPSniper(mechanism, classifier_factory, config), input_generator, config).run()

    # re-estimate epsilon and lower bound with high precision
    witness.compute_eps_high_precision(mechanism, config)
```

### Testing Your Own Mechanism

DP-Sniper is a black-box approach. To run DP-Sniper or DD-Search on your own mechanism, you only have to implement the method `m` of the abstract class `Mechanism` defined in [dpsniper/mechanisms/abstract.py](dpsniper/mechanisms/abstract.py) and modify the code snippet above. See [dpsniper/mechanisms](dpsniper/mechanisms) for example implementations of popular mechanisms.

## SP 2021 Evaluation

You can find instructions on how to reproduce the evaluation results of our SP
2021 paper in the folder [eval_sp2021](eval_sp2021/README.md).

## Citing this Work

You are encouraged to cite the following research paper if you use DP-Sniper for academic research.

    TODO: Add bibtex

## License

MIT License, see [LICENSE](LICENSE).

This repository includes third-party code from
[statdp](https://github.com/cmla-psu/statdp), marked as `MIT License, Copyright
(c) 2018-2019 Yuxin Wang`.
