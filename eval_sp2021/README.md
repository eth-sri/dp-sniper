# SP 2021 Evaluation

The instructions below describe how to reproduce the experiment results from the original SP 2021 publication.

## Installation

To install the prerequisites for running the evaluation, run the installation
script [install-statdp.sh](/install-statdp.sh)

```bash
bash ./install-statdp.sh
```

This installs a specific version of [statdp](https://github.com/cmla-psu/statdp)
(commit `16f389baf41b047dc6e70c334a8c49a7e00b5b7c` of 2020-05-14) provided in
the submodule [statdp](https://github.com/eth-sri/statdp) and relies on a
wrapper provided in `/statdpwrapper`.

## Running Experiments

Execute the following commands in the project root directory to run the
experiments. Depending on your machine, running all experiments may take a _very
long time (in the order of days)!_

```bash
# remember to activate the environment
conda activate dp-sniper

# run experiments with logistic regression
make exp-logistic

# run experiments with neural network
make exp-neuralnet

# run floating point vulnerability experiment
make exp-floating

# run floating point vulnerability experiment with snapping mechanism
make exp-floating-fixed

# run statdp experiments
make exp-statdp
```

These commands generate output files in the folder `out`. The relevant machine-readable data can be found in the files `out/logs/*_data.log`.

## Creating Plots

The plots used in the publication can be reproduced from the reference results provided in `reference_results/` as follows (run in the project root directory):

```bash
# remember to activate the environment
conda activate dp-sniper

# generates the plots (requires latex to be installed for fonts)
make plots
```

This prints some statistics and generates the following two plots: `eval_sp2021/plots/eval-powers.pdf` and `eval_sp2021/plots/eval-runtimes.pdf`.
