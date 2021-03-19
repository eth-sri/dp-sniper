# DP-Sniper

A machine-learning-based tool for discovering differential privacy violations in black-box algorithms.

## Install

We recommend installing DP-Sniper using
[conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

After installing conda, you can install DP-Sniper by running its installation
script [install.sh](./install.sh):

```bash
bash ./install.sh
```

You can ignore the warning `ResourceWarning: unclosed running multiprocessing
pool`.

_Note: The above steps are sufficient to use the main package `dpsniper`. If you would like to run the experiments from the SP 2021 research paper, you have to follow additional installation steps as described in [eval_sp2021/README.md](eval_sp2021/README.md))._

## Basic Usage

The following command tests the differential privacy of the Laplace mechanism,
explained in detail in file [dpsniper/example.py](dpsniper/example.py):

```bash
conda activate dp-sniper
python dpsniper/example.py # may take a while due to an extensive final confirmation
```

This commands stores temporary outputs and log files to the folder
`example_outputs` of the current working directory.

### Testing Your Own Mechanism

DP-Sniper is a black-box approach. To run DP-Sniper or DD-Search on your own
mechanism, you only have to implement the method `m` of the abstract class
`Mechanism` defined in
[dpsniper/mechanisms/abstract.py](dpsniper/mechanisms/abstract.py) and modify
the code snippet in [dpsniper/example.py](dpsniper/example.py). See
[dpsniper/mechanisms](dpsniper/mechanisms) for example implementations of
popular mechanisms.

## Publication

This is an implementation of the approach presented in the following research paper:

> B. Bichsel, S. Steffen, I. Bogunovic and M. Vechev. 2021.
> DP-Sniper: Black-Box Discovery of Differential Privacy Violations using Classifiers.
> In IEEE Symposium on Security and Privacy (SP 2021).

The main algorithms DD-Search and DP-Sniper from the paper can be found in
[dpsniper/search/ddsearch.py](dpsniper/search/ddsearch.py) and
[dpsniper/attack/dpsniper.py](dpsniper/attack/dpsniper.py), respectively.

### Citing this Work

You are encouraged to cite the above publication using the following BibTeX entry
if you use DP-Sniper for academic research.

    TODO: Add bibtex

### Evaluation

You can find instructions on how to reproduce the evaluation results of our paper in the folder [eval_sp2021](eval_sp2021/README.md).

## License

MIT License, see [LICENSE](LICENSE).

This repository includes third-party code from
[statdp](https://github.com/cmla-psu/statdp), marked as `MIT License, Copyright
(c) 2018-2019 Yuxin Wang`.
