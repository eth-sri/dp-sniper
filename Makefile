# use bash
SHELL=/bin/bash
# base directory for all outputs
OUT_BASE=$(CURDIR)/out

#########
# TESTS #
#########

.PHONY: tests
tests:
	@echo "Running dpsniper tests"
	python -m unittest discover -s tests/dpsniper

.PHONY: tests-statdpwrapper
tests-statdpwrapper:
	@echo "Running statdpwrapper tests"
	python -m unittest discover -s tests/statdpwrapper

###############
# EXPERIMENTS #
###############

.PHONY: exp-logistic
exp-logistic:
	@echo "Running DD-Search on all mechanisms using logistic regression classifier (results in $(OUT_BASE)/logs/dd_search_reg*)"
	python -m dpsniper.experiments --reg --output-dir "$(OUT_BASE)"

.PHONY: exp-neuralnet
exp-neuralnet:
	@echo "Running DD-Search on all mechanisms using neural network classifier (results in $(OUT_BASE)/logs/dd_search_mlp*)"
	python -m dpsniper.experiments --mlp --output-dir "$(OUT_BASE)"

.PHONY: exp-floating
exp-floating:
	@echo "Running DD-Search floating point experiment (results in $(OUT_BASE)/logs/floating_point*)"
	python -m dpsniper.experiments --floating  --output-dir "$(OUT_BASE)" --processes 4

.PHONY: exp-floating-fixed
exp-floating-fixed:
	@echo "Running DD-Search fixed floating point experiment (results in $(OUT_BASE)/logs/floating_fixed*)"
	python -m dpsniper.experiments --floating-fixed --output-dir "$(OUT_BASE)" --processes 4

.PHONY: exp-statdp
exp-statdp:
	@echo "Running StatDP-Fixed on all mechanisms (results in $(OUT_BASE)/logs/statdp*)"
	python -m statdpwrapper.experiments --output-dir "$(OUT_BASE)" --postfix "_1"
	python -m statdpwrapper.experiments --output-dir "$(OUT_BASE)" --postfix "_2"

#########
# PLOTS #
#########

.PHONY: plots
plots:
	@echo "Creating plots (results in eval_sp2021/plots)"
	python eval_sp2021/plots/create_plots.py --data-dir "eval_sp2021/reference-results" --output-dir "eval_sp2021/plots"

###############
# CONVENIENCE #
###############

.PHONY: tensorboard
tensorboard:
	@echo "Running tensorboard to display progress of training"
	tensorboard --logdir="$(OUT_BASE)/training/tensorboard" --reload_multifile=True

.PHONY: clean
clean:
	@echo "Cleaning output folder"
	rm -rf $(OUT_BASE)
