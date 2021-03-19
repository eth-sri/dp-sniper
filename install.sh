#!/bin/bash
# Installs dpsniper using conda
# 
# Installing conda:
# https://conda.io/projects/conda/en/latest/user-guide/install/index.html

# enable activating conda from within this script
eval "$(conda shell.bash hook)"

################
# INSTALLATION #
################

echo "Creating conda environment 'dp-sniper'..."
conda create -y -n dp-sniper python=3.8

echo "Activating conda environment before executing any further command..."
conda activate dp-sniper

echo "Installing package dpsniper..."
pip install .

#########
# TESTS #
#########
# to verify successful installation

echo "Running dpsniper unit tests..."
make tests
