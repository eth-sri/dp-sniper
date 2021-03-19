#!/bin/bash
# Install statdp and statdpwrapper

# enable activating conda from within this script
eval "$(conda shell.bash hook)"

echo "Activating conda environment before executing any further command..."
conda activate dp-sniper

###########
# INSTALL #
###########

echo "Initializing statdp submodule..."
git submodule init
git submodule update

echo "Installing statdp dependencies..."
conda install -y numba sympy tqdm coloredlogs
conda install -y -c intel icc_rt

echo "Installing statdp..."
cd statdp
pip install .
cd ..

#########
# TESTS #
#########
# verify successful installation of statdp

echo "Running statdpwrapper tests..."
make tests-statdpwrapper
