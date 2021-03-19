#!/bin/bash
# Installs dpsniper, statdp and statdpwrapper

# to enable activating conda from within this script
eval "$(conda shell.bash hook)"

echo "Creating conda environment 'dp-sniper'..."
conda create -y -n dp-sniper python=3.8

echo "Activating conda environment..."
conda activate dp-sniper

echo "Installing dpsniper..."
pip install .

echo "Running dpsniper tests..."
make tests

echo "Initializing statdp submodule..."
git submodule init
git submodule update

echo "Installing statdp prerequisites..."
conda install -y numba sympy tqdm coloredlogs
conda install -y -c intel icc_rt

echo "Installing statdp..."
cd statdp
pip install .
cd ..

echo "Running statdpwrapper tests..."
make tests-statdpwrapper
