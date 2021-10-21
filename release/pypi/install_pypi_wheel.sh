conda create -n structuremap_pip_test python=3.8 -y
conda activate structuremap_pip_test
pip install "structuremap[stable]"
structuremap
conda deactivate
