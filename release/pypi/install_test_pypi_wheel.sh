conda create -n structuremap_pip_test python=3.8 -y
conda activate structuremap_pip_test
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "structuremap[stable]"
structuremap
conda deactivate
