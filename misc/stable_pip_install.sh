conda create -n structuremap python=3.8 -y
conda activate structuremap
pip install -e '../.[stable,development-stable]'
structuremap
conda deactivate
