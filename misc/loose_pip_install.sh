conda create -n structuremap python=3.8 -y
conda activate structuremap
pip install -e '../.[development]'
structuremap
conda deactivate
