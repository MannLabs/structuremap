conda activate structuremap
python -m unittest test_cli
python -m unittest test_gui
python -m unittest test_processing
conda deactivate
