conda activate structuremap
python -m unittest test_cli
python -m unittest test_gui
python -m unittest test_processing
jupyter nbconvert --execute --inplace --to notebook --NotebookClient.kernel_name="python" nbs/tutorial.ipynb
conda deactivate
