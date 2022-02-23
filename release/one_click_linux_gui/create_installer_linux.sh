#!bash

# Initial cleanup
rm -rf dist
rm -rf build
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n structuremap_installer python=3.8 -y
conda activate structuremap_installer

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/one_click_linux_gui
# Make sure you include the required extra packages and always use the stable or very-stable options!
pip install "../../dist/structuremap-0.0.3-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==4.2
pyinstaller ../pyinstaller/structuremap.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../structuremap/data/*.fasta dist/structuremap/data
# WARNING: this probably does not work!!!!

# Wrapping the pyinstaller folder in a .deb package
mkdir -p dist/structuremap_gui_installer_linux/usr/local/bin
mv dist/structuremap dist/structuremap_gui_installer_linux/usr/local/bin/structuremap
mkdir dist/structuremap_gui_installer_linux/DEBIAN
cp control dist/structuremap_gui_installer_linux/DEBIAN
dpkg-deb --build --root-owner-group dist/structuremap_gui_installer_linux/
