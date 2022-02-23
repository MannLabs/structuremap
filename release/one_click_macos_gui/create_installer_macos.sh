#!bash

# Initial cleanup
rm -rf dist
rm -rf build
FILE=structuremap.pkg
if test -f "$FILE"; then
  rm structuremap.pkg
fi
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n structuremapinstaller python=3.8 -y
conda activate structuremapinstaller

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/one_click_macos_gui
pip install "../../dist/structuremap-0.0.3-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==4.2
pyinstaller ../pyinstaller/structuremap.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../structuremap/data/*.fasta dist/structuremap/data

# Wrapping the pyinstaller folder in a .pkg package
mkdir -p dist/structuremap/Contents/Resources
cp ../logos/alpha_logo.icns dist/structuremap/Contents/Resources
mv dist/structuremap_gui dist/structuremap/Contents/MacOS
cp Info.plist dist/structuremap/Contents
cp structuremap_terminal dist/structuremap/Contents/MacOS
cp ../../LICENSE.txt Resources/LICENSE.txt
cp ../logos/alpha_logo.png Resources/alpha_logo.png
chmod 777 scripts/*

pkgbuild --root dist/structuremap --identifier de.mpg.biochem.structuremap.app --version 0.0.3 --install-location /Applications/structuremap.app --scripts scripts structuremap.pkg
productbuild --distribution distribution.xml --resources Resources --package-path structuremap.pkg dist/structuremap_gui_installer_macos.pkg
