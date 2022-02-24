![Pip installation](https://github.com/MannLabs/structuremap/workflows/Default%20installation%20and%20tests/badge.svg)
![GUI and PyPi releases](https://github.com/MannLabs/structuremap/workflows/Publish%20on%20PyPi%20and%20release%20on%20GitHub/badge.svg)
[![Downloads](https://pepy.tech/badge/structuremap)](https://pepy.tech/project/structuremap)
[![Downloads](https://pepy.tech/badge/structuremap/month)](https://pepy.tech/project/structuremap)
[![Downloads](https://pepy.tech/badge/structuremap/week)](https://pepy.tech/project/structuremap)


# structuremap
An open-source Python package for integrating information from predicted protein structures deposited in the [AlphaFold database](https://alphafold.ebi.ac.uk/) with proteomics data and specifically with post-translational modifications (PTMs). PTMs on the 3D protein structures can be visulaized by [AlphaMap](https://github.com/MannLabs/alphamap). To enable all hyperlinks in this document, please view it at [GitHub](https://github.com/MannLabs/structuremap).

* [**About**](#about)
* [**License**](#license)
* [**Installation**](#installation)
  * [**Pip installer**](#pip)
  * [**Developer installer**](#developer)
* [**Usage**](#usage)
  * [**Python and jupyter notebooks**](#python-and-jupyter-notebooks)
* [**Troubleshooting**](#troubleshooting)
* [**Citing structuremap**](#citing-structuremap)
* [**How to contribute**](#how-to-contribute)
* [**Changelog**](#changelog)

---
## About

An open-source Python package for integrating information from predicted protein structures deposited in the [AlphaFold database](https://alphafold.ebi.ac.uk/) with proteomics data and specifically with post-translational modifications (PTMs). PTMs on 3D protein structures can be visulaized by [AlphaMap](https://github.com/MannLabs/alphamap).

---
## License

structuremap was developed by the [Mann Labs at the Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann) and the [University of Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/) and is freely available with an [Apache License](LICENSE.txt). External Python packages (available in the [requirements](requirements) folder) have their own licenses, which can be consulted on their respective websites.

---
## Installation

structuremap can be installed and used on all major operating systems (Windows, macOS and Linux).
There are two different types of installation possible:

* [**Pip installer:**](#pip) Choose this installation if you want to use structuremap as a Python package in an existing Python 3.8 environment (e.g. a Jupyter notebook). If needed, the GUI and CLI can be installed with pip as well.
* [**Developer installer:**](#developer) Choose this installation if you are familiar with CLI tools, [conda](https://docs.conda.io/en/latest/) and Python. This installation allows access to all available features of structuremap and even allows to modify its source code directly. Generally, the developer version of structuremap outperforms the precompiled versions which makes this the installation of choice for high-throughput experiments.

### Pip

structuremap can be installed in an existing Python 3.8 environment with a single `bash` command. *This `bash` command can also be run directly from within a Jupyter notebook by prepending it with a `!`*:

```bash
pip install structuremap
```

Installing structuremap like this avoids conflicts when integrating it in other tools, as this does not enforce strict versioning of dependancies. However, if new versions of dependancies are released, they are not guaranteed to be fully compatible with structuremap. While this should only occur in rare cases where dependencies are not backwards compatible, you can always force structuremap to use dependancy versions which are known to be compatible with:

```bash
pip install "structuremap[stable]"
```

NOTE: You might need to run `pip install pip==21.0` before installing structuremap like this. Also note the double quotes `"`.

For those who are really adventurous, it is also possible to directly install any branch (e.g. `@development`) with any extras (e.g. `#egg=structuremap[stable,development-stable]`) from GitHub with e.g.

```bash
pip install "git+https://github.com/MannLabs/structuremap.git@development#egg=structuremap[stable,development-stable]"
```

### Developer

structuremap can also be installed in editable (i.e. developer) mode with a few `bash` commands. This allows to fully customize the software and even modify the source code to your specific needs. When an editable Python package is installed, its source code is stored in a transparent location of your choice. While optional, it is advised to first (create and) navigate to e.g. a general software folder:

```bash
mkdir ~/folder/where/to/install/software
cd ~/folder/where/to/install/software
```

***The following commands assume you do not perform any additional `cd` commands anymore***.

Next, download the structuremap repository from GitHub either directly or with a `git` command. This creates a new structuremap subfolder in your current directory.

```bash
git clone https://github.com/MannLabs/structuremap.git
```

For any Python package, it is highly recommended to use a separate [conda virtual environment](https://docs.conda.io/en/latest/), as otherwise *dependancy conflicts can occur with already existing packages*.

```bash
conda create --name structuremap python=3.8 -y
conda activate structuremap
```

Finally, structuremap and all its [dependancies](requirements) need to be installed. To take advantage of all features and allow development (with the `-e` flag), this is best done by also installing the [development dependencies](requirements/requirements_development.txt) instead of only the [core dependencies](requirements/requirements.txt):

```bash
pip install -e "./structuremap[development]"
```

By default this installs loose dependancies (no explicit versioning), although it is also possible to use stable dependencies (e.g. `pip install -e "./structuremap[stable,development-stable]"`).

***By using the editable flag `-e`, all modifications to the [structuremap source code folder](structuremap) are directly reflected when running structuremap. Note that the structuremap folder cannot be moved and/or renamed if an editable version is installed. In case of confusion, you can always retrieve the location of any Python module with e.g. the command `import module` followed by `module.__file__`.***

---
## Usage

### Python and Jupyter notebooks

structuremap can be imported as a Python package into any Python script or notebook with the command `import structuremap`.

A brief [Jupyter notebook tutorial](nbs/tutorial.ipynb) on how to use the API is also present in the [nbs folder](nbs).

---
## Troubleshooting

In case of issues, check out the following:

* [Issues](https://github.com/MannLabs/structuremap/issues): Try a few different search terms to find out if a similar problem has been encountered before
* [Discussions](https://github.com/MannLabs/structuremap/discussions): Check if your problem or feature requests has been discussed before.

---
## Citing structuremap

If you use structuremap for your work, please cite our paper on biorxiv: https://www.biorxiv.org/content/10.1101/2022.02.23.481596v1

---
## How to contribute

If you like this software, you can give us a [star](https://github.com/MannLabs/structuremap/stargazers) to boost our visibility! All direct contributions are also welcome. Feel free to post a new [issue](https://github.com/MannLabs/structuremap/issues) or clone the repository and create a [pull request](https://github.com/MannLabs/structuremap/pulls) with a new branch. For an even more interactive participation, check out the [discussions](https://github.com/MannLabs/structuremap/discussions) and the [the Contributors License Agreement](misc/CLA.md).

---
## Changelog

See the [HISTORY.md](HISTORY.md) for a full overview of the changes made in each version.
