#!python


__project__ = "structuremap"
__version__ = "0.0.10"
__license__ = "Apache"
__description__ = "An open-source Python package of the AlphaPept ecosystem"
__author__ = "Isabell Bludau & Mann Labs"
__author_email__ = "opensource@alphapept.com"
__github__ = "https://github.com/MannLabs/structuremap"
__keywords__ = [
    "bioinformatics",
    "software",
    "AlphaPept ecosystem",
]
__python_version__ = ">=3.8"
__classifiers__ = [
    "Development Status :: 1 - Planning",
    # "Development Status :: 2 - Pre-Alpha",
    # "Development Status :: 3 - Alpha",
    # "Development Status :: 4 - Beta",
    # "Development Status :: 5 - Production/Stable",
    # "Development Status :: 6 - Mature",
    # "Development Status :: 7 - Inactive"
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]
__console_scripts__ = [
    "structuremap=structuremap.cli:run",
]
__urls__ = {
    "Mann Labs at MPIB": "https://www.biochem.mpg.de/mann",
    "Mann Labs at CPR": "https://www.cpr.ku.dk/research/proteomics/mann/",
    "GitHub": __github__,
    # "ReadTheDocs": None,
    # "PyPi": None,
    # "Scientific paper": None,
}
__extra_requirements__ = {
    "development": "requirements_development.txt",
}
