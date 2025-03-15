"""
PRGminer: Deep Neural Network-Based Plant Resistance Gene Prediction
Copyright (C) 2023 Naveen Duhan

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author: Naveen Duhan (naveen.duhan@usu.edu)
Organization: Kaundal Bioinformatics Lab, Utah State University
Website: https://bioinfo.usu.edu/PRGminer
"""

import os
import setuptools
from pathlib import Path

# Package metadata
NAME = "PRGminer"
DESCRIPTION = "Deep learning-based plant resistance gene prediction and classification tool"
AUTHOR = "Naveen Duhan"
AUTHOR_EMAIL = "naveen.duhan@usu.edu"
MAINTAINER = "Naveen Duhan"
MAINTAINER_EMAIL = "naveen.duhan@usu.edu"
URL = "https://bioinfo.usu.edu/PRGminer"
DOCUMENTATION = "https://bioinfo.usu.edu/PRGminer/docs"
REPOSITORY = "https://github.com/navduhan/PRGminer"
LICENSE = "GPL-3.0"
PYTHON_REQUIRES = ">=3.8,<3.11"

# Version information
VERSION = "0.1.0"  # Following semantic versioning

# Get the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

def read_requirements(filename="requirements.txt"):
    """Read requirements from file."""
    try:
        with open(filename) as f:
            return [line.strip() for line in f 
                   if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

# Core package requirements with versions
INSTALL_REQUIRES = [
    'numpy>=1.24.0',
    'pandas>=2.1.0',
    'tensorflow>=2.13.0',
    'keras>=2.13.1',
    'biopython>=1.81',
    'scikit-learn>=1.3.0',
    'scipy>=1.11.0',
    'h5py>=3.9.0',
    'matplotlib>=3.7.2',
    'seaborn>=0.12.2',
    'tqdm>=4.66.1'
]

# Optional dependencies with specific use cases
EXTRAS_REQUIRE = {
    'gpu': [
        'tensorflow-gpu>=2.13.0',
        'cudatoolkit>=11.8.0',
        'cudnn>=8.7.0'
    ],
    'dev': [
        'pytest>=7.4.2',
        'pytest-cov>=4.1.0',
        'black>=23.9.1',
        'flake8>=6.1.0',
        'mypy>=1.5.1',
        'isort>=5.12.0'
    ],
    'docs': [
        'sphinx>=7.1.2',
        'sphinx-rtd-theme>=1.3.0',
        'sphinx-autodoc-typehints>=1.24.0'
    ],
    'viz': [
        'plotly>=5.16.1',
        'dash>=2.13.0'
    ]
}

setuptools.setup(
    # Basic package information
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author information
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    
    # Project URLs
    url=URL,
    project_urls={
        "Documentation": DOCUMENTATION,
        "Source Code": REPOSITORY,
        "Bug Tracker": f"{REPOSITORY}/issues",
        "Changelog": f"{REPOSITORY}/blob/master/CHANGELOG.md",
    },
    
    # Package configuration
    packages=setuptools.find_packages(exclude=[
        "tests",
        "tests.*",
        "docs",
        "examples",
        "*.tests",
        "*.tests.*"
    ]),
    include_package_data=True,
    package_data={
        'PRGminer': [
            'data/*.h5',
            'models/*.h5',
            'config/*.json',
            'examples/*.fasta'
        ],
    },
    
    # Python requirements
    python_requires=PYTHON_REQUIRES,
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points
    entry_points={
        'console_scripts': [
            'PRGminer=PRGminer.__main__:main',
            'prgminer=PRGminer.__main__:main',
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
        "Environment :: Console",
    ],
    
    # Keywords
    keywords=[
        "bioinformatics",
        "deep-learning",
        "plant-resistance-genes",
        "gene-prediction",
        "machine-learning",
        "computational-biology",
        "r-genes",
        "plant-pathology",
        "genomics",
        "tensorflow",
    ],
    
    # Additional metadata
    platforms=["Linux", "Mac OS-X"],
    zip_safe=False,
)