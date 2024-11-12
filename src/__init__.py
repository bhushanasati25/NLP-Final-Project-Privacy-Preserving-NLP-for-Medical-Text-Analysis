"""
Privacy-Preserving NLP through Federated Learning
-----------------------------------------------

A comprehensive implementation of privacy-preserving Natural Language Processing
using Federated Learning for medical text analysis.
"""

from . import preprocessing
from . import federated
from . import models
from . import deployment
from . import evaluation
from . import utils

__version__ = '1.0.0'
__author__ = 'Bhushan Asati'
__email__ = 'bhushanasati25@gmail.com'
__license__ = 'MIT'
__copyright__ = 'Copyright 2024'
__status__ = 'Development'
__date__ = '2024-11-11'

# Version information
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'final'
}

# Package metadata
PACKAGE_INFO = {
    'name': 'privacy_preserving_nlp',
    'description': 'Privacy-Preserving NLP through Federated Learning',
    'keywords': [
        'nlp',
        'federated-learning',
        'privacy',
        'machine-learning',
        'medical-text-analysis'
    ],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
}

# Dependencies
REQUIREMENTS = {
    'required': [
        'torch>=1.9.0',
        'numpy>=1.19.5',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'flask>=2.0.1'
    ],
    'optional': {
        'visualization': [
            'matplotlib>=3.4.3',
            'seaborn>=0.11.2'
        ],
        'development': [
            'pytest>=6.2.5',
            'black>=21.7b0',
            'flake8>=3.9.2'
        ]
    }
}

def get_version():
    """Return the current version."""
    return __version__

def get_requirements():
    """Return package requirements."""
    return REQUIREMENTS
