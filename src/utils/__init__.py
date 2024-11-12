"""
Utility Functions Module
-----------------------

Collection of utility functions and helper classes for the project.
"""

from .config import ConfigLoader
from .data_utils import DataUtils
from .model_utils import ModelUtils
from .privacy_utils import PrivacyUtils
from .validation_utils import ValidationUtils
from .visualization_utils import VisualizationUtils
from .logger import Logger
from .metrics import MetricsTracker
from .experiment import ExperimentTracker
from .optimization import HyperparameterOptimizer

__all__ = [
    'ConfigLoader',
    'DataUtils',
    'ModelUtils',
    'PrivacyUtils',
    'ValidationUtils',
    'VisualizationUtils',
    'Logger',
    'MetricsTracker',
    'ExperimentTracker',
    'HyperparameterOptimizer'
]

# Module metadata
__module_name__ = 'utils'
__version__ = '1.0.0'
__last_update__ = '2024-11-11'
__maintainer__ = 'Bhushan Asati'

# Utility configurations
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': True
}

VISUALIZATION_CONFIG = {
    'style': 'seaborn',
    'figure_size': (10, 6),
    'save_format': 'png',
    'dpi': 300
}

def setup_environment():
    """Setup development environment."""
    import logging
    logging.basicConfig(**LOGGING_CONFIG)
    
    try:
        import matplotlib.pyplot as plt
        plt.style.use(VISUALIZATION_CONFIG['style'])
    except ImportError:
        pass

# Run setup when importing utils
setup_environment()
