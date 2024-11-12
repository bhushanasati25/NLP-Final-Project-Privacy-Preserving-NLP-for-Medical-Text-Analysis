"""
Model Implementation Module
--------------------------

Contains neural network models and related components for text classification.
"""

from .text_classifier import SimpleTextClassifier
from .layers import AttentionLayer, PrivacyLayer
from .loss import PrivacyAwareLoss

__all__ = [
    'SimpleTextClassifier',
    'AttentionLayer',
    'PrivacyLayer',
    'PrivacyAwareLoss'
]

# Module metadata
__module_name__ = 'models'
__version__ = '1.0.0'
__last_update__ = '2024-11-11'
__maintainer__ = 'Bhushan Asati'

# Model configurations
MODEL_CONFIGS = {
    'small': {
        'hidden_dim': 128,
        'num_layers': 2
    },
    'medium': {
        'hidden_dim': 256,
        'num_layers': 3
    },
    'large': {
        'hidden_dim': 512,
        'num_layers': 4
    }
}