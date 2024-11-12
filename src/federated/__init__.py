"""
Federated Learning Module
------------------------

Implements federated learning components including client and server
implementations with privacy-preserving mechanisms.
"""

from .client import FederatedClient
from .server import FederatedServer
from .privacy import PrivacyMechanism
from .aggregation import FederatedAggregator

__all__ = [
    'FederatedClient',
    'FederatedServer',
    'PrivacyMechanism',
    'FederatedAggregator'
]

# Module metadata
__module_name__ = 'federated'
__version__ = '1.0.0'
__last_update__ = '2024-11-11'
__maintainer__ = 'Bhushan Asati'

# Module configuration
DEFAULT_CONFIG = {
    'num_clients': 5,
    'num_rounds': 10,
    'local_epochs': 2,
    'batch_size': 32,
    'privacy': {
        'epsilon': 1.0,
        'delta': 1e-5
    }
}
