import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
import yaml

class ValidationUtils:
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """
        Validate configuration parameters
        """
        required_keys = [
            'data',
            'preprocessing',
            'model',
            'federated',
            'deployment'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config section: {key}")
        
        # Validate specific parameters
        try:
            assert 0 < config['data']['train_test_split'] < 1
            assert config['preprocessing']['max_features'] > 0
            assert config['model']['input_dim'] > 0
            assert config['federated']['num_clients'] > 0
            assert config['federated']['privacy']['epsilon'] > 0
        except AssertionError as e:
            raise ValueError(f"Invalid configuration value: {str(e)}")
        
        return True
    
    @staticmethod
    def validate_data_format(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate input data format
        """
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Found null values in columns: {null_counts[null_counts > 0]}")
        
        return True
    
    @staticmethod
    def validate_model_input(
        features: np.ndarray,
        labels: np.ndarray,
        expected_shape: tuple = None
    ) -> bool:
        """
        Validate model input format
        """
        # Check dimensions
        if len(features.shape) != 2:
            raise ValueError(f"Features must be 2D array, got shape {features.shape}")
        
        if expected_shape and features.shape[1] != expected_shape[1]:
            raise ValueError(
                f"Features must have shape {expected_shape}, "
                f"got {features.shape}"
            )
        
        # Check labels
        if len(labels.shape) != 1:
            raise ValueError(f"Labels must be 1D array, got shape {labels.shape}")
        
        if len(features) != len(labels):
            raise ValueError(
                f"Number of features ({len(features)}) must match "
                f"number of labels ({len(labels)})"
            )
        
        return True