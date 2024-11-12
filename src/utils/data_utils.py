import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from typing import Tuple, List, Dict
import json
import os

class DataUtils:
    @staticmethod
    def load_and_split_data(
        file_path: str,
        text_column: str,
        label_column: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and split data into train, validation, and test sets
        """
        # Load data
        df = pd.read_csv(file_path)
        
        # First split: train + val, test
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[label_column] if label_column in df else None
        )
        
        # Second split: train, val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val[label_column] if label_column in train_val else None
        )
        
        return train, val, test
    
    @staticmethod
    def create_data_batches(
        data: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ) -> List[np.ndarray]:
        """
        Create batches from data
        """
        if shuffle:
            indices = np.random.permutation(len(data))
            data = data[indices]
        
        return np.array_split(data, max(1, len(data) // batch_size))
    
    @staticmethod
    def save_processed_data(
        data: pd.DataFrame,
        save_dir: str,
        prefix: str = "processed"
    ) -> str:
        """
        Save processed data with timestamp
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.csv"
        filepath = os.path.join(save_dir, filename)
        
        data.to_csv(filepath, index=False)
        return filepath

    @staticmethod
    def check_data_balance(labels: np.ndarray) -> Dict:
        """
        Check class balance in labels
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        balance_info = {
            'class_distribution': dict(zip(unique_labels, counts)),
            'total_samples': len(labels),
            'imbalance_ratio': max(counts) / min(counts)
        }
        return balance_info