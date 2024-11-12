import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import pandas as pd
from datetime import datetime
import os

class VisualizationUtils:
    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_metrics(
        self,
        metrics: Dict,
        save_name: str = None
    ) -> str:
        """
        Plot training metrics
        """
        plt.figure(figsize=(12, 6))
        
        # Plot metrics
        for metric_name, values in metrics.items():
            if isinstance(values, list) and len(values) > 0:
                plt.plot(values, label=metric_name)
        
        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        if save_name is None:
            save_name = f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_privacy_analysis(
        self,
        privacy_metrics: Dict,
        save_name: str = None
    ) -> str:
        """
        Plot privacy analysis
        """
        plt.figure(figsize=(10, 6))
        
        # Plot privacy budget consumption
        plt.plot(
            privacy_metrics['budget_spent'],
            label='Privacy Budget Spent',
            color='red'
        )
        plt.axhline(
            y=privacy_metrics['budget_limit'],
            color='green',
            linestyle='--',
            label='Privacy Budget Limit'
        )
        
        plt.title('Privacy Budget Analysis')
        plt.xlabel('Round')
        plt.ylabel('Epsilon')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        if save_name is None:
            save_name = f'privacy_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_client_performance(
        self,
        client_metrics: Dict,
        save_name: str = None
    ) -> str:
        """
        Plot client performance comparison
        """
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        client_ids = list(client_metrics.keys())
        accuracies = [metrics['accuracy'][-1] for metrics in client_metrics.values()]
        
        # Create bar plot
        plt.bar(client_ids, accuracies)
        plt.title('Client Performance Comparison')
        plt.xlabel('Client ID')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # Save plot
        if save_name is None:
            save_name = f'client_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        
        return save_path