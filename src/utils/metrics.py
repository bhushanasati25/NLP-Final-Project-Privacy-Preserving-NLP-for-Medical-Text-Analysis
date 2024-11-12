import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'loss': [],
            'accuracy': [],
            'privacy_budget': [],
            'client_performances': {}
        }
        self.start_time = datetime.now()
    
    def add_metric(self, metric_name, value, round_num=None):
        """Add metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def add_client_metric(self, client_id, metric_name, value):
        """Add client-specific metric"""
        if client_id not in self.metrics['client_performances']:
            self.metrics['client_performances'][client_id] = {}
        if metric_name not in self.metrics['client_performances'][client_id]:
            self.metrics['client_performances'][client_id][metric_name] = []
        self.metrics['client_performances'][client_id][metric_name].append(value)
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot loss
        axes[0].plot(self.metrics['loss'], 'r-', label='Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(self.metrics['accuracy'], 'b-', label='Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def save_metrics(self, save_path):
        """Save metrics to file"""
        metrics_dict = {
            'metrics': self.metrics,
            'training_time': str(datetime.now() - self.start_time),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)