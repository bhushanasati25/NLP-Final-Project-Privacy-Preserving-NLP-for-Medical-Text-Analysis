import mlflow
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path

class ExperimentTracker:
    """
    Tracks experiments, metrics, and artifacts during model training
    """
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "experiments",
        use_mlflow: bool = True
    ):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.use_mlflow = use_mlflow
        self.experiment_id = None
        self.run_id = None
        
        # Create experiment directory
        self.experiment_dir = os.path.join(
            base_dir,
            f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize MLflow if enabled
        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    def start_run(self, run_name: Optional[str] = None):
        """Start a new experiment run"""
        if self.use_mlflow:
            mlflow.start_run(run_name=run_name)
            self.run_id = mlflow.active_run().info.run_id
        
        # Create run directory
        self.run_dir = os.path.join(
            self.experiment_dir,
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            'training': {},
            'validation': {},
            'privacy': {},
            'performance': {}
        }
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters for the current run"""
        if self.use_mlflow:
            mlflow.log_params(params)
        
        # Save parameters locally
        params_file = os.path.join(self.run_dir, 'parameters.json')
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=4)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for the current run"""
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        
        # Update local metrics storage
        for category in self.metrics:
            if category in metrics:
                if step not in self.metrics[category]:
                    self.metrics[category][step] = {}
                self.metrics[category][step].update(metrics[category])
    
    def log_artifact(self, local_path: str):
        """Log an artifact file"""
        if self.use_mlflow:
            mlflow.log_artifact(local_path)
        
        # Copy artifact to run directory
        import shutil
        artifact_dir = os.path.join(self.run_dir, 'artifacts')
        os.makedirs(artifact_dir, exist_ok=True)
        shutil.copy2(local_path, artifact_dir)
    
    def log_model(self, model, artifact_path: str):
        """Log a model"""
        if self.use_mlflow:
            mlflow.pytorch.log_model(model, artifact_path)
        
        # Save model locally
        model_dir = os.path.join(self.run_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    
    def end_run(self):
        """End the current run"""
        if self.use_mlflow:
            mlflow.end_run()
        
        # Save final metrics
        metrics_file = os.path.join(self.run_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate a summary report for the run"""
        summary = {
            'experiment_name': self.experiment_name,
            'run_id': self.run_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics_summary': {}
        }
        
        # Calculate metric summaries
        for category, steps in self.metrics.items():
            if steps:
                latest_step = max(steps.keys())
                summary['metrics_summary'][category] = steps[latest_step]
        
        # Save summary
        summary_file = os.path.join(self.run_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
