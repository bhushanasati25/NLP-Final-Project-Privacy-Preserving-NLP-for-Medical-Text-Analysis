import optuna
from typing import Dict, Any, Callable, Optional
import numpy as np
import torch
import json
import os
from datetime import datetime

class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna
    """
    def __init__(
        self,
        study_name: str,
        objective_function: Callable,
        direction: str = "maximize",
        storage: Optional[str] = None
    ):
        self.study_name = study_name
        self.objective_function = objective_function
        
        # Create Optuna study
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=True
        )
        
        # Initialize results storage
        self.results_dir = os.path.join(
            "optimization_results",
            f"{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
    
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the current trial
        """
        params = {
            # Model parameters
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            
            # Training parameters
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'num_epochs': trial.suggest_int('num_epochs', 5, 20),
            
            # Federated learning parameters
            'num_clients': trial.suggest_int('num_clients', 3, 10),
            'local_epochs': trial.suggest_int('local_epochs', 1, 5),
            
            # Privacy parameters
            'epsilon': trial.suggest_float('epsilon', 0.1, 2.0),
            'noise_multiplier': trial.suggest_float('noise_multiplier', 0.5, 2.0)
        }
        
        return params
    
    def optimize(self, n_trials: int = 100):
        """
        Run hyperparameter optimization
        """
        self.study.optimize(
            func=self._objective_wrapper,
            n_trials=n_trials
        )
        
        # Save optimization results
        self._save_results()
        
        return self.study.best_params, self.study.best_value
    
    def _objective_wrapper(self, trial: optuna.Trial) -> float:
        """
        Wrapper for the objective function
        """
        params = self.suggest_params(trial)
        
        try:
            value = self.objective_function(trial, params)
            return value
        except Exception as e:
            print(f"Trial failed: {str(e)}")
            return float('-inf')
    
    def _save_results(self):
        """
        Save optimization results
        """
        results = {
            'study_name': self.study_name,
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save results
        results_file = os.path.join(self.results_dir, 'optimization_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Generate optimization plots
        self._generate_optimization_plots()
    
    def _generate_optimization_plots(self):
        """
        Generate optimization visualization plots
        """
        import matplotlib.pyplot as plt
        import optuna.visualization as vis
        
        # Optimization history
        plt.figure(figsize=(10, 6))
        vis.plot_optimization_history(self.study)
        plt.savefig(os.path.join(self.results_dir, 'optimization_history.png'))
        plt.close()
        
        # Parameter importance
        plt.figure(figsize=(12, 6))
        vis.plot_param_importances(self.study)
        plt.savefig(os.path.join(self.results_dir, 'parameter_importance.png'))
        plt.close()
        
        # Parallel coordinate plot
        plt.figure(figsize=(15, 8))
        vis.plot_parallel_coordinate(self.study)
        plt.savefig(os.path.join(self.results_dir, 'parallel_coordinate.png'))
        plt.close()

def example_usage():
    """
    Example usage of HyperparameterOptimizer
    """
    def objective(trial, params):
        # Your model training and evaluation code here
        # Return the metric to optimize (e.g., validation accuracy)
        return accuracy
    
    optimizer = HyperparameterOptimizer(
        study_name="federated_nlp_optimization",
        objective_function=objective
    )
    
    best_params, best_value = optimizer.optimize(n_trials=50)
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")