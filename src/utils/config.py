import yaml
import os
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path='config/config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise Exception(f"Error loading config: {str(e)}")
    
    def get(self, key, default=None):
        """Get configuration value"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except:
            return default

    def save(self):
        """Save current configuration"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)