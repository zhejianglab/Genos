#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management module
"""

import os
import torch
import yaml
from copy import deepcopy
import random
import json
torch.use_deterministic_algorithms(True)

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For numpy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """Read-only configuration manager"""

    def __init__(self, args):
        """
        Initialize the configuration

        Args:
            config_path (str): Path to the configuration file. Supports .yaml and .yml files.
        """

        self.args = args
        self.config_path = self.args.config
        self._config = self._load_config()
        self._datasets_info = self._load_dataset_feature()
        self._model_config = self._load_model_config()
        self._process_config()
        set_seed(self._config['seed'])

    def _load_config(self):
        """Load the configuration file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                # Select the parser based on the file extension
                if self.config_path.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    return json.load(f)
                else:
                    # Try YAML format by default
                    try:
                        return yaml.safe_load(f)
                    except yaml.YAMLError:
                        raise ValueError(
                            f"Could not parse configuration file {self.config_path}. Provide valid YAML or JSON.")
        else:
            raise FileNotFoundError(f"Configuration file {self.config_path} does not exist")

    def _load_dataset_feature(self):
        """Load the dataset feature configuration file"""
        # Get the dataset feature configuration path from the main configuration
        dataset_feature_path = self._config.get(
            'datasets_feature_path', 'benchmarks/datasets_info.yaml')

        if os.path.exists(dataset_feature_path):
            try:
                with open(dataset_feature_path, 'r', encoding='utf-8') as f:
                    dataset_feature = yaml.safe_load(f)
                    print(f"✓ Loaded dataset feature configuration: {dataset_feature_path}")
                    return dataset_feature
            except yaml.YAMLError as e:
                raise ValueError(
                    f"⚠️ Warning: Could not parse dataset feature configuration {dataset_feature_path}: {e}")
        else:
            raise FileNotFoundError(
                f"⚠️ Warning: Dataset feature configuration {dataset_feature_path} does not exist")

    def _load_model_config(self):
        """Load the model configuration file"""
        # Get the model configuration path from the main configuration
        model_path = self._config.get('model_path', None)
        model_config_path = f"{model_path}/config.json" if model_path is not None else None
        if model_config_path is not None and os.path.exists(model_config_path):
            with open(model_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            assert "num_layers" in self._config, "Model configuration file is missing and num_layers was not provided"
            return {
                'num_hidden_layers': self._config['num_layers']
            }

    def _process_config(self):
        """Process configuration parameters"""
        # Validate required parameters
        if 'model_path' not in self._config or not self._config['model_path']:
            assert 'model_name' in self._config, "No model path or model_name was provided in config"
        else:
            # Remove the trailing slash from the model path and derive the model name
            model_path = self._config['model_path'].rstrip('/')
            self._config['model_name'] = model_path.split("/")[-1]

        # Process the GPU list
        if 'gpu_list' in self._config and self._config['gpu_list']:
            gpu_list = self._config['gpu_list']
            # Split gpu_list by commas if it is a string
            if isinstance(gpu_list, str):
                gpu_list = gpu_list.split(",")
            # Ensure that gpu_list is a list
            if isinstance(gpu_list, list):
                self._config['gpu_list'] = [
                    int(str(gpu).strip()) for gpu in gpu_list]
            else:
                raise ValueError("gpu_list must be a string or a list")
        else:
            device_count = torch.cuda.device_count()
            self._config['gpu_list'] = [i for i in range(device_count)]

        if 'layer_to_eval' not in self._config:
            self._config['layer_to_eval'] = list(
                range(self._model_config['num_hidden_layers']+1))
        else:
            self._config['layer_to_eval'] = [
                int(layer) for layer in self._config['layer_to_eval']]

        if 'batch_size' not in self._config:
            self._config['batch_size'] = None

        if 'eval_datasets' not in self._config:
            self._config['eval_datasets'] = self._datasets_info['default_dataset']
        if 'wandb_online' not in self._config or self._config['wandb_online']:
            pass
        else:
            os.environ["WANDB_MODE"] = "offline"

        # Print configuration details
        self._print_config()

    def _print_config(self):
        """Print configuration details"""
        print("=== Configuration ===")
        print(f"Configuration file: {self.config_path}")
        for key, value in self._config.items():
            print(f"{key}: {value}")
        print("================")

    @property
    def all_datasets_feature(self):
        """Read-only dataset feature configuration"""
        return deepcopy(self._datasets_info)

    @property
    def dataset_info(self):
        """Read-only evaluation dataset configuration"""
        return deepcopy(self._datasets_info["dataset_feature"])

    @property
    def model_config(self):
        """Read-only model configuration"""
        return deepcopy(self._model_config)

    def get(self, key, default=None):
        """
        Get a configuration value

        Args:
            key (str): Configuration key
            default: Default value

        Returns:
            Configuration value or default value
        """
        value = self._config.get(key, default)
        # Return a deep copy to prevent external modification
        if isinstance(value, (list, dict)):
            return deepcopy(value)
        return value

    def __getitem__(self, key):
        """Support dictionary-style access and return a read-only copy"""
        value = self._config[key]
        # Return a deep copy to prevent external modification
        if isinstance(value, (list, dict)):
            return deepcopy(value)
        return value

    def __contains__(self, key):
        """Support the in operator"""
        return key in self._config

    def __iter__(self):
        """Support iteration and return a copy of the keys"""
        return iter(list(self._config.keys()))

    def keys(self):
        """Return a copy of all configuration keys"""
        return list(self._config.keys())

    def values(self):
        """Return copies of all configuration values"""
        return [deepcopy(v) if isinstance(v, (list, dict)) else v for v in self._config.values()]

    def items(self):
        """Return copies of all configuration items"""
        return [(k, deepcopy(v) if isinstance(v, (list, dict)) else v) for k, v in self._config.items()]

    def copy(self):
        """Copy the configuration and return a deep copy"""
        return deepcopy(self._config)

    def to_dict(self):
        """Convert the configuration to a dictionary and return a deep copy"""
        return deepcopy(self._config)

    def __len__(self):
        """Return the number of configuration items"""
        return len(self._config)

    def __repr__(self):
        """String representation"""
        return f"Config({self.config_path})"

    def __str__(self):
        """String representation"""
        return f"Config({self.config_path})"

    # Disable mutation operations
    def __setitem__(self, key, value):
        """Configuration items cannot be set"""
        raise TypeError("Config is read-only; configuration items cannot be modified")

    def __delitem__(self, key):
        """Configuration items cannot be deleted"""
        raise TypeError("Config is read-only; configuration items cannot be deleted")

    def update(self, other):
        """Configuration cannot be updated"""
        raise TypeError("Config is read-only; it cannot be updated")

    def clear(self):
        """Configuration cannot be cleared"""
        raise TypeError("Config is read-only; it cannot be cleared")

    def pop(self, key, default=None):
        """Configuration items cannot be removed"""
        raise TypeError("Config is read-only; configuration items cannot be removed")

    def popitem(self):
        """Configuration items cannot be removed"""
        raise TypeError("Config is read-only; configuration items cannot be removed")

    def setdefault(self, key, default=None):
        """Configuration defaults cannot be set"""
        raise TypeError("Config is read-only; defaults cannot be set")
