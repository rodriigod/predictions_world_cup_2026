"""
Global configuration for the ML project
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Load config
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Paths
DATA_PATH = Path(CONFIG['paths']['raw_data'])
OUTPUT_PATH = Path(CONFIG['paths']['output_data'])
MODELS_PATH = Path(CONFIG['paths']['models_dir'])

# Random seed
RANDOM_SEED = CONFIG['project']['random_seed']

# Create directories if they don't exist
DATA_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Random Seed: {RANDOM_SEED}")
