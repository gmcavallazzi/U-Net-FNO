import os
import yaml
import random
import numpy as np
import torch
import logging
from datetime import datetime

def setup_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed}")

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_tensorboard_dir(config):
    """Create a unique, timestamped TensorBoard log directory"""
    if not config['output']['log_dir']:
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(config['output']['log_dir'], timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration to the log directory for reference
    config_path = os.path.join(log_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"View logs with: tensorboard --logdir={config['output']['log_dir']}")
    
    return log_dir

def count_parameters(model):
    """Count the number of trainable parameters in a PyTorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device_info():
    """Get information about available hardware"""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'device_capability': None
    }
    
    if device_info['cuda_available'] and device_info['device_count'] > 0:
        device_info['current_device'] = torch.cuda.current_device()
        device_info['device_name'] = torch.cuda.get_device_name(device_info['current_device'])
        
        # Get device capability
        prop = torch.cuda.get_device_properties(device_info['current_device'])
        device_info['device_capability'] = f"{prop.major}.{prop.minor}"
        device_info['total_memory'] = prop.total_memory / 1024**3  # Convert to GB
        
    return device_info

def print_device_info():
    """Print information about the available hardware"""
    info = get_device_info()
    
    if info['cuda_available']:
        print(f"CUDA available: Yes")
        print(f"Number of GPUs: {info['device_count']}")
        
        # Print information for each GPU
        for i in range(info['device_count']):
            prop = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {prop.name}")
            print(f"  Compute capability: {prop.major}.{prop.minor}")
            print(f"  Total memory: {prop.total_memory / 1024**3:.2f} GB")
            print(f"  Multi-processor count: {prop.multi_processor_count}")
        
        print(f"\nPyTorch built with CUDA: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("CUDA not available. Using CPU for training.")

def create_output_dirs(config):
    """Create output directories for results

    Args:
        config: Configuration dictionary

    Returns:
        tuple: (output_dir, log_dir, timestamp)
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create main output directory
    output_dir = os.path.join(config['output']['output_dir'], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log directory if configured
    log_dir = None
    if config['output']['log_dir']:
        log_dir = os.path.join(config['output']['log_dir'], timestamp)
        os.makedirs(log_dir, exist_ok=True)
    
    return output_dir, log_dir, timestamp

if __name__ == "__main__":
    # Set random seeds
    setup_random_seeds(42)
    
    # Print device information
    print_device_info()
    
    # Create output directories
    dirs = create_output_dirs()
    print(f"Created output directories: {dirs}")

def setup_logging(console_only=True, level=logging.INFO, log_file=None):
    """
    Configure Python logging with options for Slurm compatibility
    
    Args:
        console_only (bool): If True, only log to console (recommended for Slurm)
        level: Logging level (default: INFO)
        log_file (str): Optional path to log file (ignored if console_only=True)
    """
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create handlers list
    handlers = []
    
    # Always add console handler for Slurm output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # Optionally add file handler
    if not console_only and log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers
    )
    
    # Log setup confirmation
    if console_only:
        logging.info("Logging configured for console output only (Slurm compatible)")
    else:
        logging.info(f"Logging configured for console and file: {log_file}")