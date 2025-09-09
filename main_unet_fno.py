#!/usr/bin/env python3
"""
U-Net-FNO Training Script with Peak Preservation

Key features:
- Increased model capacity (96 channels, 16 modes)
- Attention gates in decoder
- Multi-scale loss functions
- Peak preservation techniques
- Training strategies for high-velocity regions
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
import yaml
from datetime import datetime

# Import data utilities
from data_loader import (
    find_matching_quadruplets, 
    create_dataloaders_quadruplets, 
    load_validation_data_numpy_quadruplets
)
from utils import setup_random_seeds, setup_logging, load_config

# Import components
from unet_fno_model import UNetFNO
from train_unet_fno import UNetFNOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net-FNO with Peak Preservation')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')
    
    # Quick overrides
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--no_attention', action='store_true')
    parser.add_argument('--noise_std', type=float, default=None)
    
    return parser.parse_args()

def merge_config_args(config, args):
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.no_gpu:
        config['hardware']['use_gpu'] = False
    if args.no_attention:
        config['model']['use_attention'] = False
    if args.noise_std is not None:
        config['training']['noise_std'] = args.noise_std
    
    return config

def setup_directories(output_dir, experiment_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        run_name = f"{experiment_name}_{timestamp}"
    else:
        run_name = f"run_{timestamp}"
    
    run_dir = os.path.join(output_dir, run_name)
    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return run_dir, models_dir, logs_dir


def main():
    args = parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        logging.error(f"Config file not found: {args.config}")
        return 1
    
    config = load_config(args.config)
    config = merge_config_args(config, args)
    
    # Setup
    log_level = getattr(logging, config.get('monitoring', {}).get('log_level', 'INFO').upper(), logging.INFO)
    setup_logging(console_only=True, level=log_level)
    setup_random_seeds(config.get('reproducibility', {}).get('seed', 42))
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() and config.get('hardware', {}).get('use_gpu', True) else 'cpu'
    logging.info(f"Using device: {device}")
    
    if device == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create directories
    output_dir = config.get('output', {}).get('output_dir', './results_unet_fno')
    run_dir, models_dir, logs_dir = setup_directories(output_dir, args.experiment_name)
    logging.info(f"Output directory: {run_dir}")
    
    # Save config
    config_save_path = os.path.join(run_dir, 'config_used.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Load data
    data_config = config['data']
    logging.info("Loading data...")
    train_quadruplets, val_quadruplets = find_matching_quadruplets(
        data_config['data_dir'], 
        data_config['validation_dir'],
        shuffle=True,
        max_train=data_config.get('max_train', None),
        max_val=data_config.get('max_val', None)
    )
    
    if not train_quadruplets:
        logging.error("No training data found!")
        return 1
    
    logging.info(f"Found {len(train_quadruplets)} training samples")
    logging.info(f"Found {len(val_quadruplets)} validation samples")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders_quadruplets(
        train_quadruplets,
        batch_size=data_config['batch_size'],
        shape=tuple(data_config.get('shape', [64, 64])),
        num_workers=data_config.get('num_workers', 4),
        pin_memory=(device == 'cuda' and data_config.get('pin_memory', True)),
        val_quadruplets=val_quadruplets
    )
    
    # Load validation data
    val_data = None
    if val_quadruplets:
        val_input, val_velocity = load_validation_data_numpy_quadruplets(
            val_quadruplets,
            max_samples=min(200, len(val_quadruplets)),
            shape=tuple(data_config.get('shape', [64, 64]))
        )
        if val_input is not None:
            val_data = (val_input, val_velocity)
            logging.info(f"Loaded validation data: {val_input.shape}")
    
    # Create model with attention and increased capacity
    model_config = config['model']
    model = UNetFNO(
        in_channels=model_config.get('in_channels', 2),
        out_channels=model_config.get('out_channels', 2),
        base_channels=model_config.get('base_channels', 96),
        depth=model_config.get('depth', 3),
        modes=model_config.get('modes', 16),
        enforce_incompressible=model_config.get('enforce_incompressible', True),
        use_attention=model_config.get('use_attention', True)
    )
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model created with {param_count:,} parameters")
    
    # Physics weights with peak preservation
    training_config = config['training']
    physics_weights = training_config.get('physics_weights', {
        'mse': 0.7,
        'weighted_mse': 0.3,
        'huber': 0.2,
        'magnitude_l1': 0.1,
        'divergence': 0.05,
        'boundary': 0.001,
        'frequency': 0.05
    })
    
    # Create trainer
    trainer = UNetFNOTrainer(
        model=model,
        device=device,
        learning_rate=training_config.get('learning_rate', 2e-4),
        physics_weights=physics_weights,
        use_mixed_precision=training_config.get('use_mixed_precision', False),
        noise_std=training_config.get('noise_std', 0.02)
    )
    
    # Resume if requested
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = trainer.load_model(args.resume)
        logging.info(f"Resumed from epoch {start_epoch}")
    elif args.resume:
        logging.warning(f"Resume checkpoint not found: {args.resume}")
    
    # Training config
    epochs = training_config.get('epochs', 150)
    save_interval = training_config.get('save_interval', 15)
    val_interval = training_config.get('validation_interval', 5)
    
    logging.info("\n" + "="*70)
    logging.info("U-NET-FNO TRAINING WITH PEAK PRESERVATION")
    logging.info("="*70)
    logging.info(f"  Epochs: {epochs}")
    logging.info(f"  Batch size: {data_config['batch_size']}")
    logging.info(f"  Learning rate: {training_config['learning_rate']}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Model parameters: {param_count:,}")
    logging.info(f"  Training noise std: {training_config.get('noise_std', 0.02)}")
    logging.info("\nModel Features:")
    logging.info(f"  Base channels: {model_config.get('base_channels', 96)}")
    logging.info(f"  Fourier modes: {model_config.get('modes', 16)}")
    logging.info(f"  Attention gates: {model_config.get('use_attention', True)}")
    logging.info(f"  Multi-layer output head: Yes")
    logging.info(f"  Softsign activation: Yes (peak preservation)")
    logging.info("\nLoss Functions:")
    for key, value in physics_weights.items():
        logging.info(f"  {key}: {value}")
    logging.info("\nKey Features:")
    logging.info("  - Multi-scale loss computation")
    logging.info("  - Weighted MSE for high-velocity regions")
    logging.info("  - Huber loss for peak preservation")
    logging.info("  - L1 magnitude loss")
    logging.info("  - Frequency domain consistency")
    logging.info("  - Training noise for robustness")
    logging.info("  - Attention-gated skip connections")
    logging.info("  - Warm restart learning rate schedule")
    logging.info("="*70 + "\n")
    
    # Start training
    logging.info("Starting U-Net-FNO training with peak preservation...")
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_data=val_data,
            epochs=epochs,
            save_dir=models_dir,
            log_dir=logs_dir,
            save_interval=save_interval,
            validation_interval=val_interval
        )
        
        # Training summary
        logging.info("\n" + "="*70)
        logging.info("TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("="*70)
        
        if history['val_loss']:
            final_val_loss = history['val_loss'][-1]
            best_val_loss = min(history['val_loss'])
            best_epoch = history['val_loss'].index(best_val_loss) * val_interval + 1
            logging.info(f"Final validation loss: {final_val_loss:.6f}")
            logging.info(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
        
        final_train_loss = history['train_loss'][-1]
        logging.info(f"Final training loss: {final_train_loss:.6f}")
        
        logging.info(f"\nResults saved to: {run_dir}")
        logging.info(f"TensorBoard: tensorboard --logdir={logs_dir}")
        
        # Features summary
        logging.info("\nFeatures applied:")
        logging.info("  - Increased model capacity: 96 channels, 16 modes")
        logging.info("  - Attention gates for better feature selection")
        logging.info("  - Multi-scale loss for better convergence")
        logging.info("  - Peak preservation with Huber + L1 magnitude loss")
        logging.info("  - High-velocity region emphasis")
        logging.info("  - Frequency domain consistency")
        logging.info("  - Training noise for robustness")
        logging.info("  - Visualizations with peak analysis")
        
        return 0
        
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)