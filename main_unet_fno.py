#!/usr/bin/env python3
"""
U-Net-FNO Training Script for Navier-Stokes Velocity Prediction

This script trains a hybrid U-Net-FNO model to predict velocity fields (u, v)
from pressure and wall shear stress fields for fluid dynamics applications.
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
import yaml
from datetime import datetime

# Import existing data utilities
from data_loader import (
    find_matching_quadruplets, 
    create_dataloaders_quadruplets, 
    load_validation_data_numpy_quadruplets
)
from utils import setup_random_seeds, setup_logging, load_config

# Import new U-Net-FNO components
from unet_fno_model import UNetFNO
from train_unet_fno import UNetFNOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net-FNO for velocity prediction')
    parser.add_argument('--config', type=str, default='config_unet_fno.yaml',
                       help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (path to .pt file)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for output directory')
    
    # Allow command line overrides of config file
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Training data directory (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU usage')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (overrides config)')
    
    return parser.parse_args()

def merge_config_args(config, args):
    """Merge command line arguments with config file, with args taking priority"""
    # Command line arguments override config file
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.mixed_precision:
        config['training']['use_mixed_precision'] = True
    if args.no_gpu:
        config['hardware']['use_gpu'] = False
    
    return config


def setup_directories(output_dir, experiment_name=None):
    """Create output directories with optional experiment name"""
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
    
    # Load configuration file
    if not os.path.exists(args.config):
        logging.error(f"Config file not found: {args.config}")
        return 1
    
    config = load_config(args.config)
    
    # Merge command line arguments with config (args take priority)
    config = merge_config_args(config, args)
    
    # Setup logging level
    log_level = getattr(logging, config.get('monitoring', {}).get('log_level', 'INFO').upper(), logging.INFO)
    setup_logging(console_only=True, level=log_level)
    setup_random_seeds(config.get('reproducibility', {}).get('seed', 42))
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() and config.get('hardware', {}).get('use_gpu', True) else 'cpu'
    logging.info(f"Using device: {device}")
    
    if device == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directories
    output_dir = config.get('output', {}).get('output_dir', './results_unet_fno')
    run_dir, models_dir, logs_dir = setup_directories(output_dir, args.experiment_name)
    logging.info(f"Output directory: {run_dir}")
    
    # Save configuration for reproducibility
    config_save_path = os.path.join(run_dir, 'config_used.yaml')
    with open(config_save_path, 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    logging.info(f"Configuration saved to: {config_save_path}")
    
    # Load data using config parameters
    data_config = config['data']
    logging.info("Loading training and validation data...")
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
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders_quadruplets(
        train_quadruplets,
        batch_size=data_config['batch_size'],
        shape=tuple(data_config.get('shape', [64, 64])),
        num_workers=data_config.get('num_workers', 4),
        pin_memory=(device == 'cuda' and data_config.get('pin_memory', True)),
        val_quadruplets=val_quadruplets
    )
    
    # Load validation data for evaluation
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
    
    # Create model using config parameters
    model_config = config['model']
    model = UNetFNO(
        in_channels=model_config.get('in_channels', 2),
        out_channels=model_config.get('out_channels', 2),
        base_channels=model_config.get('base_channels', 64),
        depth=model_config.get('depth', 4),
        modes=model_config.get('modes', 16),
        enforce_incompressible=model_config.get('enforce_incompressible', True)
    )
    
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Physics loss weights from config
    training_config = config['training']
    physics_weights = training_config.get('physics_weights', {
        'mse': 1.0,
        'divergence': 0.1,
        'boundary': 0.05,
        'spectral': 0.1
    })
    
    # Create trainer
    trainer = UNetFNOTrainer(
        model=model,
        device=device,
        learning_rate=training_config.get('learning_rate', 1e-4),
        physics_weights=physics_weights,
        use_mixed_precision=training_config.get('use_mixed_precision', False)
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = trainer.load_model(args.resume)
        logging.info(f"Resumed from epoch {start_epoch}")
        logging.info(f"Resuming training from checkpoint: {args.resume}")
    elif args.resume:
        logging.warning(f"Resume checkpoint not found: {args.resume}")
        logging.info("Starting training from scratch...")
    
    # Training configuration summary
    epochs = training_config.get('epochs', 200)
    save_interval = training_config.get('save_interval', 20)
    val_interval = training_config.get('validation_interval', 5)
    
    logging.info("\n" + "="*60)
    logging.info("TRAINING CONFIGURATION")
    logging.info("="*60)
    logging.info(f"  Epochs: {epochs} (starting from {start_epoch})")
    logging.info(f"  Batch size: {data_config['batch_size']}")
    logging.info(f"  Learning rate: {training_config['learning_rate']}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Mixed precision: {training_config.get('use_mixed_precision', False)}")
    logging.info(f"  Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logging.info("\nModel Architecture:")
    logging.info(f"  Fourier modes: {model_config.get('modes', 16)}")
    logging.info(f"  Base channels: {model_config.get('base_channels', 64)}")
    logging.info(f"  U-Net depth: {model_config.get('depth', 4)}")
    logging.info(f"  Incompressible flow: {model_config.get('enforce_incompressible', True)}")
    logging.info("\nPhysics Loss Weights:")
    for key, value in physics_weights.items():
        logging.info(f"  {key}: {value}")
    logging.info(f"\nOutput directories:")
    logging.info(f"  Models: {models_dir}")
    logging.info(f"  Logs: {logs_dir}")
    logging.info("="*60 + "\n")
    
    # Start training
    logging.info("Starting U-Net-FNO training...")
    
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
        
        # Create final checkpoint summary
        trainer.create_checkpoint_summary(models_dir)
        
        # Training summary
        logging.info("\n" + "="*60)
        logging.info("TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("="*60)
        
        if history['val_loss']:
            final_val_loss = history['val_loss'][-1]
            best_val_loss = min(history['val_loss'])
            best_epoch = history['val_loss'].index(best_val_loss) * val_interval + 1
            logging.info(f"Final validation loss: {final_val_loss:.6f}")
            logging.info(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
        
        final_train_loss = history['train_loss'][-1]
        logging.info(f"Final training loss: {final_train_loss:.6f}")
        logging.info(f"Total epochs completed: {len(history['train_loss'])}")
        
        logging.info(f"\nResults saved to: {run_dir}")
        logging.info(f"  Models: {models_dir}")
        logging.info(f"  Logs: {logs_dir}")
        logging.info(f"  TensorBoard: tensorboard --logdir={logs_dir}")
        
        logging.info("\nAvailable model checkpoints:")
        logging.info("  - best_model.pt (best validation loss)")
        logging.info("  - final_model.pt (final epoch)")
        logging.info(f"  - checkpoint_epoch_*.pt (every {save_interval} epochs)")
        logging.info("  - initial_checkpoint.pt (starting point)")
        
        return 0
        
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)