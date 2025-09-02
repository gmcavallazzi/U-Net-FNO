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
from datetime import datetime

# Import existing data utilities
from data_loader import (
    find_matching_quadruplets, 
    create_dataloaders_quadruplets, 
    load_validation_data_numpy_quadruplets
)
from utils import setup_random_seeds, setup_logging

# Import new U-Net-FNO components
from unet_fno_model import UNetFNO
from train_unet_fno import UNetFNOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net-FNO for velocity prediction')
    parser.add_argument('--data_dir', type=str, default='./data_wss_uv',
                       help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='./validation_wss_uv',
                       help='Validation data directory')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_train', type=int, default=6080,
                       help='Maximum training samples')
    parser.add_argument('--max_val', type=int, default=1520,
                       help='Maximum validation samples')
    parser.add_argument('--modes', type=int, default=16,
                       help='Number of Fourier modes in FNO')
    parser.add_argument('--channels', type=int, default=64,
                       help='Base number of channels')
    parser.add_argument('--depth', type=int, default=4,
                       help='U-Net depth')
    parser.add_argument('--output_dir', type=str, default='./results_unet_fno',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (path to .pt file)')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU usage')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--incompressible', action='store_true', default=True,
                       help='Enforce incompressible flow constraint')
    
    # Checkpointing and logging options
    parser.add_argument('--save_interval', type=int, default=20,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--val_interval', type=int, default=5,
                       help='Run validation every N epochs')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for output directory')
    
    return parser.parse_args()


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
    
    # Setup logging level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(console_only=True, level=log_level)
    setup_random_seeds(42)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu'
    logging.info(f"Using device: {device}")
    
    if device == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directories
    run_dir, models_dir, logs_dir = setup_directories(args.output_dir, args.experiment_name)
    logging.info(f"Output directory: {run_dir}")
    
    # Save configuration for reproducibility
    import json
    config_dict = vars(args)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    logging.info("Configuration saved to config.json")
    
    # Load data
    logging.info("Loading training and validation data...")
    train_quadruplets, val_quadruplets = find_matching_quadruplets(
        args.data_dir, args.val_dir,
        shuffle=True,
        max_train=args.max_train,
        max_val=args.max_val
    )
    
    if not train_quadruplets:
        logging.error("No training data found!")
        return 1
    
    logging.info(f"Found {len(train_quadruplets)} training samples")
    logging.info(f"Found {len(val_quadruplets)} validation samples")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders_quadruplets(
        train_quadruplets,
        batch_size=args.batch_size,
        shape=(64, 64),
        num_workers=4,
        pin_memory=(device == 'cuda'),
        val_quadruplets=val_quadruplets
    )
    
    # Load validation data for evaluation
    val_data = None
    if val_quadruplets:
        val_input, val_velocity = load_validation_data_numpy_quadruplets(
            val_quadruplets,
            max_samples=min(200, len(val_quadruplets)),  # Limit for efficiency
            shape=(64, 64)
        )
        if val_input is not None:
            val_data = (val_input, val_velocity)
            logging.info(f"Loaded validation data: {val_input.shape}")
    
    # Create model
    model = UNetFNO(
        in_channels=2,  # Pressure + WSS
        out_channels=2,  # U + V velocity
        base_channels=args.channels,
        depth=args.depth,
        modes=args.modes,
        enforce_incompressible=args.incompressible
    )
    
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Physics loss weights for Navier-Stokes
    physics_weights = {
        'mse': 1.0,           # Main reconstruction loss
        'divergence': 0.1,    # Incompressibility constraint
        'boundary': 0.05,     # Periodic boundary conditions
        'spectral': 0.1       # Energy spectrum preservation
    }
    
    # Create trainer
    trainer = UNetFNOTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        physics_weights=physics_weights,
        use_mixed_precision=args.mixed_precision
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
    
    # Training configuration
    logging.info("\n" + "="*60)
    logging.info("TRAINING CONFIGURATION")
    logging.info("="*60)
    logging.info(f"  Epochs: {args.epochs} (starting from {start_epoch})")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Learning rate: {args.lr}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Mixed precision: {args.mixed_precision}")
    logging.info(f"  Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logging.info("\nModel Architecture:")
    logging.info(f"  Fourier modes: {args.modes}")
    logging.info(f"  Base channels: {args.channels}")
    logging.info(f"  U-Net depth: {args.depth}")
    logging.info(f"  Incompressible flow: {args.incompressible}")
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
            epochs=args.epochs,
            save_dir=models_dir,
            log_dir=logs_dir,
            save_interval=args.save_interval,
            validation_interval=args.val_interval
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
            best_epoch = history['val_loss'].index(best_val_loss) * args.val_interval + 1
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
        logging.info(f"  - checkpoint_epoch_*.pt (every {args.save_interval} epochs)")
        logging.info("  - initial_checkpoint.pt (starting point)")
        
        return 0
        
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)