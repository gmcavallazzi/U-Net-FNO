#!/usr/bin/env python3
"""
U-Net-FNO Visualization Script

Loads a trained U-Net-FNO model and creates visualizations of velocity field predictions
from pressure and wall shear stress inputs.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import random
from matplotlib.colors import Normalize

from unet_fno_model import UNetFNO
from data_loader import find_matching_quadruplets, PressureWSSUVVelocityDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize U-Net-FNO predictions')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--val_dir', type=str, default='./validation_wss_uv',
                       help='Validation data directory')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU usage')
    return parser.parse_args()


def load_model(model_path, device):
    """Load trained U-Net-FNO model"""
    
    # Create model with standard parameters
    model = UNetFNO(
        in_channels=2,
        out_channels=2,
        base_channels=64,
        depth=4,
        modes=16,
        enforce_incompressible=True
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def load_validation_samples(val_dir, num_samples):
    """Load validation samples"""
    
    # Find validation quadruplets
    _, val_quadruplets = find_matching_quadruplets(
        data_dir='./data_wss_uv',
        validation_dir=val_dir,
        shuffle=True,
        max_train=0,
        max_val=None
    )
    
    if not val_quadruplets:
        raise ValueError(f"No validation data found in {val_dir}")
    
    # Create dataset
    dataset = PressureWSSUVVelocityDataset(
        val_quadruplets,
        shape=(64, 64),
        dtype='float64',
        normalize=True
    )
    
    # Select random samples
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    
    samples = []
    for idx in indices:
        input_tensor, velocity_tensor = dataset[idx]
        
        samples.append({
            'input': input_tensor.numpy(),      # [2, 64, 64] - pressure, WSS
            'target': velocity_tensor.numpy(), # [2, 64, 64] - u, v velocity
            'pressure': input_tensor[0].numpy(),
            'wss': input_tensor[1].numpy(),
            'u_velocity': velocity_tensor[0].numpy(),
            'v_velocity': velocity_tensor[1].numpy()
        })
    
    print(f"Loaded {len(samples)} validation samples")
    return samples


def generate_predictions(model, samples, device):
    """Generate predictions for samples"""
    
    model.eval()
    
    for i, sample in enumerate(samples):
        input_tensor = torch.tensor(sample['input'], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_tensor = model(input_tensor)
            pred_numpy = pred_tensor.squeeze(0).cpu().numpy()
        
        sample['prediction'] = pred_numpy
        sample['pred_u'] = pred_numpy[0]
        sample['pred_v'] = pred_numpy[1]
        
        # Calculate errors
        sample['error_u'] = np.abs(sample['u_velocity'] - sample['pred_u'])
        sample['error_v'] = np.abs(sample['v_velocity'] - sample['pred_v'])
        
        # Calculate metrics
        sample['mae_u'] = np.mean(sample['error_u'])
        sample['mae_v'] = np.mean(sample['error_v'])
        sample['mse_u'] = np.mean((sample['u_velocity'] - sample['pred_u'])**2)
        sample['mse_v'] = np.mean((sample['v_velocity'] - sample['pred_v'])**2)
        
        # Calculate correlation
        sample['corr_u'] = np.corrcoef(sample['u_velocity'].flatten(), sample['pred_u'].flatten())[0, 1]
        sample['corr_v'] = np.corrcoef(sample['v_velocity'].flatten(), sample['pred_v'].flatten())[0, 1]
        
        print(f"Sample {i+1}: U-MAE={sample['mae_u']:.4f}, V-MAE={sample['mae_v']:.4f}, "
              f"U-Corr={sample['corr_u']:.3f}, V-Corr={sample['corr_v']:.3f}")
    
    return samples


def create_comparison_plot(samples, output_dir):
    """Create detailed comparison plots"""
    
    for i, sample in enumerate(samples):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Row 1: Inputs
        im1 = axes[0, 0].imshow(sample['pressure'], cmap='viridis', aspect='equal')
        axes[0, 0].set_title('Pressure (Input)')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        im2 = axes[0, 1].imshow(sample['wss'], cmap='plasma', aspect='equal')
        axes[0, 1].set_title('Wall Shear Stress (Input)')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        axes[0, 2].axis('off')  # Empty space
        
        # Row 2: U-velocity comparison
        vmin_u, vmax_u = sample['u_velocity'].min(), sample['u_velocity'].max()
        
        im3 = axes[1, 0].imshow(sample['u_velocity'], cmap='coolwarm', vmin=vmin_u, vmax=vmax_u, aspect='equal')
        axes[1, 0].set_title('True U-velocity')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        im4 = axes[1, 1].imshow(sample['pred_u'], cmap='coolwarm', vmin=vmin_u, vmax=vmax_u, aspect='equal')
        axes[1, 1].set_title('Predicted U-velocity')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        
        im5 = axes[1, 2].imshow(sample['error_u'], cmap='Reds', aspect='equal')
        axes[1, 2].set_title('U-velocity Error')
        axes[1, 2].axis('off')
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
        
        # Row 3: V-velocity comparison
        vmin_v, vmax_v = sample['v_velocity'].min(), sample['v_velocity'].max()
        
        im6 = axes[2, 0].imshow(sample['v_velocity'], cmap='coolwarm', vmin=vmin_v, vmax=vmax_v, aspect='equal')
        axes[2, 0].set_title('True V-velocity')
        axes[2, 0].axis('off')
        plt.colorbar(im6, ax=axes[2, 0], fraction=0.046)
        
        im7 = axes[2, 1].imshow(sample['pred_v'], cmap='coolwarm', vmin=vmin_v, vmax=vmax_v, aspect='equal')
        axes[2, 1].set_title('Predicted V-velocity')
        axes[2, 1].axis('off')
        plt.colorbar(im7, ax=axes[2, 1], fraction=0.046)
        
        im8 = axes[2, 2].imshow(sample['error_v'], cmap='Reds', aspect='equal')
        axes[2, 2].set_title('V-velocity Error')
        axes[2, 2].axis('off')
        plt.colorbar(im8, ax=axes[2, 2], fraction=0.046)
        
        # Add metrics as text
        metrics_text = (f"U-velocity: MAE={sample['mae_u']:.4f}, MSE={sample['mse_u']:.4f}, Corr={sample['corr_u']:.3f}\n"
                       f"V-velocity: MAE={sample['mae_v']:.4f}, MSE={sample['mse_v']:.4f}, Corr={sample['corr_v']:.3f}")
        
        fig.suptitle(f"Sample {i+1} - U-Net-FNO Prediction\n{metrics_text}", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}_detailed.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_summary_plot(samples, output_dir):
    """Create summary comparison plot"""
    
    fig, axes = plt.subplots(len(samples), 5, figsize=(20, 4*len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        # Pressure
        axes[i, 0].imshow(sample['pressure'], cmap='viridis', aspect='equal')
        axes[i, 0].set_title('Pressure')
        axes[i, 0].axis('off')
        
        # WSS
        axes[i, 1].imshow(sample['wss'], cmap='plasma', aspect='equal')
        axes[i, 1].set_title('WSS')
        axes[i, 1].axis('off')
        
        # True velocity magnitude
        vel_mag_true = np.sqrt(sample['u_velocity']**2 + sample['v_velocity']**2)
        axes[i, 2].imshow(vel_mag_true, cmap='viridis', aspect='equal')
        axes[i, 2].set_title('True |V|')
        axes[i, 2].axis('off')
        
        # Predicted velocity magnitude
        vel_mag_pred = np.sqrt(sample['pred_u']**2 + sample['pred_v']**2)
        axes[i, 3].imshow(vel_mag_pred, cmap='viridis', aspect='equal')
        axes[i, 3].set_title('Pred |V|')
        axes[i, 3].axis('off')
        
        # Error magnitude
        error_mag = np.abs(vel_mag_true - vel_mag_pred)
        axes[i, 4].imshow(error_mag, cmap='Reds', aspect='equal')
        axes[i, 4].set_title(f'Error |V|\nMAE: {np.mean(error_mag):.3f}')
        axes[i, 4].axis('off')
    
    plt.suptitle('U-Net-FNO Velocity Magnitude Predictions', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Load validation samples
    samples = load_validation_samples(args.val_dir, args.num_samples)
    
    # Generate predictions
    samples_with_preds = generate_predictions(model, samples, device)
    
    # Create visualizations
    create_comparison_plot(samples_with_preds, args.output_dir)
    create_summary_plot(samples_with_preds, args.output_dir)
    
    # Print summary statistics
    mae_u_values = [s['mae_u'] for s in samples_with_preds]
    mae_v_values = [s['mae_v'] for s in samples_with_preds]
    corr_u_values = [s['corr_u'] for s in samples_with_preds]
    corr_v_values = [s['corr_v'] for s in samples_with_preds]
    
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Average U-velocity MAE: {np.mean(mae_u_values):.4f} ± {np.std(mae_u_values):.4f}")
    print(f"Average V-velocity MAE: {np.mean(mae_v_values):.4f} ± {np.std(mae_v_values):.4f}")
    print(f"Average U-velocity Correlation: {np.mean(corr_u_values):.3f} ± {np.std(corr_u_values):.3f}")
    print(f"Average V-velocity Correlation: {np.mean(corr_v_values):.3f} ± {np.std(corr_v_values):.3f}")
    print(f"\nVisualizationsbsaved to: {args.output_dir}")
    
    print("\nGenerated files:")
    print("  - summary_comparison.png")
    for i in range(len(samples_with_preds)):
        print(f"  - sample_{i+1}_detailed.png")


if __name__ == "__main__":
    main()