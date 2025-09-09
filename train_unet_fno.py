import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging
from datetime import datetime

from unet_fno_model import UNetFNO, compute_multiscale_physics_losses, count_parameters


class UNetFNOTrainer:
    def __init__(self, model, device='cuda', learning_rate=2e-4, 
                 physics_weights=None, use_mixed_precision=False, noise_std=0.02):
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.noise_std = noise_std
        
        # Physics weights for multi-scale training
        if physics_weights is None:
            physics_weights = {
                'mse': 0.7,
                'weighted_mse': 0.3,
                'huber': 0.2,
                'magnitude_l1': 0.1,
                'divergence': 0.05,
                'boundary': 0.001,
                'frequency': 0.05
            }
        self.physics_weights = physics_weights
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=5e-6,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=25, T_mult=2, eta_min=1e-6
        )
        
        # Mixed precision
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [], 'physics_losses': {},
            'learning_rate': []
        }
        
        for key in self.physics_weights.keys():
            self.history['physics_losses'][key] = []
        
        # High velocity sampling
        self.high_velocity_threshold = 0.8
        
    def add_training_noise(self, input_data):
        """Add gaussian noise to inputs during training"""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(input_data) * self.noise_std
            return input_data + noise
        return input_data
    
    def sample_high_velocity_regions(self, target_batch, sample_ratio=0.3):
        """Sample more from high velocity regions during training"""
        velocity_mag = torch.sqrt(target_batch[:, 0]**2 + target_batch[:, 1]**2)
        high_vel_mask = velocity_mag > self.high_velocity_threshold
        
        # Create sampling weights
        weights = torch.ones_like(velocity_mag)
        weights[high_vel_mask] *= 3.0  # 3x more weight for high velocity regions
        
        return weights.unsqueeze(1)
    
    def compute_loss(self, pred_velocity, true_velocity, input_data):
        physics_losses = compute_multiscale_physics_losses(pred_velocity, true_velocity, input_data)
        
        total_loss = 0
        loss_dict = {}
        
        # Apply high velocity region weighting to main losses
        if self.training:
            velocity_weights = self.sample_high_velocity_regions(true_velocity)
            
            # Apply weights to main reconstruction losses
            if 'mse' in physics_losses:
                mse_loss = physics_losses['mse']
                weighted_mse = torch.mean(velocity_weights * (pred_velocity - true_velocity)**2)
                physics_losses['mse'] = 0.7 * mse_loss + 0.3 * weighted_mse
        
        for key, weight in self.physics_weights.items():
            if key in physics_losses:
                loss_value = physics_losses[key]
                total_loss += weight * loss_value
                loss_dict[key] = loss_value.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_losses = {key: 0 for key in self.physics_weights.keys()}
        epoch_losses['total'] = 0
        
        # Progressive learning: reduce noise over time
        current_noise = self.noise_std * max(0.1, 1.0 - epoch / 100)
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            
            # Add training noise
            if current_noise > 0:
                input_batch = self.add_training_noise(input_batch)
            
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    pred_batch = self.model(input_batch)
                    total_loss, losses = self.compute_loss(pred_batch, target_batch, input_batch)
                
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.3)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_batch = self.model(input_batch)
                total_loss, losses = self.compute_loss(pred_batch, target_batch, input_batch)
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.3)
                self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value
            
            if batch_idx % 20 == 0:
                logging.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                           f'Loss: {total_loss.item():.6f}, Noise: {current_noise:.4f}')
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        return epoch_losses
    
    def validate(self, val_data):
        if val_data is None:
            return None
        
        self.model.eval()
        val_input, val_target = val_data
        
        total_losses = {key: 0 for key in self.physics_weights.keys()}
        total_losses['total'] = 0
        num_batches = 0
        
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(val_input), batch_size):
                end_idx = min(i + batch_size, len(val_input))
                
                input_batch = torch.tensor(val_input[i:end_idx], dtype=torch.float32).to(self.device)
                target_batch = torch.tensor(val_target[i:end_idx], dtype=torch.float32).to(self.device)
                
                pred_batch = self.model(input_batch)
                _, losses = self.compute_loss(pred_batch, target_batch, input_batch)
                
                for key, value in losses.items():
                    total_losses[key] += value
                num_batches += 1
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
        
        return total_losses
    
    def create_tensorboard_visualizations(self, val_data, epoch, writer, num_samples=4):
        """Create visualizations with peak analysis"""
        if val_data is None:
            return None
        
        self.model.eval()
        val_input, val_target = val_data
        
        np.random.seed(42)
        indices = np.random.choice(len(val_input), min(num_samples, len(val_input)), replace=False)
        
        sample_input = torch.tensor(val_input[indices], dtype=torch.float32).to(self.device)
        sample_target = torch.tensor(val_target[indices], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            sample_pred = self.model(sample_input)
        
        input_np = sample_input.cpu().numpy()
        target_np = sample_target.cpu().numpy()
        pred_np = sample_pred.cpu().numpy()
        
        # Create visualization grid with peak analysis
        fig, axes = plt.subplots(num_samples, 9, figsize=(36, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Input fields
            im1 = axes[i, 0].imshow(input_np[i, 0], cmap='viridis', aspect='equal')
            axes[i, 0].set_title(f'Sample {i+1}\nPressure Input', fontsize=10)
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            im2 = axes[i, 1].imshow(input_np[i, 1], cmap='plasma', aspect='equal')
            axes[i, 1].set_title('WSS Input', fontsize=10)
            axes[i, 1].axis('off')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # U-velocity comparison
            u_min, u_max = target_np[i, 0].min(), target_np[i, 0].max()
            
            im3 = axes[i, 2].imshow(target_np[i, 0], cmap='coolwarm', vmin=u_min, vmax=u_max, aspect='equal')
            axes[i, 2].set_title('True U-velocity', fontsize=10)
            axes[i, 2].axis('off')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            im4 = axes[i, 3].imshow(pred_np[i, 0], cmap='coolwarm', vmin=u_min, vmax=u_max, aspect='equal')
            axes[i, 3].set_title('Pred U-velocity', fontsize=10)
            axes[i, 3].axis('off')
            plt.colorbar(im4, ax=axes[i, 3], fraction=0.046, pad=0.04)
            
            # V-velocity comparison
            v_min, v_max = target_np[i, 1].min(), target_np[i, 1].max()
            
            im5 = axes[i, 4].imshow(target_np[i, 1], cmap='coolwarm', vmin=v_min, vmax=v_max, aspect='equal')
            axes[i, 4].set_title('True V-velocity', fontsize=10)
            axes[i, 4].axis('off')
            plt.colorbar(im5, ax=axes[i, 4], fraction=0.046, pad=0.04)
            
            im6 = axes[i, 5].imshow(pred_np[i, 1], cmap='coolwarm', vmin=v_min, vmax=v_max, aspect='equal')
            axes[i, 5].set_title('Pred V-velocity', fontsize=10)
            axes[i, 5].axis('off')
            plt.colorbar(im6, ax=axes[i, 5], fraction=0.046, pad=0.04)
            
            # Error maps with peak analysis
            u_error = np.abs(target_np[i, 0] - pred_np[i, 0])
            v_error = np.abs(target_np[i, 1] - pred_np[i, 1])
            
            im7 = axes[i, 6].imshow(u_error, cmap='Reds', aspect='equal')
            u_mae = np.mean(u_error)
            u_peak_error = np.max(u_error)
            axes[i, 6].set_title(f'U Error\nMAE: {u_mae:.4f}\nPeak: {u_peak_error:.4f}', fontsize=10)
            axes[i, 6].axis('off')
            plt.colorbar(im7, ax=axes[i, 6], fraction=0.046, pad=0.04)
            
            im8 = axes[i, 7].imshow(v_error, cmap='Reds', aspect='equal')
            v_mae = np.mean(v_error)
            v_peak_error = np.max(v_error)
            axes[i, 7].set_title(f'V Error\nMAE: {v_mae:.4f}\nPeak: {v_peak_error:.4f}', fontsize=10)
            axes[i, 7].axis('off')
            plt.colorbar(im8, ax=axes[i, 7], fraction=0.046, pad=0.04)
            
            # Velocity magnitude comparison with peak analysis
            true_mag = np.sqrt(target_np[i, 0]**2 + target_np[i, 1]**2)
            pred_mag = np.sqrt(pred_np[i, 0]**2 + pred_np[i, 1]**2)
            mag_error = np.abs(true_mag - pred_mag)
            
            im9 = axes[i, 8].imshow(mag_error, cmap='Reds', aspect='equal')
            mag_mae = np.mean(mag_error)
            max_true_mag = np.max(true_mag)
            max_pred_mag = np.max(pred_mag)
            axes[i, 8].set_title(f'Mag Error\nMAE: {mag_mae:.4f}\nTrue Max: {max_true_mag:.3f}\nPred Max: {max_pred_mag:.3f}', fontsize=9)
            axes[i, 8].axis('off')
            plt.colorbar(im9, ax=axes[i, 8], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Add to TensorBoard
        writer.add_figure('Validation/Velocity_Comparisons_with_Peaks', fig, epoch)
        
        # Log validation metrics with peak analysis
        self._log_validation_metrics_with_peaks(target_np, pred_np, writer, epoch)
        
        plt.close()
        return fig
    
    def _log_validation_metrics_with_peaks(self, target_np, pred_np, writer, epoch):
        """Log validation metrics with peak analysis"""
        
        # Standard metrics
        u_true, u_pred = target_np[:, 0], pred_np[:, 0]
        v_true, v_pred = target_np[:, 1], pred_np[:, 1]
        
        u_mae = np.mean(np.abs(u_true - u_pred))
        v_mae = np.mean(np.abs(v_true - v_pred))
        u_mse = np.mean((u_true - u_pred) ** 2)
        v_mse = np.mean((v_true - v_pred) ** 2)
        
        u_corr = np.corrcoef(u_true.flatten(), u_pred.flatten())[0, 1]
        v_corr = np.corrcoef(v_true.flatten(), v_pred.flatten())[0, 1]
        
        # Peak analysis
        u_true_max = np.max(np.abs(u_true))
        u_pred_max = np.max(np.abs(u_pred))
        v_true_max = np.max(np.abs(v_true))
        v_pred_max = np.max(np.abs(v_pred))
        
        # Peak preservation ratio
        u_peak_ratio = u_pred_max / (u_true_max + 1e-8)
        v_peak_ratio = v_pred_max / (v_true_max + 1e-8)
        
        # High velocity region analysis
        mag_true = np.sqrt(u_true**2 + v_true**2)
        mag_pred = np.sqrt(u_pred**2 + v_pred**2)
        
        # Find high velocity regions (top 10%)
        high_vel_threshold = np.percentile(mag_true, 90)
        high_vel_mask = mag_true > high_vel_threshold
        
        if np.any(high_vel_mask):
            high_vel_mae = np.mean(np.abs(mag_true[high_vel_mask] - mag_pred[high_vel_mask]))
            high_vel_corr = np.corrcoef(mag_true[high_vel_mask].flatten(), 
                                       mag_pred[high_vel_mask].flatten())[0, 1]
        else:
            high_vel_mae = 0
            high_vel_corr = 1
        
        # Standard metrics
        writer.add_scalar('Validation_Metrics/U_MAE', u_mae, epoch)
        writer.add_scalar('Validation_Metrics/V_MAE', v_mae, epoch)
        writer.add_scalar('Validation_Metrics/U_MSE', u_mse, epoch)
        writer.add_scalar('Validation_Metrics/V_MSE', v_mse, epoch)
        writer.add_scalar('Validation_Metrics/U_Correlation', u_corr, epoch)
        writer.add_scalar('Validation_Metrics/V_Correlation', v_corr, epoch)
        
        # Peak analysis metrics
        writer.add_scalar('Peak_Analysis/U_True_Max', u_true_max, epoch)
        writer.add_scalar('Peak_Analysis/U_Pred_Max', u_pred_max, epoch)
        writer.add_scalar('Peak_Analysis/V_True_Max', v_true_max, epoch)
        writer.add_scalar('Peak_Analysis/V_Pred_Max', v_pred_max, epoch)
        writer.add_scalar('Peak_Analysis/U_Peak_Ratio', u_peak_ratio, epoch)
        writer.add_scalar('Peak_Analysis/V_Peak_Ratio', v_peak_ratio, epoch)
        
        # High velocity region metrics
        writer.add_scalar('High_Velocity/MAE', high_vel_mae, epoch)
        writer.add_scalar('High_Velocity/Correlation', high_vel_corr, epoch)
        writer.add_scalar('High_Velocity/Threshold', high_vel_threshold, epoch)
        
        # Magnitude metrics
        mag_mae = np.mean(np.abs(mag_true - mag_pred))
        mag_corr = np.corrcoef(mag_true.flatten(), mag_pred.flatten())[0, 1]
        writer.add_scalar('Validation_Metrics/Magnitude_MAE', mag_mae, epoch)
        writer.add_scalar('Validation_Metrics/Magnitude_Correlation', mag_corr, epoch)
        
        # Log detailed statistics
        logging.info(f"Validation Metrics with Peak Analysis - Epoch {epoch}:")
        logging.info(f"  U-velocity: MAE={u_mae:.4f}, MSE={u_mse:.4f}, Corr={u_corr:.3f}")
        logging.info(f"  V-velocity: MAE={v_mae:.4f}, MSE={v_mse:.4f}, Corr={v_corr:.3f}")
        logging.info(f"  Peak ratios: U={u_peak_ratio:.3f}, V={v_peak_ratio:.3f}")
        logging.info(f"  High-vel regions: MAE={high_vel_mae:.4f}, Corr={high_vel_corr:.3f}")
    
    def train(self, train_loader, val_data=None, epochs=150, save_dir='./models', 
              log_dir='./logs', save_interval=15, validation_interval=5):
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        writer = SummaryWriter(log_dir)
        logging.info(f"TensorBoard logging enabled. View with: tensorboard --logdir={log_dir}")
        
        best_val_loss = float('inf')
        
        logging.info(f"Starting training for {epochs} epochs")
        logging.info(f"Model parameters: {count_parameters(self.model):,}")
        
        # Log model info
        writer.add_text('Model/Architecture', f"""
        U-Net-FNO Model with Attention and Multi-scale Loss:
        - Base channels: 96
        - FNO modes: 16
        - Attention gates in decoder
        - Multi-layer output head with residual
        - Multi-scale loss computation
        - Peak preservation techniques
        - Training noise: {self.noise_std}
        - Total parameters: {count_parameters(self.model):,}
        """, 0)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            for key in self.physics_weights.keys():
                if key in train_losses:
                    self.history['physics_losses'][key].append(train_losses[key])
            
            # Learning rate step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Validation
            val_losses = None
            if val_data is not None and (epoch + 1) % validation_interval == 0:
                val_losses = self.validate(val_data)
                if val_losses:
                    self.history['val_loss'].append(val_losses['total'])
                    
                    if val_losses['total'] < best_val_loss:
                        best_val_loss = val_losses['total']
                        self.save_model(os.path.join(save_dir, 'best_model.pt'), epoch)
            
            # TensorBoard logging
            writer.add_scalar('Loss/Train', train_losses['total'], epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            for key, value in train_losses.items():
                if key != 'total':
                    writer.add_scalar(f'Train_Physics/{key}', value, epoch)
            
            if val_losses:
                writer.add_scalar('Loss/Validation', val_losses['total'], epoch)
                for key, value in val_losses.items():
                    if key != 'total':
                        writer.add_scalar(f'Val_Physics/{key}', value, epoch)
            
            # Log gradient norms
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            writer.add_scalar('Gradients/Total_Norm', total_norm, epoch)
            
            writer.flush()
            
            # Visualizations with peak analysis
            if (epoch + 1) % validation_interval == 0:
                self.create_tensorboard_visualizations(val_data, epoch, writer, num_samples=4)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_model(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'), epoch)
            
            # Logging
            epoch_time = time.time() - start_time
            val_msg = f", Val Loss: {val_losses['total']:.6f}" if val_losses else ""
            
            logging.info(f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_losses['total']:.6f}{val_msg} - "
                        f"Time: {epoch_time:.2f}s, LR: {current_lr:.2e}")
        
        writer.close()
        logging.info("Training completed!")
        
        return self.history
    
    def save_model(self, filepath, epoch, **kwargs):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'physics_weights': self.physics_weights,
            'noise_std': self.noise_std,
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'in_channels': 2,
                'out_channels': 2,
                'base_channels': 96,
                'depth': 3,
                'modes': 16,
                'use_attention': True
            }
        }
        
        if hasattr(self, 'scaler'):
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved: {os.path.basename(filepath)}")
    
    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and hasattr(self, 'scaler'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        if 'noise_std' in checkpoint:
            self.noise_std = checkpoint['noise_std']
        
        epoch = checkpoint.get('epoch', -1)
        logging.info(f"Model loaded from epoch {epoch + 1}")
        
        return epoch + 1