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

from unet_fno_model import UNetFNO, compute_physics_losses, count_parameters


class UNetFNOTrainer:
    def __init__(self, model, device='cuda', learning_rate=2e-4, 
                 physics_weights=None, use_mixed_precision=False):
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        
        # Simplified physics weights - no spectral loss
        if physics_weights is None:
            physics_weights = {
                'mse': 1.0,
                'divergence': 0.01,
                'boundary': 0.001
            }
        self.physics_weights = physics_weights
        
        # Optimizer with lighter regularization
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5,  # Reduced weight decay
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler with higher minimum
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=75, eta_min=1e-5
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
        
    def compute_loss(self, pred_velocity, true_velocity, input_data):
        physics_losses = compute_physics_losses(pred_velocity, true_velocity, input_data)
        
        total_loss = 0
        loss_dict = {}
        
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
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    pred_batch = self.model(input_batch)
                    total_loss, losses = self.compute_loss(pred_batch, target_batch, input_batch)
                
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_batch = self.model(input_batch)
                total_loss, losses = self.compute_loss(pred_batch, target_batch, input_batch)
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value
            
            if batch_idx % 20 == 0:
                logging.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                           f'Loss: {total_loss.item():.6f}')
        
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
            batch_size = 64  # Larger batch for validation
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
        """Create comprehensive velocity visualizations for TensorBoard"""
        if val_data is None:
            return None
        
        self.model.eval()
        val_input, val_target = val_data
        
        # Fixed indices for consistent comparison across epochs
        np.random.seed(42)  # Fixed seed for consistent samples
        indices = np.random.choice(len(val_input), min(num_samples, len(val_input)), replace=False)
        
        sample_input = torch.tensor(val_input[indices], dtype=torch.float32).to(self.device)
        sample_target = torch.tensor(val_target[indices], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            sample_pred = self.model(sample_input)
        
        input_np = sample_input.cpu().numpy()
        target_np = sample_target.cpu().numpy()
        pred_np = sample_pred.cpu().numpy()
        
        # Create comprehensive visualization grid
        fig, axes = plt.subplots(num_samples, 8, figsize=(32, 4*num_samples))
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
            
            # U-velocity comparison (maintain same color scale)
            u_min, u_max = target_np[i, 0].min(), target_np[i, 0].max()
            
            im3 = axes[i, 2].imshow(target_np[i, 0], cmap='coolwarm', vmin=u_min, vmax=u_max, aspect='equal')
            axes[i, 2].set_title('True U-velocity', fontsize=10)
            axes[i, 2].axis('off')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            im4 = axes[i, 3].imshow(pred_np[i, 0], cmap='coolwarm', vmin=u_min, vmax=u_max, aspect='equal')
            axes[i, 3].set_title('Pred U-velocity', fontsize=10)
            axes[i, 3].axis('off')
            plt.colorbar(im4, ax=axes[i, 3], fraction=0.046, pad=0.04)
            
            # V-velocity comparison (maintain same color scale)
            v_min, v_max = target_np[i, 1].min(), target_np[i, 1].max()
            
            im5 = axes[i, 4].imshow(target_np[i, 1], cmap='coolwarm', vmin=v_min, vmax=v_max, aspect='equal')
            axes[i, 4].set_title('True V-velocity', fontsize=10)
            axes[i, 4].axis('off')
            plt.colorbar(im5, ax=axes[i, 4], fraction=0.046, pad=0.04)
            
            im6 = axes[i, 5].imshow(pred_np[i, 1], cmap='coolwarm', vmin=v_min, vmax=v_max, aspect='equal')
            axes[i, 5].set_title('Pred V-velocity', fontsize=10)
            axes[i, 5].axis('off')
            plt.colorbar(im6, ax=axes[i, 5], fraction=0.046, pad=0.04)
            
            # Error maps
            u_error = np.abs(target_np[i, 0] - pred_np[i, 0])
            v_error = np.abs(target_np[i, 1] - pred_np[i, 1])
            
            im7 = axes[i, 6].imshow(u_error, cmap='Reds', aspect='equal')
            u_mae = np.mean(u_error)
            axes[i, 6].set_title(f'U-velocity Error\nMAE: {u_mae:.4f}', fontsize=10)
            axes[i, 6].axis('off')
            plt.colorbar(im7, ax=axes[i, 6], fraction=0.046, pad=0.04)
            
            im8 = axes[i, 7].imshow(v_error, cmap='Reds', aspect='equal')
            v_mae = np.mean(v_error)
            axes[i, 7].set_title(f'V-velocity Error\nMAE: {v_mae:.4f}', fontsize=10)
            axes[i, 7].axis('off')
            plt.colorbar(im8, ax=axes[i, 7], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Add to TensorBoard
        writer.add_figure('Validation/Velocity_Comparisons', fig, epoch)
        
        # Save individual images to TensorBoard for easier viewing
        for i in range(num_samples):
            # Input fields
            writer.add_image(f'Sample_{i+1}/Input_Pressure', 
                           self._normalize_for_tensorboard(input_np[i, 0]), epoch)
            writer.add_image(f'Sample_{i+1}/Input_WSS', 
                           self._normalize_for_tensorboard(input_np[i, 1]), epoch)
            
            # U-velocity
            writer.add_image(f'Sample_{i+1}/U_True', 
                           self._normalize_for_tensorboard(target_np[i, 0]), epoch)
            writer.add_image(f'Sample_{i+1}/U_Pred', 
                           self._normalize_for_tensorboard(pred_np[i, 0]), epoch)
            writer.add_image(f'Sample_{i+1}/U_Error', 
                           self._normalize_for_tensorboard(u_error), epoch)
            
            # V-velocity
            writer.add_image(f'Sample_{i+1}/V_True', 
                           self._normalize_for_tensorboard(target_np[i, 1]), epoch)
            writer.add_image(f'Sample_{i+1}/V_Pred', 
                           self._normalize_for_tensorboard(pred_np[i, 1]), epoch)
            writer.add_image(f'Sample_{i+1}/V_Error', 
                           self._normalize_for_tensorboard(v_error), epoch)
        
        # Compute and log validation metrics
        self._log_validation_metrics(target_np, pred_np, writer, epoch)
        
        plt.close()
        return fig
    
    def _normalize_for_tensorboard(self, image):
        """Normalize image for TensorBoard display"""
        # Normalize to [0, 1] for TensorBoard
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            normalized = np.zeros_like(image)
        
        # Add channel dimension and convert to tensor
        return torch.tensor(normalized).unsqueeze(0)
    
    def _log_validation_metrics(self, target_np, pred_np, writer, epoch):
        """Log detailed validation metrics to TensorBoard"""
        
        # Compute metrics for both velocity components
        u_true, u_pred = target_np[:, 0], pred_np[:, 0]
        v_true, v_pred = target_np[:, 1], pred_np[:, 1]
        
        # MAE metrics
        u_mae = np.mean(np.abs(u_true - u_pred))
        v_mae = np.mean(np.abs(v_true - v_pred))
        total_mae = (u_mae + v_mae) / 2
        
        # MSE metrics
        u_mse = np.mean((u_true - u_pred) ** 2)
        v_mse = np.mean((v_true - v_pred) ** 2)
        total_mse = (u_mse + v_mse) / 2
        
        # Correlation metrics
        u_corr = np.corrcoef(u_true.flatten(), u_pred.flatten())[0, 1]
        v_corr = np.corrcoef(v_true.flatten(), v_pred.flatten())[0, 1]
        
        # Velocity magnitude metrics
        mag_true = np.sqrt(u_true**2 + v_true**2)
        mag_pred = np.sqrt(u_pred**2 + v_pred**2)
        mag_mae = np.mean(np.abs(mag_true - mag_pred))
        mag_corr = np.corrcoef(mag_true.flatten(), mag_pred.flatten())[0, 1]
        
        # Log to TensorBoard
        writer.add_scalar('Validation_Metrics/U_MAE', u_mae, epoch)
        writer.add_scalar('Validation_Metrics/V_MAE', v_mae, epoch)
        writer.add_scalar('Validation_Metrics/Total_MAE', total_mae, epoch)
        
        writer.add_scalar('Validation_Metrics/U_MSE', u_mse, epoch)
        writer.add_scalar('Validation_Metrics/V_MSE', v_mse, epoch)
        writer.add_scalar('Validation_Metrics/Total_MSE', total_mse, epoch)
        
        writer.add_scalar('Validation_Metrics/U_Correlation', u_corr, epoch)
        writer.add_scalar('Validation_Metrics/V_Correlation', v_corr, epoch)
        
        writer.add_scalar('Validation_Metrics/Magnitude_MAE', mag_mae, epoch)
        writer.add_scalar('Validation_Metrics/Magnitude_Correlation', mag_corr, epoch)
        
        # Log detailed statistics
        logging.info(f"Validation Metrics - Epoch {epoch}:")
        logging.info(f"  U-velocity: MAE={u_mae:.4f}, MSE={u_mse:.4f}, Corr={u_corr:.3f}")
        logging.info(f"  V-velocity: MAE={v_mae:.4f}, MSE={v_mse:.4f}, Corr={v_corr:.3f}")
        logging.info(f"  Magnitude: MAE={mag_mae:.4f}, Corr={mag_corr:.3f}")
    
    def create_visualizations(self, val_data, epoch, num_samples=4):
        """Legacy method - kept for compatibility"""
        if val_data is None:
            return None
        
        self.model.eval()
        val_input, val_target = val_data
        
        indices = np.random.choice(len(val_input), min(num_samples, len(val_input)), replace=False)
        
        sample_input = torch.tensor(val_input[indices], dtype=torch.float32).to(self.device)
        sample_target = torch.tensor(val_target[indices], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            sample_pred = self.model(sample_input)
        
        input_np = sample_input.cpu().numpy()
        target_np = sample_target.cpu().numpy()
        pred_np = sample_pred.cpu().numpy()
        
        fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Pressure input
            axes[i, 0].imshow(input_np[i, 0], cmap='viridis', aspect='equal')
            axes[i, 0].set_title('Pressure')
            axes[i, 0].axis('off')
            
            # WSS input
            axes[i, 1].imshow(input_np[i, 1], cmap='plasma', aspect='equal')
            axes[i, 1].set_title('Wall Shear Stress')
            axes[i, 1].axis('off')
            
            # True u-velocity
            axes[i, 2].imshow(target_np[i, 0], cmap='coolwarm', aspect='equal')
            axes[i, 2].set_title('True U-velocity')
            axes[i, 2].axis('off')
            
            # Predicted u-velocity
            axes[i, 3].imshow(pred_np[i, 0], cmap='coolwarm', aspect='equal')
            axes[i, 3].set_title('Predicted U-velocity')
            axes[i, 3].axis('off')
            
            # Error map
            error = np.abs(target_np[i, 0] - pred_np[i, 0])
            axes[i, 4].imshow(error, cmap='Reds', aspect='equal')
            axes[i, 4].set_title('U-velocity Error')
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'visualization_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def train(self, train_loader, val_data=None, epochs=150, save_dir='./models', 
              log_dir='./logs', save_interval=15, validation_interval=5):
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        writer = SummaryWriter(log_dir)
        logging.info(f"TensorBoard logging enabled. View with: tensorboard --logdir={log_dir}")
        
        best_val_loss = float('inf')
        
        logging.info(f"Starting simplified training for {epochs} epochs")
        logging.info(f"Model parameters: {count_parameters(self.model):,}")
        
        # Log model info
        writer.add_text('Model/Architecture', f"""
        Simplified U-Net-FNO Model:
        - Depth: 3 (reduced from 4)
        - FNO modes: 12 (reduced from 16)  
        - Base channels: 64
        - Removed spectral loss
        - Reduced physics weights
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
            
            # Enhanced TensorBoard visualizations with both velocity components
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
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'in_channels': 2,
                'out_channels': 2,
                'base_channels': 64,
                'depth': 3,
                'modes': 12
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
        
        epoch = checkpoint.get('epoch', -1)
        logging.info(f"Model loaded from epoch {epoch + 1}")
        
        return epoch + 1