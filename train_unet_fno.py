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
    """Streamlined trainer for U-Net-FNO model"""
    
    def __init__(self, model, device='cuda', learning_rate=1e-4, 
                 physics_weights=None, use_mixed_precision=False):
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        
        # Default physics weights
        if physics_weights is None:
            physics_weights = {
                'mse': 1.0,
                'divergence': 0.1,
                'boundary': 0.05,
                'spectral': 0.1
            }
        self.physics_weights = physics_weights
        
        # Optimizer with warm restart
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
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
        """Compute combined loss"""
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
        """Train for one epoch"""
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_batch = self.model(input_batch)
                total_loss, losses = self.compute_loss(pred_batch, target_batch, input_batch)
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        """Validate on held-out data"""
        if val_data is None:
            return None
        
        self.model.eval()
        val_input, val_target = val_data
        
        total_losses = {key: 0 for key in self.physics_weights.keys()}
        total_losses['total'] = 0
        num_batches = 0
        
        with torch.no_grad():
            batch_size = 32
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
    
    def create_visualizations(self, val_data, epoch, num_samples=4):
        """Create visualization of predictions"""
        if val_data is None:
            return None
        
        self.model.eval()
        val_input, val_target = val_data
        
        # Select random samples
        indices = np.random.choice(len(val_input), min(num_samples, len(val_input)), replace=False)
        
        sample_input = torch.tensor(val_input[indices], dtype=torch.float32).to(self.device)
        sample_target = torch.tensor(val_target[indices], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            sample_pred = self.model(sample_input)
        
        # Convert to numpy
        input_np = sample_input.cpu().numpy()
        target_np = sample_target.cpu().numpy()
        pred_np = sample_pred.cpu().numpy()
        
        # Create visualization grid
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
    
    def train(self, train_loader, val_data=None, epochs=100, save_dir='./models', 
              log_dir='./logs', save_interval=10, validation_interval=5):
        """Main training loop"""
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        writer = SummaryWriter(log_dir)
        
        best_val_loss = float('inf')
        
        logging.info(f"Starting training for {epochs} epochs")
        logging.info(f"Model parameters: {count_parameters(self.model):,}")
        
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
                    
                    # Save best model
                    if val_losses['total'] < best_val_loss:
                        best_val_loss = val_losses['total']
                        self.save_model(os.path.join(save_dir, 'best_model.pt'), epoch)
            
            # TensorBoard logging
            writer.add_scalar('Loss/Train', train_losses['total'], epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            for key, value in train_losses.items():
                if key != 'total':
                    writer.add_scalar(f'Physics/{key}', value, epoch)
            
            if val_losses:
                writer.add_scalar('Loss/Validation', val_losses['total'], epoch)
            
            # Visualizations
            if (epoch + 1) % validation_interval == 0:
                self.create_visualizations(val_data, epoch)
            
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
        """Save comprehensive model checkpoint"""
        
        # Collect additional metrics if provided
        metrics = {}
        for key, value in kwargs.items():
            if value is not None:
                metrics[key] = value
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'physics_weights': self.physics_weights,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'in_channels': getattr(self.model, 'in_channels', 2),
                'out_channels': getattr(self.model, 'out_channels', 2),
                'base_channels': getattr(self.model, 'base_channels', 64),
                'depth': getattr(self.model, 'depth', 4),
                'modes': getattr(self.model, 'modes', 16),
                'enforce_incompressible': getattr(self.model, 'enforce_incompressible', True)
            }
        }
        
        # Add scaler state if using mixed precision
        if hasattr(self, 'scaler'):
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        # Log checkpoint details
        checkpoint_info = f"Checkpoint saved: {os.path.basename(filepath)}"
        if 'is_best' in kwargs:
            checkpoint_info += " (BEST MODEL)"
        logging.info(checkpoint_info)
    
    def load_model(self, filepath):
        """Load comprehensive model checkpoint"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        logging.info(f"Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler if present
        if 'scaler_state_dict' in checkpoint and hasattr(self, 'scaler'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training history
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            logging.info(f"Loaded training history with {len(self.history['train_loss'])} epochs")
        
        # Load physics weights
        if 'physics_weights' in checkpoint:
            self.physics_weights = checkpoint['physics_weights']
        
        # Log checkpoint info
        epoch = checkpoint.get('epoch', -1)
        timestamp = checkpoint.get('timestamp', 'unknown')
        metrics = checkpoint.get('metrics', {})
        
        logging.info(f"Checkpoint loaded successfully:")
        logging.info(f"  Epoch: {epoch + 1}")
        logging.info(f"  Saved: {timestamp}")
        if metrics:
            for key, value in metrics.items():
                logging.info(f"  {key}: {value:.6f}")
        
        return epoch + 1  # Return next epoch to start from
    
    def save_training_plots(self, save_dir, epoch):
        """Save training progress plots"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Training and validation loss
            epochs_range = range(1, len(self.history['train_loss']) + 1)
            axes[0, 0].plot(epochs_range, self.history['train_loss'], 'b-', label='Training Loss')
            if self.history['val_loss']:
                val_epochs = range(5, len(self.history['val_loss']) * 5 + 1, 5)  # validation_interval = 5
                axes[0, 0].plot(val_epochs, self.history['val_loss'], 'r-', label='Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Progress')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Physics losses
            for key in ['mse', 'divergence', 'boundary', 'spectral']:
                if key in self.history['physics_losses'] and self.history['physics_losses'][key]:
                    axes[0, 1].plot(epochs_range, self.history['physics_losses'][key], 
                                   label=key.title())
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Physics Loss Components')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
            
            # Learning rate
            axes[1, 0].plot(epochs_range, self.history['learning_rate'], 'g-')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Loss ratio (if validation available)
            if self.history['val_loss'] and len(self.history['val_loss']) > 1:
                train_interp = np.interp(val_epochs, epochs_range, self.history['train_loss'])
                loss_ratio = np.array(self.history['val_loss']) / train_interp
                axes[1, 1].plot(val_epochs, loss_ratio, 'm-')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Val Loss / Train Loss')
                axes[1, 1].set_title('Overfitting Monitor')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Validation data\nnot available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Overfitting Monitor')
            
            plt.tight_layout()
            plot_path = os.path.join(save_dir, f'training_progress_epoch_{epoch+1}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Training plots saved: {plot_path}")
            
        except Exception as e:
            logging.warning(f"Failed to save training plots: {e}")
    
    def create_checkpoint_summary(self, save_dir):
        """Create a summary of all checkpoints"""
        
        try:
            checkpoint_files = []
            for filename in os.listdir(save_dir):
                if filename.endswith('.pt'):
                    filepath = os.path.join(save_dir, filename)
                    try:
                        checkpoint = torch.load(filepath, map_location='cpu')
                        info = {
                            'filename': filename,
                            'epoch': checkpoint.get('epoch', -1),
                            'timestamp': checkpoint.get('timestamp', 'unknown'),
                            'metrics': checkpoint.get('metrics', {})
                        }
                        checkpoint_files.append(info)
                    except:
                        continue
            
            # Sort by epoch
            checkpoint_files.sort(key=lambda x: x['epoch'])
            
            # Write summary
            summary_path = os.path.join(save_dir, 'checkpoint_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("CHECKPOINT SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                for info in checkpoint_files:
                    f.write(f"File: {info['filename']}\n")
                    f.write(f"Epoch: {info['epoch'] + 1}\n")
                    f.write(f"Timestamp: {info['timestamp']}\n")
                    
                    if info['metrics']:
                        f.write("Metrics:\n")
                        for key, value in info['metrics'].items():
                            f.write(f"  {key}: {value:.6f}\n")
                    f.write("\n" + "-" * 30 + "\n\n")
            
            logging.info(f"Checkpoint summary saved: {summary_path}")
            
        except Exception as e:
            logging.warning(f"Failed to create checkpoint summary: {e}")