import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SpectralConv2d(nn.Module):
    """Fourier layer for capturing global dependencies"""
    
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))

    def complex_mult(self, input, weights):
        weights_complex = torch.complex(weights[..., 0], weights[..., 1])
        return torch.einsum("bixy,ioxy->boxy", input, weights_complex)

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                           device=x.device, dtype=torch.cfloat)
        
        modes1 = min(self.modes1, x_ft.shape[-2])
        modes2 = min(self.modes2, x_ft.shape[-1])
        
        out_ft[:, :, :modes1, :modes2] = self.complex_mult(
            x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2])
        
        out_ft[:, :, -modes1:, :modes2] = self.complex_mult(
            x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, :modes1, :modes2])
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNOBlock(nn.Module):
    """FNO block with spectral and local convolutions"""
    
    def __init__(self, channels, modes1, modes2):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.local_conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)
        out = x1 + x2
        return self.activation(self.norm(out))


class ResidualBlock(nn.Module):
    """Simple residual block with normalization"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class DownBlock(nn.Module):
    """Downsampling block with FNO and residual components"""
    
    def __init__(self, in_channels, out_channels, modes=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.fno = FNOBlock(out_channels, modes, modes)
        self.residual = ResidualBlock(out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        
    def forward(self, x):
        x = F.gelu(self.norm(self.conv(x)))
        x = self.fno(x)
        x = self.residual(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with FNO and skip connections"""
    
    def __init__(self, in_channels, out_channels, skip_channels, modes=8):
        super().__init__()
        # Upsample and reduce channels to out_channels 
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        # After concatenation with skip, total channels = out_channels + skip_channels
        concat_channels = out_channels + skip_channels if skip_channels > 0 else out_channels
        # Process concatenated features
        self.conv = nn.Conv2d(concat_channels, out_channels, 3, padding=1)
        self.fno = FNOBlock(out_channels, modes, modes)
        self.residual = ResidualBlock(out_channels)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.skip_channels = skip_channels
        
    def forward(self, x, skip):
        x = self.upsample(x)
        
        if self.skip_channels > 0 and skip is not None:
            # Handle size mismatch without tensor-to-boolean conversion
            x_h, x_w = x.shape[-2:]
            skip_h, skip_w = skip.shape[-2:]
            if x_h != skip_h or x_w != skip_w:
                x = F.interpolate(x, size=(skip_h, skip_w), mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
        
        x = F.gelu(self.norm(self.conv(x)))
        x = self.fno(x)
        x = self.residual(x)
        return x


class UNetFNO(nn.Module):
    """U-Net with Fourier Neural Operator components for Navier-Stokes equations"""
    
    def __init__(self, in_channels=2, out_channels=2, base_channels=64, 
                 depth=4, modes=16, enforce_incompressible=True):
        super().__init__()
        
        self.depth = depth
        self.enforce_incompressible = enforce_incompressible
        
        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.GELU()
        )
        
        # Encoder - track channel progression
        self.down_blocks = nn.ModuleList()
        self.skip_channels = []  # Track skip connection channel counts
        channels = base_channels
        for i in range(depth):
            out_ch = min(512, channels * 2) if i < depth - 1 else channels
            self.down_blocks.append(DownBlock(channels, out_ch, modes))
            # All down blocks produce skip connections (used in reverse order in decoder)
            self.skip_channels.append(out_ch)
            channels = out_ch
        
        # Bottleneck with enhanced FNO processing
        self.bottleneck = nn.Sequential(
            FNOBlock(channels, modes, modes),
            ResidualBlock(channels),
            FNOBlock(channels, modes, modes)
        )
        
        # Decoder - use skip connection channels in reverse order
        self.up_blocks = nn.ModuleList()
        current_channels = channels
        # Skip connections: all encoder outputs in reverse order
        skip_channels_for_decoder = self.skip_channels[::-1]  # Reverse order
        
        for i in range(depth):
            # Output channels: progressively reduce back to base_channels
            if i == depth - 1:
                out_ch = base_channels
            else:
                out_ch = max(base_channels, current_channels // 2)
            
            # Skip channels: from encoder outputs in reverse order
            skip_ch = skip_channels_for_decoder[i] if i < len(skip_channels_for_decoder) else 0
            
            self.up_blocks.append(UpBlock(current_channels, out_ch, skip_ch, modes))
            current_channels = out_ch
        
        # Output head
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.GroupNorm(min(4, base_channels // 2), base_channels // 2),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, out_channels, 1)
        )
        
        # Divergence-free projection for incompressible flow
        if enforce_incompressible:
            self.div_projection = DivergenceFreeProjection()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input processing
        x = self.input_conv(x)
        
        # Encoder
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_connections[-(i+1)] if i < len(skip_connections) else None
            x = up_block(x, skip)
        
        # Output
        x = self.output_conv(x)
        
        # Apply divergence-free constraint for incompressible flow
        if self.enforce_incompressible and hasattr(self, 'div_projection'):
            x = self.div_projection(x)
        
        return torch.tanh(x)


class DivergenceFreeProjection(nn.Module):
    """Soft divergence-free projection for incompressible flow"""
    
    def __init__(self, strength=0.1):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(strength))
        
        # Gradient operators
        self.register_buffer('grad_x', torch.tensor([[[-0.5, 0, 0.5]]], dtype=torch.float32))
        self.register_buffer('grad_y', torch.tensor([[[-0.5], [0], [0.5]]], dtype=torch.float32))
    
    def compute_divergence(self, u, v):
        """Compute divergence using central differences"""
        du_dx = F.conv2d(u, self.grad_x.expand(u.size(1), 1, -1, -1), 
                        padding=(0, 1), groups=u.size(1))
        dv_dy = F.conv2d(v, self.grad_y.expand(v.size(1), 1, -1, -1), 
                        padding=(1, 0), groups=v.size(1))
        return du_dx + dv_dy
    
    def forward(self, velocity_field):
        u, v = velocity_field[:, 0:1, :, :], velocity_field[:, 1:2, :, :]
        
        # Compute divergence
        div = self.compute_divergence(u, v)
        
        # Soft correction
        correction = torch.sigmoid(self.strength) * 0.1
        
        # Apply correction (simplified approach)
        u_corrected = u - correction * F.conv2d(div, self.grad_x.expand(1, 1, -1, -1), 
                                              padding=(0, 1))
        v_corrected = v - correction * F.conv2d(div, self.grad_y.expand(1, 1, -1, -1), 
                                              padding=(1, 0))
        
        return torch.cat([u_corrected, v_corrected], dim=1)


def compute_physics_losses(pred_velocity, true_velocity, input_pressure_wss):
    """Compute physics-informed losses for Navier-Stokes equations"""
    losses = {}
    
    # Extract components
    pred_u, pred_v = pred_velocity[:, 0:1], pred_velocity[:, 1:2]
    true_u, true_v = true_velocity[:, 0:1], true_velocity[:, 1:2]
    pressure = input_pressure_wss[:, 0:1]
    
    # MSE loss
    losses['mse'] = F.mse_loss(pred_velocity, true_velocity)
    
    # Divergence loss (incompressibility)
    grad_x = torch.tensor([[[-0.5, 0, 0.5]]], device=pred_u.device, dtype=pred_u.dtype)
    grad_y = torch.tensor([[[-0.5], [0], [0.5]]], device=pred_u.device, dtype=pred_u.dtype)
    
    du_dx = F.conv2d(pred_u, grad_x.expand(pred_u.size(1), 1, -1, -1), 
                     padding=(0, 1), groups=pred_u.size(1))
    dv_dy = F.conv2d(pred_v, grad_y.expand(pred_v.size(1), 1, -1, -1), 
                     padding=(1, 0), groups=pred_v.size(1))
    
    divergence = du_dx + dv_dy
    losses['divergence'] = torch.mean(divergence ** 2)
    
    # Periodic boundary conditions
    losses['boundary'] = (
        F.mse_loss(pred_u[:, :, 0, :], pred_u[:, :, -1, :]) +
        F.mse_loss(pred_u[:, :, :, 0], pred_u[:, :, :, -1]) +
        F.mse_loss(pred_v[:, :, 0, :], pred_v[:, :, -1, :]) +
        F.mse_loss(pred_v[:, :, :, 0], pred_v[:, :, :, -1])
    )
    
    # Spectral loss for energy cascade
    pred_fft = torch.fft.rfft2(pred_velocity)
    true_fft = torch.fft.rfft2(true_velocity)
    losses['spectral'] = F.mse_loss(torch.abs(pred_fft), torch.abs(true_fft))
    
    return losses


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)