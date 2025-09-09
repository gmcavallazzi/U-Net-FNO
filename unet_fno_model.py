import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d(nn.Module):
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


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        
    def forward(self, x):
        residual = x
        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.gelu(out + residual)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.spectral = SpectralConv2d(out_channels, out_channels, modes, modes)
        self.local = nn.Conv2d(out_channels, out_channels, 1)
        self.residual = SimpleResBlock(out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        
    def forward(self, x):
        x = F.gelu(self.norm(self.conv(x)))
        # FNO block
        x1 = self.spectral(x)
        x2 = self.local(x)
        x = F.gelu(x1 + x2)
        x = self.residual(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, modes=8):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        concat_channels = out_channels + skip_channels if skip_channels > 0 else out_channels
        self.conv = nn.Conv2d(concat_channels, out_channels, 3, padding=1)
        self.spectral = SpectralConv2d(out_channels, out_channels, modes, modes)
        self.local = nn.Conv2d(out_channels, out_channels, 1)
        self.residual = SimpleResBlock(out_channels)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.skip_channels = skip_channels
        
    def forward(self, x, skip):
        x = self.upsample(x)
        
        if self.skip_channels > 0 and skip is not None:
            x_h, x_w = x.shape[-2:]
            skip_h, skip_w = skip.shape[-2:]
            if x_h != skip_h or x_w != skip_w:
                x = F.interpolate(x, size=(skip_h, skip_w), mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = F.gelu(self.norm(self.conv(x)))
        # FNO block
        x1 = self.spectral(x)
        x2 = self.local(x)
        x = F.gelu(x1 + x2)
        x = self.residual(x)
        return x


class UNetFNO(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_channels=64, 
                 depth=3, modes=12, enforce_incompressible=True):
        super().__init__()
        
        self.depth = depth
        self.enforce_incompressible = enforce_incompressible
        
        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.GELU()
        )
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        self.skip_channels = []
        channels = base_channels
        for i in range(depth):
            out_ch = min(256, channels * 2) if i < depth - 1 else channels
            self.down_blocks.append(DownBlock(channels, out_ch, modes))
            self.skip_channels.append(out_ch)
            channels = out_ch
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            SpectralConv2d(channels, channels, modes, modes),
            SimpleResBlock(channels)
        )
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        current_channels = channels
        skip_channels_for_decoder = self.skip_channels[::-1]
        
        for i in range(depth):
            if i == depth - 1:
                out_ch = base_channels
            else:
                out_ch = max(base_channels, current_channels // 2)
            
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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
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
        return torch.tanh(x)


def compute_physics_losses(pred_velocity, true_velocity, input_data):
    """Simplified physics losses - removed problematic spectral loss"""
    losses = {}
    
    # Extract components
    pred_u, pred_v = pred_velocity[:, 0:1], pred_velocity[:, 1:2]
    
    # MSE loss (main reconstruction loss)
    losses['mse'] = F.mse_loss(pred_velocity, true_velocity)
    
    # Divergence loss (incompressibility) with proper gradient computation
    # Use Sobel operators for more stable gradients
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                          device=pred_u.device, dtype=pred_u.dtype) / 8.0
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                          device=pred_u.device, dtype=pred_u.dtype) / 8.0
    
    du_dx = F.conv2d(pred_u, sobel_x.expand(pred_u.size(1), 1, -1, -1), 
                     padding=1, groups=pred_u.size(1))
    dv_dy = F.conv2d(pred_v, sobel_y.expand(pred_v.size(1), 1, -1, -1), 
                     padding=1, groups=pred_v.size(1))
    
    divergence = du_dx + dv_dy
    losses['divergence'] = torch.mean(divergence ** 2)
    
    # Simplified boundary loss (just corners to avoid over-constraining)
    losses['boundary'] = (
        F.mse_loss(pred_velocity[:, :, 0, 0], pred_velocity[:, :, -1, -1]) +
        F.mse_loss(pred_velocity[:, :, 0, -1], pred_velocity[:, :, -1, 0])
    )
    
    return losses


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)