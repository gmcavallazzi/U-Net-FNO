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


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(min(4, F_int), F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(min(4, F_int), F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


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
    def __init__(self, in_channels, out_channels, modes=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.spectral = SpectralConv2d(out_channels, out_channels, modes, modes)
        self.local = nn.Conv2d(out_channels, out_channels, 1)
        self.residual = SimpleResBlock(out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        
    def forward(self, x):
        x = F.gelu(self.norm(self.conv(x)))
        x1 = self.spectral(x)
        x2 = self.local(x)
        x = F.gelu(x1 + x2)
        x = self.residual(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, modes=16, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        
        if use_attention and skip_channels > 0:
            self.attention = AttentionGate(out_channels, skip_channels, skip_channels // 2)
        
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
            
            if self.use_attention:
                skip = self.attention(x, skip)
            
            x = torch.cat([x, skip], dim=1)
        
        x = F.gelu(self.norm(self.conv(x)))
        x1 = self.spectral(x)
        x2 = self.local(x)
        x = F.gelu(x1 + x2)
        x = self.residual(x)
        return x


class UNetFNO(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_channels=96, 
                 depth=3, modes=16, enforce_incompressible=True, use_attention=True):
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
            out_ch = min(384, channels * 2) if i < depth - 1 else channels
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
            self.up_blocks.append(UpBlock(current_channels, out_ch, skip_ch, modes, use_attention))
            current_channels = out_ch
        
        # Output head with residual connection
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.GroupNorm(min(4, base_channels // 2), base_channels // 2),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, base_channels // 4, 3, padding=1),
            nn.GroupNorm(min(4, base_channels // 4), base_channels // 4),
            nn.GELU(),
            nn.Conv2d(base_channels // 4, out_channels, 1)
        )
        
        # Residual connection for output
        self.output_residual = nn.Conv2d(base_channels, out_channels, 1)
        
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
        
        # Output with residual connection
        residual = self.output_residual(x)
        output = self.output_conv(x)
        output = output + residual
        
        # Use softsign for better peak preservation
        return torch.softsign(output)


def compute_multiscale_physics_losses(pred_velocity, true_velocity, input_data):
    """Multi-scale physics losses with peak preservation"""
    losses = {}
    device = pred_velocity.device
    
    # Extract components
    pred_u, pred_v = pred_velocity[:, 0:1], pred_velocity[:, 1:2]
    true_u, true_v = true_velocity[:, 0:1], true_velocity[:, 1:2]
    
    # Multi-scale MSE loss
    mse_loss = 0
    scales = [1.0, 0.5, 0.25]
    scale_weights = [1.0, 0.5, 0.25]
    
    for scale, weight in zip(scales, scale_weights):
        if scale < 1.0:
            size = int(pred_velocity.shape[-1] * scale)
            pred_scaled = F.interpolate(pred_velocity, size=(size, size), mode='bilinear', align_corners=False)
            true_scaled = F.interpolate(true_velocity, size=(size, size), mode='bilinear', align_corners=False)
        else:
            pred_scaled = pred_velocity
            true_scaled = true_velocity
        
        mse_loss += weight * F.mse_loss(pred_scaled, true_scaled)
    
    losses['mse'] = mse_loss / sum(scale_weights)
    
    # Weighted MSE emphasizing high-velocity regions
    velocity_mag = torch.sqrt(true_u**2 + true_v**2 + 1e-6)
    velocity_weights = 1 + 2 * (velocity_mag / (velocity_mag.max() + 1e-6))
    losses['weighted_mse'] = torch.mean(velocity_weights * (pred_velocity - true_velocity)**2)
    
    # Huber loss for peak preservation
    huber_delta = 0.1
    diff = pred_velocity - true_velocity
    abs_diff = torch.abs(diff)
    huber_loss = torch.where(abs_diff <= huber_delta, 
                            0.5 * diff**2, 
                            huber_delta * (abs_diff - 0.5 * huber_delta))
    losses['huber'] = torch.mean(huber_loss)
    
    # L1 loss on velocity magnitude for peak preservation
    pred_mag = torch.sqrt(pred_u**2 + pred_v**2 + 1e-6)
    true_mag = torch.sqrt(true_u**2 + true_v**2 + 1e-6)
    losses['magnitude_l1'] = F.l1_loss(pred_mag, true_mag)
    
    # Divergence loss with Sobel gradients
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                          device=device, dtype=pred_u.dtype) / 8.0
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                          device=device, dtype=pred_u.dtype) / 8.0
    
    du_dx = F.conv2d(pred_u, sobel_x.expand(pred_u.size(1), 1, -1, -1), 
                     padding=1, groups=pred_u.size(1))
    dv_dy = F.conv2d(pred_v, sobel_y.expand(pred_v.size(1), 1, -1, -1), 
                     padding=1, groups=pred_v.size(1))
    
    divergence = du_dx + dv_dy
    losses['divergence'] = torch.mean(divergence ** 2)
    
    # Boundary loss
    losses['boundary'] = (
        F.mse_loss(pred_velocity[:, :, 0, 0], pred_velocity[:, :, -1, -1]) +
        F.mse_loss(pred_velocity[:, :, 0, -1], pred_velocity[:, :, -1, 0])
    )
    
    # Frequency domain loss
    pred_fft = torch.fft.rfft2(pred_velocity)
    true_fft = torch.fft.rfft2(true_velocity)
    losses['frequency'] = F.mse_loss(torch.abs(pred_fft), torch.abs(true_fft))
    
    return losses


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)