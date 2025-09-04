#!/usr/bin/env python3
"""
Quick test script to verify U-Net-FNO model architecture
"""

import torch
from unet_fno_model import UNetFNO, count_parameters


def test_model_forward():
    """Test model forward pass with different configurations"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    # Test configurations
    configs = [
        {'base_channels': 32, 'depth': 3, 'modes': 8},
        {'base_channels': 64, 'depth': 4, 'modes': 16},
        {'base_channels': 96, 'depth': 4, 'modes': 32},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{'='*50}")
        print(f"Testing Configuration {i+1}: {config}")
        print('='*50)
        
        try:
            # Create model
            model = UNetFNO(
                in_channels=2,
                out_channels=2,
                **config,
                enforce_incompressible=True
            ).to(device)
            
            # Print model info
            total_params = count_parameters(model)
            print(f"Total parameters: {total_params:,}")
            
            # Test forward pass
            batch_size = 2
            input_tensor = torch.randn(batch_size, 2, 64, 64, device=device)
            print(f"Input shape: {input_tensor.shape}")
            
            with torch.no_grad():
                output = model(input_tensor)
                print(f"Output shape: {output.shape}")
                
                # Check output range (should be in [-1, 1] due to tanh)
                print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
                
                # Check for NaN or Inf
                if torch.isnan(output).any():
                    print("WARNING: Output contains NaN!")
                elif torch.isinf(output).any():
                    print("WARNING: Output contains Inf!")
                else:
                    print("PASS: Forward pass successful!")
            
            # Test with different input sizes
            print("\nTesting different input sizes:")
            for size in [32, 64, 128]:
                if size <= 128:  # Skip large sizes for memory
                    test_input = torch.randn(1, 2, size, size, device=device)
                    try:
                        with torch.no_grad():
                            test_output = model(test_input)
                        print(f"  Size {size}x{size}: PASS - Output {test_output.shape}")
                    except Exception as e:
                        print(f"  Size {size}x{size}: FAIL - {e}")
                        
        except Exception as e:
            print(f"FAIL: Configuration {i+1} failed - {e}")
            continue


def test_channel_progression():
    """Test and display channel progression through the network"""
    
    print(f"\n{'='*50}")
    print("Channel Progression Analysis")
    print('='*50)
    
    base_channels = 64
    depth = 4
    
    # Calculate channel progression
    channels = [base_channels]
    for i in range(1, depth):
        next_ch = min(512, channels[-1] * 2)
        channels.append(next_ch)
    
    print(f"Base channels: {base_channels}")
    print(f"Depth: {depth}")
    print(f"Channel progression: {channels}")
    
    print("\nEncoder (Down blocks):")
    for i in range(depth - 1):
        in_ch = channels[i]
        out_ch = channels[i + 1]
        print(f"  Down Block {i+1}: {in_ch} → {out_ch}")
    
    print(f"\nBottleneck: {channels[-1]} channels")
    
    print("\nDecoder (Up blocks):")
    for i in range(depth - 1):
        in_ch = channels[depth - 1 - i]
        skip_ch = channels[depth - 2 - i]
        out_ch = channels[depth - 2 - i]
        concat_ch = out_ch + skip_ch  # After upsampling and concatenation
        print(f"  Up Block {i+1}: {in_ch} → {out_ch} (upsample), + {skip_ch} (skip) = {concat_ch} (concat) → {out_ch}")


def debug_channel_flow():
    """Debug channel flow in U-Net-FNO"""
    # Test configuration
    config = {'base_channels': 32, 'depth': 3, 'modes': 8}
    
    print(f"\n{'='*50}")
    print("Channel Flow Debug")
    print('='*50)
    print(f"Testing config: {config}")
    print("="*50)
    
    # Create model
    model = UNetFNO(
        in_channels=2,
        out_channels=2,
        **config,
        enforce_incompressible=True
    )
    
    print(f"Skip channels tracked: {model.skip_channels}")
    
    # Print encoder structure
    print("\nEncoder structure:")
    for i, down_block in enumerate(model.down_blocks):
        in_ch = down_block.conv.in_channels
        out_ch = down_block.conv.out_channels
        print(f"  Down {i}: {in_ch} → {out_ch}")
    
    print(f"\nBottleneck channels: {model.bottleneck[0].spectral_conv.in_channels}")
    
    # Print decoder structure
    print("\nDecoder structure:")
    skip_channels_for_decoder = model.skip_channels[::-1]
    print(f"Skip channels for decoder: {skip_channels_for_decoder}")
    
    for i, up_block in enumerate(model.up_blocks):
        upsample_in = up_block.upsample.in_channels
        upsample_out = up_block.upsample.out_channels
        conv_in = up_block.conv.in_channels
        conv_out = up_block.conv.out_channels
        skip_ch = up_block.skip_channels
        expected_concat = upsample_out + skip_ch if skip_ch > 0 else upsample_out
        print(f"  Up {i}: upsample {upsample_in}→{upsample_out}, skip={skip_ch}, expected_concat={expected_concat}, conv {conv_in}→{conv_out}")
        if conv_in != expected_concat:
            print(f"    ERROR: Conv expects {conv_in} but will get {expected_concat}")
    
    # Test with actual forward pass to see real channel flow
    print("\nTesting actual forward pass to see channel dimensions:")
    try:
        x = torch.randn(1, 2, 64, 64)
        x = model.input_conv(x)
        print(f"After input_conv: {x.shape}")
        
        skip_connections = []
        for i, down_block in enumerate(model.down_blocks):
            x, skip = down_block(x)
            skip_connections.append(skip)
            print(f"After down {i}: x={x.shape}, skip={skip.shape}")
        
        x = model.bottleneck(x)
        print(f"After bottleneck: {x.shape}")
        
        for i, up_block in enumerate(model.up_blocks):
            skip = skip_connections[-(i+1)] if i < len(skip_connections) else None
            skip_shape = skip.shape if skip is not None else None
            print(f"Before up {i}: x={x.shape}, skip={skip_shape}")
            x = up_block(x, skip)
            print(f"After up {i}: x={x.shape}")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")


def main():
    """Run all tests"""
    
    print("U-Net-FNO Model Architecture Test")
    print("="*50)
    
    # Debug channel flow first
    debug_channel_flow()
    
    # Test model forward passes
    test_model_forward()
    
    print(f"\n{'='*50}")
    print("All tests completed!")


def test_channel_progression():
    """Test and display channel progression through the network"""
    
    print(f"\n{'='*50}")
    print("Channel Progression Analysis")
    print('='*50)
    
    base_channels = 64
    depth = 4
    
    # Calculate channel progression
    channels = [base_channels]
    for i in range(1, depth):
        next_ch = min(512, channels[-1] * 2)
        channels.append(next_ch)
    
    print(f"Base channels: {base_channels}")
    print(f"Depth: {depth}")
    print(f"Channel progression: {channels}")
    
    print("\nEncoder (Down blocks):")
    for i in range(depth - 1):
        in_ch = channels[i]
        out_ch = channels[i + 1]
        print(f"  Down Block {i+1}: {in_ch} → {out_ch}")
    
    print(f"\nBottleneck: {channels[-1]} channels")
    
    print("\nDecoder (Up blocks):")
    for i in range(depth - 1):
        in_ch = channels[depth - 1 - i]
        skip_ch = channels[depth - 2 - i]
        out_ch = channels[depth - 2 - i]
        concat_ch = out_ch + skip_ch  # After upsampling and concatenation
        print(f"  Up Block {i+1}: {in_ch} → {out_ch} (upsample), + {skip_ch} (skip) = {concat_ch} (concat) → {out_ch}")


if __name__ == "__main__":
    main()