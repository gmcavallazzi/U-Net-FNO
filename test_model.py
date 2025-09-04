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


def main():
    """Run all tests"""
    
    print("U-Net-FNO Model Architecture Test")
    print("="*50)
    
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