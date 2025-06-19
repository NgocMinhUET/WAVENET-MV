"""
WaveletTransformCNN - Implementation theo đúng specification
PredictCNN: Conv3x3(64→64) + ReLU → Conv3x3(64→64) + ReLU → Conv1x1(64→C') → LH,HL,HH
UpdateCNN: [X‖H] → Conv3x3((64+3C')→64) + ReLU → Conv3x3(64→64) + ReLU → Conv1x1(64→C') → LL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PredictCNN(nn.Module):
    """Predict high-freq residual H = P(X)"""
    
    def __init__(self, input_channels=64, output_channels=64):
        super().__init__()
        
        self.predict_layers = nn.Sequential(
            # Conv3x3 64→64 + ReLU
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv3x3 64→64 + ReLU  
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv1x1 64→C' (3×C' for LH, HL, HH)
            nn.Conv2d(64, 3 * output_channels, kernel_size=1, stride=1, padding=0)
        )
        
        self.output_channels = output_channels
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 64, H, W]
        Returns:
            high_freq: [B, 3*C', H, W] - concatenated LH, HL, HH
        """
        return self.predict_layers(x)


class UpdateCNN(nn.Module):
    """Update low-freq base L = U(X, H)"""
    
    def __init__(self, input_channels=64, high_freq_channels=64, output_channels=64):
        super().__init__()
        
        # [X ‖ H] concatenation along channel dimension
        concat_channels = input_channels + 3 * high_freq_channels
        
        self.update_layers = nn.Sequential(
            # Conv3x3 (64+3C')→64 + ReLU
            nn.Conv2d(concat_channels, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
            
            # Conv3x3 64→64 + ReLU
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv1x1 64→C' (1×C' for LL)
            nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x, h):
        """
        Args:
            x: Input tensor [B, 64, H, W]
            h: High frequency [B, 3*C', H, W]  
        Returns:
            low_freq: [B, C', H, W] - LL component
        """
        # Concatenate X and H along channel dimension
        concat_input = torch.cat([x, h], dim=1)
        return self.update_layers(concat_input)


class WaveletTransformCNN(nn.Module):
    """
    Complete Wavelet Transform CNN theo specification
    Output: cat(LL, LH, HL, HH) = 4×C' channels
    """
    
    def __init__(self, input_channels=3, feature_channels=64, wavelet_channels=64):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.wavelet_channels = wavelet_channels
        
        # Stage-0: RGB → feature (Conv3x3 in=3 out=64 stride=1 pad=1 + ReLU)
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Predict branch - predict high-freq residual
        self.predict_cnn = PredictCNN(
            input_channels=feature_channels,
            output_channels=wavelet_channels
        )
        
        # Update branch - update low-freq base  
        self.update_cnn = UpdateCNN(
            input_channels=feature_channels,
            high_freq_channels=wavelet_channels,
            output_channels=wavelet_channels
        )
        
    def forward(self, x):
        """
        Forward pass theo specification
        Args:
            x: Input image [B, 3, H, W]
        Returns:
            wavelet_coeffs: [B, 4*C', H, W] - cat(LL, LH, HL, HH)
        """
        # Stage-0: RGB → feature  
        features = self.input_conv(x)  # [B, 64, H, W]
        
        # Predict branch: H = P(X)
        high_freq = self.predict_cnn(features)  # [B, 3*C', H, W] - LH, HL, HH
        
        # Update branch: L = U(X, H)  
        low_freq = self.update_cnn(features, high_freq)  # [B, C', H, W] - LL
        
        # Output: cat(LL, LH, HL, HH) = 4×C' channels
        wavelet_coeffs = torch.cat([low_freq, high_freq], dim=1)  # [B, 4*C', H, W]
        
        return wavelet_coeffs
    
    def get_coefficients(self, x):
        """
        Get separated wavelet coefficients
        Returns:
            low_freq (LL): [B, C', H, W]
            high_freq (LH,HL,HH): [B, 3*C', H, W]
        """
        features = self.input_conv(x)
        high_freq = self.predict_cnn(features)
        low_freq = self.update_cnn(features, high_freq)
        return low_freq, high_freq
    
    def inverse_transform(self, wavelet_coeffs):
        """
        Inverse wavelet transform để reconstruction
        Args:
            wavelet_coeffs: [B, 4*C', H, W]
        Returns:
            reconstructed: [B, 3, H, W]
        """
        # Split coefficients
        C = self.wavelet_channels
        low_freq = wavelet_coeffs[:, :C]  # LL
        high_freq = wavelet_coeffs[:, C:]  # LH, HL, HH
        
        # Simple inverse - có thể cải thiện với learnable inverse
        # Reconstruct features từ wavelet coefficients
        reconstructed_features = low_freq + torch.sum(high_freq.view(-1, 3, C, *high_freq.shape[2:]), dim=1)
        
        # Project back to RGB space
        rgb_reconstruction = nn.Conv2d(C, 3, 3, padding=1).to(wavelet_coeffs.device)
        reconstructed = torch.tanh(rgb_reconstruction(reconstructed_features))
        
        return reconstructed


def test_wavelet_transform_cnn():
    """Unit test for WaveletTransformCNN"""
    model = WaveletTransformCNN(input_channels=3, feature_channels=64, wavelet_channels=64)
    
    # Test forward pass
    x = torch.randn(2, 3, 128, 128)
    wavelet_coeffs = model(x)
    
    # Check output shape
    expected_shape = (2, 4 * 64, 128, 128)  # 4*C' channels
    assert wavelet_coeffs.shape == expected_shape, f"Expected {expected_shape}, got {wavelet_coeffs.shape}"
    
    # Test coefficient separation
    low_freq, high_freq = model.get_coefficients(x)
    assert low_freq.shape == (2, 64, 128, 128), f"LL shape mismatch: {low_freq.shape}"
    assert high_freq.shape == (2, 3*64, 128, 128), f"High freq shape mismatch: {high_freq.shape}"
    
    print("✓ WaveletTransformCNN tests passed!")


if __name__ == "__main__":
    test_wavelet_transform_cnn() 