"""
AdaMixNet - Adaptive Mixing Network 
Input: (B, 4C', H, W) → (B, C_mix=128, H, W)
N=4 parallel filters + softmax attention mixing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaMixNet(nn.Module):
    """
    Adaptive Mixing Network theo specification
    - N=4 parallel filters: Conv3x3((4C'/N)→(C'/2)) + ReLU  
    - Attention: Conv3x3(4C'→64) + ReLU → Conv1x1(64→N) → Softmax
    - Mixing: Y = Σᵢ wᵢ(x)·Fᵢ(x)
    """
    
    def __init__(self, input_channels=256, C_prime=64, C_mix=128, N=4):
        super().__init__()
        
        self.input_channels = input_channels  # 4*C'
        self.C_prime = C_prime
        self.C_mix = C_mix
        self.N = N
        
        # Validate input channels
        if input_channels != 4 * C_prime:
            raise ValueError(f"input_channels ({input_channels}) should be 4 * C_prime ({4 * C_prime})")
        
        # N=4 parallel filters
        self.parallel_filters = nn.ModuleList()
        channels_per_group = input_channels // N  # 4C'/N = C'
        output_per_filter = C_prime // 2  # C'/2
        
        for i in range(N):
            filter_block = nn.Sequential(
                # Conv3x3 (4C'/N)→(C'/2) + ReLU
                nn.Conv2d(channels_per_group, output_per_filter, 
                         kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
            self.parallel_filters.append(filter_block)
        
        # Attention mechanism
        self.attention_cnn = nn.Sequential(
            # Conv3x3 4C'→64 + ReLU
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv1x1 64→N (logits for N filters)
            nn.Conv2d(64, N, kernel_size=1, stride=1, padding=0)
        )
        
        # Channel reduction (optional) - từ C'/2 → C_mix
        self.output_projection = nn.Conv2d(output_per_filter, C_mix, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor [B, 4*C', H, W] từ wavelet coefficients
        Returns:
            y: Mixed output [B, C_mix, H, W]
        """
        B, C, H, W = x.shape
        
        if C != self.input_channels:
            raise ValueError(f"Input channels {C} != expected {self.input_channels}")
        
        # Split input into N groups for parallel processing
        channels_per_group = C // self.N
        filter_outputs = []
        
        for i in range(self.N):
            # Extract group i
            start_idx = i * channels_per_group
            end_idx = (i + 1) * channels_per_group
            group_input = x[:, start_idx:end_idx]  # [B, C'/1, H, W]
            
            # Apply filter i
            filter_output = self.parallel_filters[i](group_input)  # [B, C'/2, H, W]
            filter_outputs.append(filter_output)
        
        # Stack filter outputs for easier processing
        stacked_outputs = torch.stack(filter_outputs, dim=1)  # [B, N, C'/2, H, W]
        
        # Compute attention weights
        attention_logits = self.attention_cnn(x)  # [B, N, H, W]
        attention_weights = F.softmax(attention_logits, dim=1)  # Softmax over N filters
        
        # Expand attention weights to match filter output dimensions
        attention_weights = attention_weights.unsqueeze(2)  # [B, N, 1, H, W]
        
        # Weighted mixing: Y = Σᵢ wᵢ(x) · Fᵢ(x)
        mixed_output = torch.sum(attention_weights * stacked_outputs, dim=1)  # [B, C'/2, H, W]
        
        # Channel reduction to C_mix
        final_output = self.output_projection(mixed_output)  # [B, C_mix, H, W]
        
        return final_output
    
    def get_attention_weights(self, x):
        """
        Get attention weights for analysis
        Args:
            x: Input tensor [B, 4*C', H, W]
        Returns:
            attention_weights: [B, N, H, W]
        """
        attention_logits = self.attention_cnn(x)
        return F.softmax(attention_logits, dim=1)
    
    def get_filter_outputs(self, x):
        """
        Get individual filter outputs for analysis
        Args:
            x: Input tensor [B, 4*C', H, W]
        Returns:
            filter_outputs: List of [B, C'/2, H, W] tensors
        """
        B, C, H, W = x.shape
        channels_per_group = C // self.N
        filter_outputs = []
        
        for i in range(self.N):
            start_idx = i * channels_per_group
            end_idx = (i + 1) * channels_per_group
            group_input = x[:, start_idx:end_idx]
            filter_output = self.parallel_filters[i](group_input)
            filter_outputs.append(filter_output)
            
        return filter_outputs


class EnhancedAdaMixNet(AdaMixNet):
    """
    Enhanced AdaMixNet với additional features
    - Residual connections
    - Batch normalization
    - Dropout for regularization
    """
    
    def __init__(self, input_channels=256, C_prime=64, C_mix=128, N=4, dropout=0.1):
        super().__init__(input_channels, C_prime, C_mix, N)
        
        # Add batch normalization cho parallel filters
        self.filter_norms = nn.ModuleList()
        output_per_filter = C_prime // 2
        
        for i in range(N):
            norm = nn.BatchNorm2d(output_per_filter)
            self.filter_norms.append(norm)
        
        # Batch norm cho attention
        self.attention_norm = nn.BatchNorm2d(64)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout)
        
        # Residual projection (nếu cần)
        self.residual_proj = None
        if input_channels != C_mix:
            self.residual_proj = nn.Conv2d(input_channels, C_mix, 1)
    
    def forward(self, x):
        """Enhanced forward với residual connection"""
        B, C, H, W = x.shape
        
        # Original AdaMixNet forward
        channels_per_group = C // self.N
        filter_outputs = []
        
        for i in range(self.N):
            start_idx = i * channels_per_group
            end_idx = (i + 1) * channels_per_group
            group_input = x[:, start_idx:end_idx]
            
            # Apply filter với batch norm
            filter_output = self.parallel_filters[i](group_input)
            filter_output = self.filter_norms[i](filter_output)
            filter_outputs.append(filter_output)
        
        stacked_outputs = torch.stack(filter_outputs, dim=1)
        
        # Enhanced attention với batch norm
        attention_features = self.attention_cnn[0](x)  # Conv3x3
        attention_features = self.attention_norm(attention_features)
        attention_features = self.attention_cnn[1](attention_features)  # ReLU
        attention_logits = self.attention_cnn[2](attention_features)  # Conv1x1
        
        attention_weights = F.softmax(attention_logits, dim=1)
        attention_weights = attention_weights.unsqueeze(2)
        
        # Weighted mixing
        mixed_output = torch.sum(attention_weights * stacked_outputs, dim=1)
        mixed_output = self.dropout(mixed_output)
        
        # Output projection
        final_output = self.output_projection(mixed_output)
        
        # Residual connection (nếu có)
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
            final_output = final_output + residual
        
        return final_output


def test_adamixnet():
    """Unit test for AdaMixNet"""
    # Test standard AdaMixNet
    C_prime = 64
    input_channels = 4 * C_prime  # 256
    C_mix = 128
    
    model = AdaMixNet(input_channels=input_channels, C_prime=C_prime, C_mix=C_mix)
    
    # Test forward pass
    x = torch.randn(2, input_channels, 64, 64)
    output = model(x)
    
    # Check output shape
    expected_shape = (2, C_mix, 64, 64)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    # Test attention weights
    attention_weights = model.get_attention_weights(x)
    assert attention_weights.shape == (2, 4, 64, 64), f"Attention weights shape: {attention_weights.shape}"
    
    # Check softmax property (should sum to 1 across N dimension)
    attention_sum = torch.sum(attention_weights, dim=1)
    assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-6), "Attention weights don't sum to 1"
    
    # Test filter outputs
    filter_outputs = model.get_filter_outputs(x)
    assert len(filter_outputs) == 4, f"Expected 4 filter outputs, got {len(filter_outputs)}"
    for i, output in enumerate(filter_outputs):
        expected_filter_shape = (2, C_prime // 2, 64, 64)
        assert output.shape == expected_filter_shape, f"Filter {i} shape: {output.shape}"
    
    print("✓ AdaMixNet tests passed!")
    
    # Test enhanced version
    enhanced_model = EnhancedAdaMixNet(input_channels=input_channels, C_prime=C_prime, C_mix=C_mix)
    enhanced_output = enhanced_model(x)
    assert enhanced_output.shape == expected_shape, f"Enhanced output shape: {enhanced_output.shape}"
    
    print("✓ EnhancedAdaMixNet tests passed!")


if __name__ == "__main__":
    test_adamixnet() 