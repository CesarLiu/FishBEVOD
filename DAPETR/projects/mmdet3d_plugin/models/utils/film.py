# film.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class FiLM2d(nn.Module):
    def __init__(self, ray_ch: int, feat_ch: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(ray_ch, feat_ch*2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch*2, feat_ch*2, 1, bias=True)
        )
        # small init so γ,β start near identity/zero
        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Safer modulation scales and lightweight debug controls
        self.gamma_scale = 0.1
        self.beta_scale = 0.1

    def forward(self, feat, ray_feat):
        # Run the conditioning in full precision to avoid overflow/underflow
        with autocast(enabled=False):
            rf = ray_feat.float()
            gb = self.mlp(rf)                      # fp32
            gamma, beta = gb.chunk(2, 1)
            # bound them to safe ranges
            gamma = torch.tanh(gamma) * 0.5 + 1.0  # ~[0.5, 1.5]
            beta  = torch.tanh(beta)  * 0.1        # ~[-0.1, 0.1]

        # Cast back to the feature dtype right before applying
        gamma = gamma.to(feat.dtype)
        beta  = beta.to(feat.dtype)
        out = gamma * feat + beta
        return out
    
class MultiModalDistortionFiLM(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # Separate encoders for each geometric property
        self.theta_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 1), nn.ReLU(),
            nn.Conv2d(64, feature_dim, 1)
        )
        
        self.zs_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 1), nn.ReLU(), 
            nn.Conv2d(64, feature_dim, 1)
        )
        
        self.lambda_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 1), nn.ReLU(),
            nn.Conv2d(64, feature_dim, 1)
        )
        
        # Fusion network for combined modulation
        self.fusion_net = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim * 2, 1),  # → γ and β
            nn.ReLU()
        )
        
    def forward(self, features, theta, zs, lambda_field):
        # Enhanced FP16 safety: force fisheye feature processing in FP32
        with autocast(enabled=False):
            # Convert fisheye features to FP32 for stability
            theta = theta.float()
            zs = zs.float() 
            lambda_field = lambda_field.float()
            
            # Clamp inputs to prevent overflow in any precision
            theta = torch.clamp(theta, -10, 10)
            zs = torch.clamp(zs, -10, 10) 
            lambda_field = torch.clamp(lambda_field, -10, 10)
            
            # Process each channel separately using conv2d
            theta_emb = self.theta_encoder(theta)           # Angular coverage
            zs_emb = self.zs_encoder(zs)                   # Depth component  
            lambda_emb = self.lambda_encoder(lambda_field)  # MEI scaling
            
            # Combine embeddings with numerical stability
            combined = torch.cat([theta_emb, zs_emb, lambda_emb], dim=1)
            gamma_beta = self.fusion_net(combined)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        
        # Apply safe modulation bounds with extra stability
        gamma = torch.tanh(torch.clamp(gamma, -5, 5)) * 0.5 + 1.0  # Range [0.5, 1.5]
        beta = torch.tanh(torch.clamp(beta, -5, 5)) * 0.1           # Range [-0.1, 0.1]
        
        # Final safety: check for NaN/Inf and fallback to identity
        result = gamma * features + beta
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("FiLM modulation produced NaN/Inf, skipping modulation")
            return features  # Fallback to identity transformation
        return result

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
    
def make_ray_pyramid(ray5, sizes):
    """
    ray5: (B, 5, H, W)  = [cosφ, sinφ, cosθ, sinθ, log_area]
    sizes: list of (H_i, W_i) to resize for each consumer
    returns: [ray5_i ...] in same order as sizes
    """
    return [F.interpolate(ray5, size=s, mode='bilinear', align_corners=False) for s in sizes]
