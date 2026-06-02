import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast

class MultiModalDistortionFiLM(nn.Module):
    def __init__(self, feature_dim, use_unified_encoder=True):
        super().__init__()
        self.use_unified_encoder = use_unified_encoder
        
        if use_unified_encoder:
            # Option 1: Unified encoder for 3-channel input (RECOMMENDED)
            self.unified_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),     # Larger kernel for cross-channel interaction
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),   # Spatial convolution
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, feature_dim * 2, 1), # 1x1 for final projection
                nn.Tanh()  # Bound outputs
            )
        else:
            # Option 2: Separate encoders (your current approach)
            def create_stable_encoder(feature_dim):
                return nn.Sequential(
                    nn.Conv2d(1, 32, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(64, feature_dim, 1),
                    nn.Tanh()
                )
            
            self.rays_encoder = create_stable_encoder(feature_dim)
            self.angles_encoder = create_stable_encoder(feature_dim) 
            self.radial_encoder = create_stable_encoder(feature_dim)
            
            # Fusion network
            self.fusion_net = nn.Sequential(
                nn.Conv2d(feature_dim * 3, feature_dim * 2, 1),
                nn.BatchNorm2d(feature_dim * 2),
                nn.Tanh()
            )
        
        # Initialize all weights conservatively
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for numerical stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Very conservative initialization
                nn.init.xavier_uniform_(m.weight, gain=0.05)  # Even smaller gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def preprocess_distortion_channels(self, rays, angles, radial):
        """Robust preprocessing based on actual data statistics"""
        # Normalize based on observed ranges from your data
        
        # Rays: Regular cameras [0.596, 1.0], Fisheye [-0.393, 1.0]
        rays = torch.clamp(rays, -0.5, 1.0)
        rays = rays / 1.5 

        # Angles: Regular cameras [0.0015, 0.932], Fisheye [-1.571, 1.571] (≈ ±π/2)
        angles = torch.clamp(angles, -1.6, 1.6)
        angles = angles / 1.6  # Normalize to [-1, 1]
        
        # Radial: Both types [0.0005, 1.0] - already well normalized
        radial = torch.clamp(radial, 0, 1.0)
        # Keep as [0, 1] - already in good range
        
        # Replace any remaining NaN/Inf values with neutral values
        rays = torch.where(torch.isnan(rays) | torch.isinf(rays), 
                          torch.full_like(rays, 1.0), rays)  # Mid-range
        angles = torch.where(torch.isnan(angles) | torch.isinf(angles), 
                           torch.zeros_like(angles), angles)  # Zero angle
        radial = torch.where(torch.isnan(radial) | torch.isinf(radial), 
                           torch.full_like(radial, 0.5), radial)  # Mid-range
        
        return rays, angles, radial
    
    def debug_input_stats(self, rays, angles, radial):
        """Print input statistics for debugging"""
        print(f"Rays stats: min={rays.min():.4f}, max={rays.max():.4f}, "
              f"mean={rays.mean():.4f}, std={rays.std():.4f}")
        print(f"Angles stats: min={angles.min():.4f}, max={angles.max():.4f}, "
              f"mean={angles.mean():.4f}, std={angles.std():.4f}")
        print(f"Radial stats: min={radial.min():.4f}, max={radial.max():.4f}, "
              f"mean={radial.mean():.4f}, std={radial.std():.4f}")
        
    def forward(self, features, rays, angles, radial, debug=False):
        # Enhanced FP16 safety: force ALL computations in FP32
        with autocast(enabled=False):
            # Convert all inputs to FP32 for stability
            features = features.float()
            rays = rays.float()
            angles = angles.float() 
            radial = radial.float()
            
            # Debug input statistics if requested
            if debug:
                print("=== FiLM Input Debug ===")
                self.debug_input_stats(rays, angles, radial)
            
            # Robust preprocessing
            try:
                rays, angles, radial = self.preprocess_distortion_channels(
                    rays, angles, radial)
            except Exception as e:
                print(f"FiLM preprocessing error: {e}")
                return features
            
            if self.use_unified_encoder:
                # Unified encoder approach
                try:
                    # Stack channels: [B, 3, H, W]
                    distortion_features = torch.cat([rays, angles, radial], dim=1)
                    gamma_beta = self.unified_encoder(distortion_features)
                    
                    if torch.isnan(gamma_beta).any() or torch.isinf(gamma_beta).any():
                        print("Unified encoder produced NaN/Inf, skipping modulation")
                        return features
                        
                except Exception as e:
                    print(f"Unified encoder error: {e}")
                    return features
            else:
                # Separate encoders approach
                try:
                    rays_emb = self.rays_encoder(rays)
                    if torch.isnan(rays_emb).any() or torch.isinf(rays_emb).any():
                        print("NaN/Inf in rays_encoder output")
                        return features
                        
                    angles_emb = self.angles_encoder(angles)
                    if torch.isnan(angles_emb).any() or torch.isinf(angles_emb).any():
                        print("NaN/Inf in angles_encoder output") 
                        return features
                        
                    radial_emb = self.radial_encoder(radial)
                    if torch.isnan(radial_emb).any() or torch.isinf(radial_emb).any():
                        print("NaN/Inf in radial_encoder output")
                        return features
                        
                except Exception as e:
                    print(f"FiLM encoder error: {e}")
                    return features
                
                # Combine embeddings with error checking
                try:
                    combined = torch.cat([rays_emb, angles_emb, radial_emb], dim=1)
                    gamma_beta = self.fusion_net(combined)
                    
                    if torch.isnan(gamma_beta).any() or torch.isinf(gamma_beta).any():
                        print("FiLM fusion produced NaN/Inf, skipping modulation")
                        return features
                        
                except Exception as e:
                    print(f"FiLM fusion error: {e}")
                    return features
            
            # Split and apply safe modulation (same for both approaches)
            try:
                gamma, beta = gamma_beta.chunk(2, dim=1)
                
                # More conservative modulation bounds
                gamma = torch.sigmoid(gamma) + 0.5  # Range [0.5, 1.5]
                beta = torch.tanh(beta) * 0.1       # Range [-0.1, 0.1]
                
                # Apply modulation
                result = gamma * features + beta
                
                # Final safety check
                if torch.isnan(result).any() or torch.isinf(result).any():
                    print("FiLM modulation produced NaN/Inf, skipping modulation")
                    return features
                    
                return result.to(features.dtype)
                
            except Exception as e:
                print(f"FiLM modulation error: {e}")
                return features

class MultiModalDistortionFiLM_FP16(nn.Module):
    """FP16-optimized version of MultiModalDistortionFiLM for faster training"""
    
    def __init__(self, feature_dim, use_unified_encoder=True, fp16_safe_ops=True):
        super().__init__()
        self.use_unified_encoder = use_unified_encoder
        self.fp16_safe_ops = fp16_safe_ops
        self.feature_dim = feature_dim
        
        # Input normalization parameters (learnable for adaptation)
        self.register_buffer('rays_scale', torch.tensor(1.5))
        self.register_buffer('rays_offset', torch.tensor(0.0))
        self.register_buffer('angles_scale', torch.tensor(1.6))
        self.register_buffer('radial_scale', torch.tensor(1.0))
        
        # Adaptive modulation bounds (learnable parameters)
        self.gamma_scale = nn.Parameter(torch.tensor(0.5))      # Controls gamma range width
        self.gamma_offset = nn.Parameter(torch.tensor(0.75))    # Controls gamma center point
        self.beta_scale = nn.Parameter(torch.tensor(0.05))      # Controls beta range width
        
        
        if use_unified_encoder:
            # Unified encoder with FP16-friendly architecture
            self.unified_encoder = nn.Sequential(
                # Use smaller intermediate dimensions for FP16 efficiency
                nn.Conv2d(3, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32, eps=1e-4),  # Larger eps for FP16
                nn.ReLU(inplace=True),
                
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64, eps=1e-4),
                nn.ReLU(inplace=True),
                
                # Final projection with bias for fine-tuning
                nn.Conv2d(64, feature_dim * 2, 1, bias=True)
            )
        else:
            # Separate encoders with shared architecture
            def create_fp16_encoder():
                return nn.Sequential(
                    nn.Conv2d(1, 16, 1, bias=False),  # Smaller channels for speed
                    nn.BatchNorm2d(16, eps=1e-4),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(16, 32, 1, bias=False),
                    nn.BatchNorm2d(32, eps=1e-4), 
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(32, feature_dim, 1, bias=True)
                )
            
            self.rays_encoder = create_fp16_encoder()
            self.angles_encoder = create_fp16_encoder()
            self.radial_encoder = create_fp16_encoder()
            
            # Lightweight fusion
            self.fusion_net = nn.Conv2d(feature_dim * 3, feature_dim * 2, 1, bias=True)
        
        # FP16-friendly weight initialization
        self._init_weights_fp16()
        
        # Debug counters
        self.forward_count = 0
        self.nan_count = 0
        
    def _init_weights_fp16(self):
        """FP16-friendly weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use smaller initialization values for FP16 stability
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                std = (1.0 / fan_in) ** 0.5 * 0.1  # Very conservative
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def preprocess_inputs_fp16(self, rays, angles, radial):
        """FP16-optimized input preprocessing"""
        if self.fp16_safe_ops:
            # Use clamping ranges that work well in FP16
            rays = torch.clamp(rays, -0.5, 1.0) / self.rays_scale
            angles = torch.clamp(angles, -1.5, 1.5) / self.angles_scale  
            radial = torch.clamp(radial, 0.0, 1.0) / self.radial_scale
            
            # Replace NaN/Inf with safe FP16 values
            rays = torch.where(torch.isfinite(rays), rays, torch.zeros_like(rays))
            angles = torch.where(torch.isfinite(angles), angles, torch.zeros_like(angles))
            radial = torch.where(torch.isfinite(radial), radial, torch.full_like(radial, 0.5))
        else:
            # Standard preprocessing
            rays = torch.clamp(rays, -0.5, 1.0) / 1.5
            angles = torch.clamp(angles, -1.6, 1.6) / 1.6
            radial = torch.clamp(radial, 0, 1.0)
        
        return rays, angles, radial
    
    def generate_modulation_params_fp16(self, encoded_features):
        """Generate gamma and beta with adaptive FP16-safe bounds"""
        # Split features
        gamma_raw, beta_raw = encoded_features.chunk(2, dim=1)
        
        if self.fp16_safe_ops:
            # Simple: just use sigmoid/tanh directly (they're already bounded!)
            gamma = torch.sigmoid(gamma_raw) * self.gamma_scale + self.gamma_offset  # [0.75, 1.25]
            beta = torch.tanh(beta_raw) * self.beta_scale             # [-0.05, 0.05]
            
        else:
            # Standard modulation bounds (fallback)
            gamma = torch.sigmoid(gamma_raw) + 0.5  # [0.5, 1.5]
            beta = torch.tanh(beta_raw) * 0.1       # [-0.1, 0.1]
        
        return gamma, beta
    
    @autocast(enabled=True)  # Enable mixed precision
    def forward(self, features, rays, angles, radial, debug=False):
        """FP16-optimized forward pass"""
        self.forward_count += 1
        
        try:
            # Keep inputs in their original precision (likely FP16)
            original_dtype = features.dtype
            
            # Preprocess inputs (stays in FP16 if enabled)
            rays, angles, radial = self.preprocess_inputs_fp16(rays, angles, radial)
            
            # Check for NaN in preprocessed inputs
            if not (torch.isfinite(rays).all() and torch.isfinite(angles).all() and torch.isfinite(radial).all()):
                if debug:
                    print(f"[FiLM FP16] NaN detected in inputs at forward #{self.forward_count}")
                self.nan_count += 1
                return features
            
            if self.use_unified_encoder:
                # Unified approach - let autocast handle precision
                distortion_input = torch.cat([rays, angles, radial], dim=1)
                encoded_features = self.unified_encoder(distortion_input)
            else:
                # Separate encoders approach
                rays_emb = self.rays_encoder(rays)
                angles_emb = self.angles_encoder(angles)
                radial_emb = self.radial_encoder(radial)
                
                # Check individual encoder outputs
                if not (torch.isfinite(rays_emb).all() and torch.isfinite(angles_emb).all() and torch.isfinite(radial_emb).all()):
                    if debug:
                        print(f"[FiLM FP16] NaN in encoder outputs at forward #{self.forward_count}")
                    self.nan_count += 1
                    return features
                
                # Combine and fuse
                combined = torch.cat([rays_emb, angles_emb, radial_emb], dim=1)
                encoded_features = self.fusion_net(combined)
            
            # Check encoder output
            if not torch.isfinite(encoded_features).all():
                if debug:
                    print(f"[FiLM FP16] NaN in encoded features at forward #{self.forward_count}")
                self.nan_count += 1
                return features
            
            # Generate modulation parameters
            gamma, beta = self.generate_modulation_params_fp16(encoded_features)
            
            # Apply modulation (autocast will handle precision)
            modulated = gamma * features + beta
            
            # Final safety check
            if not torch.isfinite(modulated).all():
                if debug:
                    print(f"[FiLM FP16] NaN in final output at forward #{self.forward_count}")
                self.nan_count += 1
                return features
            
            # Debug statistics
            if debug and self.forward_count % 100 == 0:
                print(f"[FiLM FP16] Stats after {self.forward_count} forwards:")
                print(f"  NaN occurrences: {self.nan_count}")
                print(f"  Current dtype: {modulated.dtype}")
                print(f"  Gamma range: [{gamma.min():.4f}, {gamma.max():.4f}]")
                print(f"  Beta range: [{beta.min():.4f}, {beta.max():.4f}]")
                print(f"  Output range: [{modulated.min():.4f}, {modulated.max():.4f}]")
                
                # Adaptive bounds statistics
                self._print_adaptive_bounds_stats()
            
            # Ensure output matches input dtype
            return modulated.to(original_dtype)
            
        except Exception as e:
            if debug:
                print(f"[FiLM FP16] Exception at forward #{self.forward_count}: {e}")
            self.nan_count += 1
            return features
    
    def get_stats(self):
        """Return debugging statistics including adaptive bounds"""
        # Get current effective ranges
        gamma_scale_safe = torch.clamp(self.gamma_scale, 0.1, 1.0)
        gamma_offset_safe = torch.clamp(self.gamma_offset, 0.5, 1.0)
        beta_scale_safe = torch.clamp(self.beta_scale, 0.01, 0.2)
        gamma_clamp = torch.clamp(self.gamma_clamp_bound, 3.0, 15.0)
        beta_clamp = torch.clamp(self.beta_clamp_bound, 2.0, 10.0)
        
        return {
            'forward_count': self.forward_count,
            'nan_count': self.nan_count,
            'nan_rate': self.nan_count / max(self.forward_count, 1),
            'use_unified_encoder': self.use_unified_encoder,
            'fp16_safe_ops': self.fp16_safe_ops,
            
            # Adaptive bounds info
            'adaptive_bounds': {
                'gamma_range': [gamma_offset_safe.item(), 
                              (gamma_offset_safe + gamma_scale_safe).item()],
                'beta_range': [-beta_scale_safe.item(), beta_scale_safe.item()],
                'gamma_clamp_bound': gamma_clamp.item(),
                'beta_clamp_bound': beta_clamp.item(),
                
                # Raw parameter values
                'raw_gamma_scale': self.gamma_scale.item(),
                'raw_gamma_offset': self.gamma_offset.item(),
                'raw_beta_scale': self.beta_scale.item(),
                'raw_gamma_clamp_bound': self.gamma_clamp_bound.item(),
                'raw_beta_clamp_bound': self.beta_clamp_bound.item()
            }
        }
    
    def _print_adaptive_bounds_stats(self):
        """Print current adaptive bounds configuration"""
        stats = self.get_stats()
        bounds = stats['adaptive_bounds']
        
        print(f"  Adaptive Bounds:")
        print(f"    Gamma range: [{bounds['gamma_range'][0]:.4f}, {bounds['gamma_range'][1]:.4f}]")
        print(f"    Beta range: [{bounds['beta_range'][0]:.4f}, {bounds['beta_range'][1]:.4f}]")
        print(f"    Gamma clamp: ±{bounds['gamma_clamp_bound']:.2f}")
        print(f"    Beta clamp: ±{bounds['beta_clamp_bound']:.2f}")
        print(f"    Raw params: γ_scale={bounds['raw_gamma_scale']:.4f}, "
              f"γ_offset={bounds['raw_gamma_offset']:.4f}, β_scale={bounds['raw_beta_scale']:.4f}")
    

    
    def set_adaptive_bounds(self, gamma_scale=None, gamma_offset=None, beta_scale=None, 
                           gamma_clamp=None, beta_clamp=None):
        """Manually set adaptive bounds (useful for curriculum learning)"""
        if gamma_scale is not None:
            self.gamma_scale.data.fill_(gamma_scale)
        if gamma_offset is not None:
            self.gamma_offset.data.fill_(gamma_offset)
        if beta_scale is not None:
            self.beta_scale.data.fill_(beta_scale)
        if gamma_clamp is not None:
            self.gamma_clamp_bound.data.fill_(gamma_clamp)
        if beta_clamp is not None:
            self.beta_clamp_bound.data.fill_(beta_clamp)
    
    def reset_stats(self):
        """Reset debugging counters"""
        self.forward_count = 0
        self.nan_count = 0

# Factory function to choose between FP32 and FP16 versions
def create_film_module(feature_dim, use_fp16=True, use_unified_encoder=True, 
                      adaptive_bounds_config=None, **kwargs):
    """
    Factory function to create appropriate FiLM module
    
    Args:
        feature_dim: Feature dimension for modulation
        use_fp16: Whether to use FP16-optimized version
        use_unified_encoder: Whether to use unified vs separate encoders
        adaptive_bounds_config: Dict with adaptive bounds settings:
            {
                'gamma_scale': 0.5,      # Initial gamma range width
                'gamma_offset': 0.75,    # Initial gamma center point  
                'beta_scale': 0.05,      # Initial beta range width
                'gamma_clamp': 10.0,     # Initial gamma clamping bound
                'beta_clamp': 5.0        # Initial beta clamping bound
            }
    """
    if use_fp16:
        film_module = MultiModalDistortionFiLM_FP16(
            feature_dim=feature_dim,
            use_unified_encoder=use_unified_encoder,
            **kwargs
        )
        
        # Configure adaptive bounds if provided
        if adaptive_bounds_config is not None:
            film_module.set_adaptive_bounds(**adaptive_bounds_config)
            print(f"[FiLM] Configured adaptive bounds: {adaptive_bounds_config}")
            
        return film_module
    else:
        return MultiModalDistortionFiLM(
            feature_dim=feature_dim,
            use_unified_encoder=use_unified_encoder
        )