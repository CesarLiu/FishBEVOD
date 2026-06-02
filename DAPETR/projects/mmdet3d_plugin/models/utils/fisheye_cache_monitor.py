import torch
import time
from collections import defaultdict

class FisheyeCacheMonitor:
    """Monitor and analyze fisheye coordinate cache performance during training"""
    
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_count = 0
        self.timing_stats = defaultdict(list)
        self.cache_stats_history = []
        
    def log_step(self, fisheye_module, step_time=None):
        """Log cache performance for current step"""
        self.step_count += 1
        
        if step_time is not None:
            self.timing_stats['step_time'].append(step_time)
        
        # Get cache stats
        stats = fisheye_module.get_cache_stats()
        self.cache_stats_history.append(stats)
        
        # Log periodically
        if self.step_count % self.log_interval == 0:
            self._print_summary(fisheye_module)
    
    def _print_summary(self, fisheye_module):
        """Print cache performance summary"""
        stats = fisheye_module.get_cache_stats()
        recent_stats = self.cache_stats_history[-self.log_interval:]
        
        print(f"\n=== Fisheye Cache Stats (Step {self.step_count}) ===")
        print(f"Cache enabled: {stats['enabled']}")
        print(f"Current cache size: {stats['cache_size']}")
        print(f"Total hits: {stats['cache_hits']}")
        print(f"Total misses: {stats['cache_misses']}")
        print(f"Hit rate: {stats['hit_rate']:.2%}")
        
        if self.timing_stats['step_time']:
            avg_time = sum(self.timing_stats['step_time'][-self.log_interval:]) / min(len(self.timing_stats['step_time']), self.log_interval)
            print(f"Avg step time: {avg_time:.4f}s")
        
        # Analyze recent cache efficiency
        if len(recent_stats) > 1:
            recent_hits = recent_stats[-1]['cache_hits'] - recent_stats[0]['cache_hits']
            recent_misses = recent_stats[-1]['cache_misses'] - recent_stats[0]['cache_misses']
            recent_total = recent_hits + recent_misses
            recent_hit_rate = recent_hits / max(recent_total, 1)
            print(f"Recent hit rate (last {self.log_interval} steps): {recent_hit_rate:.2%}")
        
        print("=" * 45)
    
    def analyze_augmentation_impact(self, fisheye_module):
        """Analyze how augmentation affects cache performance"""
        stats = fisheye_module.get_cache_stats()
        
        analysis = {
            'cache_efficiency': 'high' if stats['hit_rate'] > 0.7 else 'medium' if stats['hit_rate'] > 0.3 else 'low',
            'recommendations': []
        }
        
        if stats['hit_rate'] < 0.3:
            analysis['recommendations'].extend([
                "Consider reducing resize_lim range to improve cache hits",
                "Verify that rotation and cropping are minimal",
                "Check if distortion parameters are stable across samples"
            ])
        elif stats['hit_rate'] < 0.7:
            analysis['recommendations'].append("Cache is moderately effective - current settings are reasonable")
        else:
            analysis['recommendations'].append("Excellent cache performance - augmentation is cache-friendly")
        
        return analysis

# Usage example for integration into training loop
def create_cache_monitor():
    """Factory function to create cache monitor"""
    return FisheyeCacheMonitor(log_interval=50)  # Log every 50 steps

# Hook for MMDetection training
class CacheMonitorHook:
    """Hook to integrate cache monitoring into MMDetection training"""
    
    def __init__(self, log_interval=100):
        self.monitor = FisheyeCacheMonitor(log_interval)
        self.start_time = None
    
    def before_iter(self, runner):
        self.start_time = time.time()
    
    def after_iter(self, runner):
        if self.start_time is not None:
            step_time = time.time() - self.start_time
            
            # Find fisheye module in the model
            fisheye_module = self._find_fisheye_module(runner.model)
            if fisheye_module is not None:
                self.monitor.log_step(fisheye_module, step_time)
    
    def _find_fisheye_module(self, model):
        """Recursively find FisheyeCoordAug module in the model"""
        from .fisheye_coords import FisheyeCoordAug
        
        for module in model.modules():
            if isinstance(module, FisheyeCoordAug):
                return module
        return None
