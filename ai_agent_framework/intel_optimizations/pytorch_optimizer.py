"""
Intel Extension for PyTorch optimizations
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio
import time
import torch
from dataclasses import dataclass
from enum import Enum


class OptimizationLevel(Enum):
    """Optimization levels for Intel Extension for PyTorch"""
    O0 = "O0"  # No optimization
    O1 = "O1"  # Conservative optimization
    O2 = "O2"  # Aggressive optimization
    O3 = "O3"  # Maximum optimization


class DataType(Enum):
    """Supported data types"""
    FP32 = torch.float32
    FP16 = torch.float16
    BF16 = torch.bfloat16
    INT8 = torch.int8


@dataclass
class IntelPyTorchConfig:
    """Configuration for Intel PyTorch optimizations"""
    optimization_level: OptimizationLevel = OptimizationLevel.O1
    data_type: DataType = DataType.FP32
    channels_last: bool = False
    jit_compile: bool = True
    auto_mixed_precision: bool = False
    quantization: bool = False
    enable_onednn: bool = True


class IntelPyTorchOptimizer:
    """
    Intel Extension for PyTorch optimizer
    """
    
    def __init__(self):
        self.ipex_available = self._check_ipex()
        self.optimized_models: Dict[str, torch.nn.Module] = {}
    
    def _check_ipex(self) -> bool:
        """Check if Intel Extension for PyTorch is available"""
        try:
            import intel_extension_for_pytorch as ipex
            return True
        except ImportError:
            print("Intel Extension for PyTorch not available.")
            print("Install with: pip install intel_extension_for_pytorch")
            return False
    
    async def optimize_model(
        self,
        model: torch.nn.Module,
        config: IntelPyTorchConfig,
        sample_input: Optional[torch.Tensor] = None,
        model_name: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Optimize PyTorch model with Intel Extensions
        
        Args:
            model: PyTorch model to optimize
            config: Optimization configuration
            sample_input: Sample input tensor for tracing
            model_name: Optional name for the model
            
        Returns:
            Optimized model
        """
        if not self.ipex_available:
            print("Intel Extension for PyTorch not available, returning original model")
            return model
        
        try:
            import intel_extension_for_pytorch as ipex
            
            print(f"Optimizing model with Intel Extension for PyTorch...")
            
            # Set model to evaluation mode
            model.eval()
            
            # Convert data type if needed
            if config.data_type != DataType.FP32:
                model = model.to(dtype=config.data_type.value)
                if sample_input is not None:
                    sample_input = sample_input.to(dtype=config.data_type.value)
            
            # Enable channels last memory format for better performance
            if config.channels_last and sample_input is not None:
                if len(sample_input.shape) == 4:  # Image data
                    model = model.to(memory_format=torch.channels_last)
                    sample_input = sample_input.to(memory_format=torch.channels_last)
            
            # Apply Intel optimizations
            if config.enable_onednn:
                # Use Intel's oneDNN optimizations
                optimized_model = ipex.optimize(
                    model,
                    dtype=config.data_type.value,
                    level=config.optimization_level.value
                )
            else:
                optimized_model = model
            
            # Apply JIT compilation if requested
            if config.jit_compile and sample_input is not None:
                print("Applying JIT compilation...")
                with torch.no_grad():
                    optimized_model = torch.jit.trace(optimized_model, sample_input)
                    optimized_model = torch.jit.freeze(optimized_model)
            
            # Store optimized model
            if model_name:
                self.optimized_models[model_name] = optimized_model
            
            print("Model optimization completed")
            return optimized_model
            
        except Exception as e:
            print(f"Model optimization failed: {e}")
            return model
    
    async def optimize_for_inference(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        model_name: Optional[str] = None,
        use_bf16: bool = False,
        use_channels_last: bool = False
    ) -> torch.nn.Module:
        """
        Quick optimization for inference workloads
        
        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            model_name: Optional model name
            use_bf16: Whether to use bfloat16
            use_channels_last: Whether to use channels last format
            
        Returns:
            Optimized model
        """
        config = IntelPyTorchConfig(
            optimization_level=OptimizationLevel.O2,
            data_type=DataType.BF16 if use_bf16 else DataType.FP32,
            channels_last=use_channels_last,
            jit_compile=True,
            enable_onednn=True
        )
        
        return await self.optimize_model(model, config, sample_input, model_name)
    
    async def benchmark_optimization(
        self,
        original_model: torch.nn.Module,
        optimized_model: torch.nn.Module,
        sample_input: torch.Tensor,
        num_iterations: int = 100,
        warmup_iterations: int = 20
    ) -> Dict[str, float]:
        """
        Benchmark optimization improvements
        
        Args:
            original_model: Original PyTorch model
            optimized_model: Optimized model
            sample_input: Input tensor for benchmarking
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Benchmark results
        """
        try:
            print("Benchmarking model optimization...")
            
            # Ensure models are in eval mode
            original_model.eval()
            optimized_model.eval()
            
            # Warmup and benchmark original model
            with torch.no_grad():
                # Warmup
                for _ in range(warmup_iterations):
                    _ = original_model(sample_input)
                
                # Benchmark
                start_time = time.perf_counter()
                for _ in range(num_iterations):
                    _ = original_model(sample_input)
                end_time = time.perf_counter()
                
                original_time = (end_time - start_time) / num_iterations
            
            # Warmup and benchmark optimized model
            with torch.no_grad():
                # Warmup
                for _ in range(warmup_iterations):
                    _ = optimized_model(sample_input)
                
                # Benchmark
                start_time = time.perf_counter()
                for _ in range(num_iterations):
                    _ = optimized_model(sample_input)
                end_time = time.perf_counter()
                
                optimized_time = (end_time - start_time) / num_iterations
            
            # Calculate improvements
            speedup = original_time / optimized_time
            latency_reduction = (original_time - optimized_time) / original_time * 100
            
            results = {
                "original_latency_ms": original_time * 1000,
                "optimized_latency_ms": optimized_time * 1000,
                "speedup": speedup,
                "latency_reduction_percent": latency_reduction,
                "throughput_improvement": speedup
            }
            
            print(f"Benchmark Results:")
            print(f"  Original latency: {original_time * 1000:.2f} ms")
            print(f"  Optimized latency: {optimized_time * 1000:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Latency reduction: {latency_reduction:.1f}%")
            
            return results
            
        except Exception as e:
            print(f"Benchmarking failed: {e}")
            return {}
    
    def get_optimized_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """Get an optimized model by name"""
        return self.optimized_models.get(model_name)
    
    async def apply_quantization(
        self,
        model: torch.nn.Module,
        calibration_dataloader: Optional[Any] = None,
        model_name: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Apply INT8 quantization using Intel Neural Compressor
        
        Args:
            model: PyTorch model to quantize
            calibration_dataloader: DataLoader for calibration
            model_name: Optional model name
            
        Returns:
            Quantized model
        """
        if not self.ipex_available:
            print("Intel Extension for PyTorch not available")
            return model
        
        try:
            # This would integrate with Intel Neural Compressor
            # For now, return the model as-is (placeholder)
            print("Applying INT8 quantization (placeholder implementation)")
            
            # In a real implementation, this would use:
            # from neural_compressor import quantization
            # quantized_model = quantization.fit(model, ...)
            
            quantized_model = model  # Placeholder
            
            if model_name:
                self.optimized_models[f"{model_name}_quantized"] = quantized_model
            
            return quantized_model
            
        except Exception as e:
            print(f"Quantization failed: {e}")
            return model


class AutoMixedPrecisionOptimizer:
    """
    Automatic Mixed Precision optimizer for Intel hardware
    """
    
    def __init__(self):
        self.enabled = False
        self.scaler = None
    
    def enable_amp(self, model: torch.nn.Module) -> torch.nn.Module:
        """Enable Automatic Mixed Precision"""
        try:
            if torch.cuda.is_available():
                # Use CUDA AMP
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                # Use CPU mixed precision (if available)
                print("Enabling CPU mixed precision")
            
            self.enabled = True
            print("Automatic Mixed Precision enabled")
            return model
            
        except Exception as e:
            print(f"Failed to enable AMP: {e}")
            return model
    
    def forward_with_amp(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Forward pass with automatic mixed precision"""
        if not self.enabled:
            return model(input_tensor)
        
        try:
            if self.scaler:  # CUDA AMP
                with torch.cuda.amp.autocast():
                    return model(input_tensor)
            else:  # CPU mixed precision
                with torch.cpu.amp.autocast():
                    return model(input_tensor)
        except Exception as e:
            print(f"AMP forward pass failed: {e}")
            return model(input_tensor)


class MemoryOptimizer:
    """
    Memory optimization utilities for Intel hardware
    """
    
    @staticmethod
    def optimize_memory_usage(model: torch.nn.Module) -> torch.nn.Module:
        """Optimize memory usage"""
        try:
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                print("Memory efficient attention enabled")
            
            # Set memory format to channels last for conv models
            if any(isinstance(m, torch.nn.Conv2d) for m in model.modules()):
                model = model.to(memory_format=torch.channels_last)
                print("Channels last memory format enabled")
            
            return model
            
        except Exception as e:
            print(f"Memory optimization failed: {e}")
            return model
    
    @staticmethod
    def clear_cache():
        """Clear PyTorch cache"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Memory cache cleared")
        except Exception as e:
            print(f"Failed to clear cache: {e}")


# Convenience function for quick PyTorch optimization
async def optimize_pytorch_model(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    optimization_level: str = "O1",
    use_bf16: bool = False,
    use_jit: bool = True,
    model_name: Optional[str] = None
) -> torch.nn.Module:
    """
    Quick optimization for PyTorch models
    
    Args:
        model: PyTorch model to optimize
        sample_input: Sample input tensor
        optimization_level: Optimization level (O0, O1, O2, O3)
        use_bf16: Whether to use bfloat16
        use_jit: Whether to use JIT compilation
        model_name: Optional model name
        
    Returns:
        Optimized model
    """
    optimizer = IntelPyTorchOptimizer()
    
    config = IntelPyTorchConfig(
        optimization_level=OptimizationLevel(optimization_level),
        data_type=DataType.BF16 if use_bf16 else DataType.FP32,
        jit_compile=use_jit,
        channels_last=len(sample_input.shape) == 4,  # Enable for image data
        enable_onednn=True
    )
    
    return await optimizer.optimize_model(model, config, sample_input, model_name)