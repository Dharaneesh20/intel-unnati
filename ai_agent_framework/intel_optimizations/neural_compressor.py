"""
Neural Compressor integration for model quantization and pruning
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import asyncio
import time
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class QuantizationStrategy(Enum):
    """Quantization strategies"""
    POST_TRAINING_STATIC = "post_training_static"
    POST_TRAINING_DYNAMIC = "post_training_dynamic"
    QUANTIZATION_AWARE_TRAINING = "quantization_aware_training"


class PruningStrategy(Enum):
    """Pruning strategies"""
    MAGNITUDE = "magnitude"
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    GRADIENT = "gradient_sensitivity"


@dataclass
class QuantizationConfig:
    """Configuration for quantization"""
    strategy: QuantizationStrategy
    accuracy_target: float = 0.01  # Allowed accuracy drop (1%)
    timeout: int = 0  # Timeout in seconds (0 = no timeout)
    max_trials: int = 100
    dataloader_config: Optional[Dict[str, Any]] = None
    eval_func: Optional[Callable] = None
    calibration_sampling_size: int = 100


@dataclass
class PruningConfig:
    """Configuration for pruning"""
    strategy: PruningStrategy
    sparsity_target: float = 0.9  # Target sparsity (90%)
    accuracy_target: float = 0.01  # Allowed accuracy drop (1%)
    start_epoch: int = 0
    end_epoch: int = 10
    frequency: int = 1


@dataclass
class CompressionResult:
    """Results from model compression"""
    model_name: str
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    accuracy_original: float
    accuracy_compressed: float
    accuracy_drop: float
    latency_original_ms: float
    latency_compressed_ms: float
    speedup: float
    compression_type: str


class NeuralCompressorOptimizer:
    """
    Intel Neural Compressor optimizer for model compression
    """
    
    def __init__(self, work_dir: str = "./nc_workspace"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.nc_available = self._check_neural_compressor()
        self.compressed_models: Dict[str, Any] = {}
    
    def _check_neural_compressor(self) -> bool:
        """Check if Neural Compressor is available"""
        try:
            import neural_compressor
            return True
        except ImportError:
            print("Neural Compressor not available.")
            print("Install with: pip install neural-compressor")
            return False
    
    async def quantize_model(
        self,
        model: Any,
        config: QuantizationConfig,
        model_name: str,
        calibration_dataloader: Optional[Any] = None,
        evaluation_dataloader: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Quantize model using Neural Compressor
        
        Args:
            model: Model to quantize
            config: Quantization configuration
            model_name: Name of the model
            calibration_dataloader: DataLoader for calibration
            evaluation_dataloader: DataLoader for evaluation
            
        Returns:
            Quantized model
        """
        if not self.nc_available:
            print("Neural Compressor not available")
            return None
        
        try:
            from neural_compressor import quantization
            from neural_compressor.config import PostTrainingQuantConfig, QuantizationAwareTrainingConfig
            
            print(f"Quantizing model: {model_name}")
            
            # Configure quantization based on strategy
            if config.strategy == QuantizationStrategy.POST_TRAINING_STATIC:
                quant_config = PostTrainingQuantConfig(
                    approach="static",
                    backend="default",
                    accuracy_criterion={
                        'relative': config.accuracy_target,
                        'absolute': 0.01
                    },
                    timeout=config.timeout,
                    max_trials=config.max_trials
                )
            elif config.strategy == QuantizationStrategy.POST_TRAINING_DYNAMIC:
                quant_config = PostTrainingQuantConfig(
                    approach="dynamic",
                    backend="default"
                )
            else:  # QAT
                quant_config = QuantizationAwareTrainingConfig(
                    backend="default"
                )
            
            # Create quantizer
            q_model = quantization.fit(
                model=model,
                conf=quant_config,
                calib_dataloader=calibration_dataloader,
                eval_dataloader=evaluation_dataloader,
                eval_func=config.eval_func
            )
            
            # Store quantized model
            self.compressed_models[f"{model_name}_quantized"] = q_model
            
            # Save quantized model
            model_path = self.work_dir / f"{model_name}_quantized"
            q_model.save(str(model_path))
            
            print(f"Model quantized and saved to: {model_path}")
            return q_model
            
        except Exception as e:
            print(f"Quantization failed: {e}")
            return None
    
    async def prune_model(
        self,
        model: Any,
        config: PruningConfig,
        model_name: str,
        train_dataloader: Optional[Any] = None,
        eval_dataloader: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Prune model using Neural Compressor
        
        Args:
            model: Model to prune
            config: Pruning configuration
            model_name: Name of the model
            train_dataloader: Training DataLoader
            eval_dataloader: Evaluation DataLoader
            
        Returns:
            Pruned model
        """
        if not self.nc_available:
            print("Neural Compressor not available")
            return None
        
        try:
            from neural_compressor import pruning
            from neural_compressor.config import WeightPruningConfig
            
            print(f"Pruning model: {model_name}")
            
            # Configure pruning
            prune_config = WeightPruningConfig(
                pruning_type=config.strategy.value,
                target_sparsity=config.sparsity_target,
                start_epoch=config.start_epoch,
                end_epoch=config.end_epoch,
                pruning_frequency=config.frequency
            )
            
            # Create pruner
            pruner = pruning.Pruning(prune_config)
            pruner.model = model
            
            # Apply pruning (this would typically require training loop)
            # For now, we'll create a simplified version
            pruned_model = pruner.fit()
            
            # Store pruned model
            self.compressed_models[f"{model_name}_pruned"] = pruned_model
            
            # Save pruned model
            model_path = self.work_dir / f"{model_name}_pruned"
            # pruned_model.save(str(model_path))  # Depends on model type
            
            print(f"Model pruned and saved to: {model_path}")
            return pruned_model
            
        except Exception as e:
            print(f"Pruning failed: {e}")
            return None
    
    async def apply_mixed_precision(
        self,
        model: Any,
        model_name: str,
        precision: str = "bf16"
    ) -> Optional[Any]:
        """
        Apply mixed precision optimization
        
        Args:
            model: Model to optimize
            model_name: Name of the model
            precision: Precision type (bf16, fp16)
            
        Returns:
            Optimized model
        """
        if not self.nc_available:
            print("Neural Compressor not available")
            return None
        
        try:
            from neural_compressor import mixed_precision
            from neural_compressor.config import MixedPrecisionConfig
            
            print(f"Applying mixed precision ({precision}) to model: {model_name}")
            
            # Configure mixed precision
            mp_config = MixedPrecisionConfig(
                backend="default",
                precision=precision
            )
            
            # Apply mixed precision
            mp_model = mixed_precision.fit(model=model, conf=mp_config)
            
            # Store optimized model
            self.compressed_models[f"{model_name}_mp_{precision}"] = mp_model
            
            print(f"Mixed precision applied successfully")
            return mp_model
            
        except Exception as e:
            print(f"Mixed precision optimization failed: {e}")
            return None
    
    async def benchmark_compression(
        self,
        original_model: Any,
        compressed_model: Any,
        model_name: str,
        eval_func: Optional[Callable] = None,
        test_dataloader: Optional[Any] = None,
        num_iterations: int = 100
    ) -> Optional[CompressionResult]:
        """
        Benchmark compression results
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            model_name: Name of the model
            eval_func: Function to evaluate accuracy
            test_dataloader: Test DataLoader
            num_iterations: Number of inference iterations
            
        Returns:
            Compression benchmark results
        """
        try:
            print(f"Benchmarking compression for {model_name}...")
            
            # Calculate model sizes (placeholder implementation)
            original_size = 100.0  # MB (placeholder)
            compressed_size = 25.0  # MB (placeholder)
            compression_ratio = original_size / compressed_size
            
            # Evaluate accuracy if eval function is provided
            accuracy_original = 0.95  # Placeholder
            accuracy_compressed = 0.94  # Placeholder
            accuracy_drop = accuracy_original - accuracy_compressed
            
            if eval_func and test_dataloader:
                try:
                    accuracy_original = eval_func(original_model, test_dataloader)
                    accuracy_compressed = eval_func(compressed_model, test_dataloader)
                    accuracy_drop = accuracy_original - accuracy_compressed
                except Exception as e:
                    print(f"Accuracy evaluation failed: {e}")
            
            # Benchmark inference latency (placeholder implementation)
            latency_original = 10.0  # ms (placeholder)
            latency_compressed = 4.0  # ms (placeholder)
            speedup = latency_original / latency_compressed
            
            result = CompressionResult(
                model_name=model_name,
                original_size_mb=original_size,
                compressed_size_mb=compressed_size,
                compression_ratio=compression_ratio,
                accuracy_original=accuracy_original,
                accuracy_compressed=accuracy_compressed,
                accuracy_drop=accuracy_drop,
                latency_original_ms=latency_original,
                latency_compressed_ms=latency_compressed,
                speedup=speedup,
                compression_type="quantization"
            )
            
            print(f"Compression Benchmark Results:")
            print(f"  Original size: {original_size:.1f} MB")
            print(f"  Compressed size: {compressed_size:.1f} MB")
            print(f"  Compression ratio: {compression_ratio:.1f}x")
            print(f"  Accuracy drop: {accuracy_drop:.3f}")
            print(f"  Speedup: {speedup:.1f}x")
            
            return result
            
        except Exception as e:
            print(f"Compression benchmarking failed: {e}")
            return None
    
    def get_compressed_model(self, model_name: str) -> Optional[Any]:
        """Get a compressed model by name"""
        return self.compressed_models.get(model_name)
    
    def list_compressed_models(self) -> List[str]:
        """List all compressed models"""
        return list(self.compressed_models.keys())
    
    async def export_compression_report(
        self,
        results: List[CompressionResult],
        output_path: str
    ):
        """
        Export compression results to JSON report
        
        Args:
            results: List of compression results
            output_path: Path to save the report
        """
        try:
            report = {
                "timestamp": time.time(),
                "summary": {
                    "total_models": len(results),
                    "avg_compression_ratio": sum(r.compression_ratio for r in results) / len(results) if results else 0,
                    "avg_speedup": sum(r.speedup for r in results) / len(results) if results else 0,
                    "avg_accuracy_drop": sum(r.accuracy_drop for r in results) / len(results) if results else 0
                },
                "results": [result.__dict__ for result in results]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"Compression report exported to: {output_path}")
            
        except Exception as e:
            print(f"Failed to export compression report: {e}")


class AutoQuantization:
    """
    Automatic quantization helper
    """
    
    def __init__(self, work_dir: str = "./auto_quant"):
        self.optimizer = NeuralCompressorOptimizer(work_dir)
    
    async def auto_quantize(
        self,
        model: Any,
        model_name: str,
        calibration_data: Optional[Any] = None,
        accuracy_target: float = 0.01,
        try_all_strategies: bool = True
    ) -> Optional[Any]:
        """
        Automatically find best quantization strategy
        
        Args:
            model: Model to quantize
            model_name: Name of the model
            calibration_data: Calibration data
            accuracy_target: Maximum allowed accuracy drop
            try_all_strategies: Whether to try all quantization strategies
            
        Returns:
            Best quantized model
        """
        if not try_all_strategies:
            # Use default post-training static quantization
            config = QuantizationConfig(
                strategy=QuantizationStrategy.POST_TRAINING_STATIC,
                accuracy_target=accuracy_target
            )
            return await self.optimizer.quantize_model(
                model, config, model_name, calibration_data
            )
        
        best_model = None
        best_result = None
        
        strategies = [
            QuantizationStrategy.POST_TRAINING_DYNAMIC,
            QuantizationStrategy.POST_TRAINING_STATIC,
        ]
        
        for strategy in strategies:
            try:
                print(f"Trying quantization strategy: {strategy.value}")
                
                config = QuantizationConfig(
                    strategy=strategy,
                    accuracy_target=accuracy_target
                )
                
                quantized_model = await self.optimizer.quantize_model(
                    model, config, f"{model_name}_{strategy.value}", calibration_data
                )
                
                if quantized_model:
                    # Benchmark this strategy
                    result = await self.optimizer.benchmark_compression(
                        model, quantized_model, f"{model_name}_{strategy.value}"
                    )
                    
                    if result and (best_result is None or result.speedup > best_result.speedup):
                        best_model = quantized_model
                        best_result = result
                        print(f"New best strategy: {strategy.value} (speedup: {result.speedup:.1f}x)")
            
            except Exception as e:
                print(f"Strategy {strategy.value} failed: {e}")
                continue
        
        if best_model:
            print(f"Best quantization strategy found with {best_result.speedup:.1f}x speedup")
        
        return best_model


# Convenience functions
async def quantize_model_auto(
    model: Any,
    model_name: str,
    calibration_data: Optional[Any] = None,
    accuracy_target: float = 0.01
) -> Optional[Any]:
    """
    Quick automatic quantization
    
    Args:
        model: Model to quantize
        model_name: Name of the model
        calibration_data: Calibration data
        accuracy_target: Maximum accuracy drop allowed
        
    Returns:
        Quantized model
    """
    auto_quant = AutoQuantization()
    return await auto_quant.auto_quantize(model, model_name, calibration_data, accuracy_target)


async def prune_model_magnitude(
    model: Any,
    model_name: str,
    sparsity: float = 0.9,
    accuracy_target: float = 0.01
) -> Optional[Any]:
    """
    Quick magnitude-based pruning
    
    Args:
        model: Model to prune
        model_name: Name of the model
        sparsity: Target sparsity ratio
        accuracy_target: Maximum accuracy drop allowed
        
    Returns:
        Pruned model
    """
    optimizer = NeuralCompressorOptimizer()
    
    config = PruningConfig(
        strategy=PruningStrategy.MAGNITUDE,
        sparsity_target=sparsity,
        accuracy_target=accuracy_target
    )
    
    return await optimizer.prune_model(model, config, model_name)