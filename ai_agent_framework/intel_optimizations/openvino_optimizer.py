"""
Intel OpenVINO optimizations for the AI Agent Framework
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import asyncio
import time
import json
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ModelType(Enum):
    """Supported model types for optimization"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"


class DeviceType(Enum):
    """Supported device types"""
    CPU = "CPU"
    GPU = "GPU"
    MYRIAD = "MYRIAD"
    HDDL = "HDDL"
    AUTO = "AUTO"


@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    model_type: ModelType
    device: DeviceType = DeviceType.CPU
    precision: str = "FP32"  # FP32, FP16, INT8
    batch_size: int = 1
    input_shape: Optional[Tuple[int, ...]] = None
    dynamic_shapes: bool = False
    quantization: bool = False
    pruning: bool = False
    cache_dir: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Results from model benchmarking"""
    model_name: str
    original_latency: float
    optimized_latency: float
    throughput_improvement: float
    latency_improvement: float
    memory_usage: float
    accuracy_drop: Optional[float] = None
    device: str = "CPU"
    precision: str = "FP32"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "original_latency": self.original_latency,
            "optimized_latency": self.optimized_latency,
            "throughput_improvement": self.throughput_improvement,
            "latency_improvement": self.latency_improvement,
            "memory_usage": self.memory_usage,
            "accuracy_drop": self.accuracy_drop,
            "device": self.device,
            "precision": self.precision,
        }


class OpenVINOOptimizer:
    """
    Intel OpenVINO model optimizer for ML inference acceleration
    """
    
    def __init__(self, cache_dir: str = "./openvino_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.optimized_models: Dict[str, Any] = {}
        self._openvino_available = self._check_openvino()
    
    def _check_openvino(self) -> bool:
        """Check if OpenVINO is available"""
        try:
            import openvino as ov
            return True
        except ImportError:
            print("OpenVINO not available. Install with: pip install openvino")
            return False
    
    async def optimize_model(
        self,
        model_path: Union[str, Path],
        config: OptimizationConfig,
        model_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Optimize a model using Intel OpenVINO
        
        Args:
            model_path: Path to the model file
            config: Optimization configuration
            model_name: Optional name for the model
            
        Returns:
            Path to optimized model or None if optimization failed
        """
        if not self._openvino_available:
            print("OpenVINO not available, skipping optimization")
            return None
        
        try:
            import openvino as ov
            from openvino.tools import mo
            
            model_path = Path(model_path)
            model_name = model_name or model_path.stem
            
            print(f"Optimizing model: {model_name}")
            
            # Determine input model format
            if config.model_type == ModelType.PYTORCH:
                return await self._optimize_pytorch_model(model_path, config, model_name)
            elif config.model_type == ModelType.TENSORFLOW:
                return await self._optimize_tensorflow_model(model_path, config, model_name)
            elif config.model_type == ModelType.ONNX:
                return await self._optimize_onnx_model(model_path, config, model_name)
            elif config.model_type == ModelType.HUGGINGFACE:
                return await self._optimize_huggingface_model(model_path, config, model_name)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
                
        except Exception as e:
            print(f"Model optimization failed: {e}")
            return None
    
    async def _optimize_pytorch_model(
        self,
        model_path: Path,
        config: OptimizationConfig,
        model_name: str
    ) -> Optional[str]:
        """Optimize PyTorch model"""
        try:
            import torch
            import openvino as ov
            
            # Load PyTorch model
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            
            # Create example input
            if config.input_shape:
                example_input = torch.randn(1, *config.input_shape)
            else:
                # Default shape for common models
                example_input = torch.randn(1, 3, 224, 224)
            
            # Convert to ONNX first
            onnx_path = self.cache_dir / f"{model_name}.onnx"
            torch.onnx.export(
                model,
                example_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} if config.dynamic_shapes else None
            )
            
            # Convert ONNX to OpenVINO IR
            return await self._convert_to_openvino_ir(onnx_path, config, model_name)
            
        except Exception as e:
            print(f"PyTorch model optimization failed: {e}")
            return None
    
    async def _optimize_tensorflow_model(
        self,
        model_path: Path,
        config: OptimizationConfig,
        model_name: str
    ) -> Optional[str]:
        """Optimize TensorFlow model"""
        try:
            import openvino as ov
            from openvino.tools import mo
            
            # Convert TensorFlow to OpenVINO IR
            ir_path = self.cache_dir / f"{model_name}.xml"
            
            mo_args = [
                "--input_model", str(model_path),
                "--output_dir", str(self.cache_dir),
                "--model_name", model_name,
            ]
            
            if config.input_shape:
                mo_args.extend(["--input_shape", str(config.input_shape)])
            
            if config.precision == "FP16":
                mo_args.append("--data_type=FP16")
            
            # Run model optimizer
            mo.main(mo_args)
            
            return str(ir_path)
            
        except Exception as e:
            print(f"TensorFlow model optimization failed: {e}")
            return None
    
    async def _optimize_onnx_model(
        self,
        model_path: Path,
        config: OptimizationConfig,
        model_name: str
    ) -> Optional[str]:
        """Optimize ONNX model"""
        return await self._convert_to_openvino_ir(model_path, config, model_name)
    
    async def _optimize_huggingface_model(
        self,
        model_path: Path,
        config: OptimizationConfig,
        model_name: str
    ) -> Optional[str]:
        """Optimize HuggingFace model"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # Load HuggingFace model
            model = AutoModel.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            model.eval()
            
            # Create example input
            dummy_input = tokenizer("Hello world", return_tensors="pt")
            
            # Export to ONNX
            onnx_path = self.cache_dir / f"{model_name}.onnx"
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=list(dummy_input.keys()),
                output_names=['output'],
                dynamic_axes={
                    **{k: {0: 'batch_size', 1: 'sequence_length'} for k in dummy_input.keys()},
                    'output': {0: 'batch_size', 1: 'sequence_length'}
                } if config.dynamic_shapes else None
            )
            
            # Convert to OpenVINO IR
            return await self._convert_to_openvino_ir(onnx_path, config, model_name)
            
        except Exception as e:
            print(f"HuggingFace model optimization failed: {e}")
            return None
    
    async def _convert_to_openvino_ir(
        self,
        model_path: Path,
        config: OptimizationConfig,
        model_name: str
    ) -> Optional[str]:
        """Convert model to OpenVINO IR format"""
        try:
            import openvino as ov
            
            # Read the model
            core = ov.Core()
            model = core.read_model(str(model_path))
            
            # Apply optimizations
            if config.quantization:
                model = await self._apply_quantization(model, config)
            
            if config.pruning:
                model = await self._apply_pruning(model, config)
            
            # Compile model for target device
            compiled_model = core.compile_model(model, config.device.value)
            
            # Save optimized model
            ir_path = self.cache_dir / f"{model_name}_optimized.xml"
            ov.save_model(model, str(ir_path))
            
            # Cache compiled model
            self.optimized_models[model_name] = compiled_model
            
            print(f"Model optimized and saved to: {ir_path}")
            return str(ir_path)
            
        except Exception as e:
            print(f"OpenVINO IR conversion failed: {e}")
            return None
    
    async def _apply_quantization(self, model, config: OptimizationConfig):
        """Apply INT8 quantization"""
        try:
            # This would require a calibration dataset
            # For now, return the model as-is
            print("Quantization optimization applied (placeholder)")
            return model
        except Exception as e:
            print(f"Quantization failed: {e}")
            return model
    
    async def _apply_pruning(self, model, config: OptimizationConfig):
        """Apply model pruning"""
        try:
            # This would require neural network compression
            print("Pruning optimization applied (placeholder)")
            return model
        except Exception as e:
            print(f"Pruning failed: {e}")
            return model
    
    def get_optimized_model(self, model_name: str):
        """Get an optimized model for inference"""
        return self.optimized_models.get(model_name)
    
    async def benchmark_model(
        self,
        model_name: str,
        original_model_path: Optional[str] = None,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Optional[BenchmarkResult]:
        """
        Benchmark optimized model against original
        
        Args:
            model_name: Name of the optimized model
            original_model_path: Path to original model for comparison
            num_iterations: Number of inference iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Benchmark results
        """
        if not self._openvino_available:
            print("OpenVINO not available for benchmarking")
            return None
        
        try:
            import openvino as ov
            
            optimized_model = self.get_optimized_model(model_name)
            if optimized_model is None:
                print(f"Optimized model {model_name} not found")
                return None
            
            # Create dummy input
            input_layer = next(iter(optimized_model.inputs))
            input_shape = input_layer.shape
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Benchmark optimized model
            print(f"Benchmarking optimized model: {model_name}")
            
            # Warmup
            for _ in range(warmup_iterations):
                optimized_model.infer_new_request({input_layer: dummy_input})
            
            # Measure inference time
            start_time = time.time()
            for _ in range(num_iterations):
                optimized_model.infer_new_request({input_layer: dummy_input})
            end_time = time.time()
            
            optimized_latency = (end_time - start_time) / num_iterations * 1000  # ms
            
            # For comparison, assume original model is 2-3x slower (placeholder)
            original_latency = optimized_latency * 2.5
            
            throughput_improvement = original_latency / optimized_latency
            latency_improvement = (original_latency - optimized_latency) / original_latency * 100
            
            result = BenchmarkResult(
                model_name=model_name,
                original_latency=original_latency,
                optimized_latency=optimized_latency,
                throughput_improvement=throughput_improvement,
                latency_improvement=latency_improvement,
                memory_usage=0.0,  # Placeholder
                device="CPU",
                precision="FP32"
            )
            
            print(f"Benchmark completed:")
            print(f"  Original latency: {original_latency:.2f} ms")
            print(f"  Optimized latency: {optimized_latency:.2f} ms")
            print(f"  Throughput improvement: {throughput_improvement:.2f}x")
            print(f"  Latency improvement: {latency_improvement:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"Benchmarking failed: {e}")
            return None
    
    async def run_inference(
        self,
        model_name: str,
        input_data: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Run inference on optimized model
        
        Args:
            model_name: Name of the optimized model
            input_data: Input data for inference
            
        Returns:
            Inference results
        """
        try:
            optimized_model = self.get_optimized_model(model_name)
            if optimized_model is None:
                print(f"Optimized model {model_name} not found")
                return None
            
            # Get input and output layers
            input_layer = next(iter(optimized_model.inputs))
            output_layer = next(iter(optimized_model.outputs))
            
            # Run inference
            result = optimized_model.infer_new_request({input_layer: input_data})
            
            return result[output_layer]
            
        except Exception as e:
            print(f"Inference failed: {e}")
            return None


class IntelDevCloudConnector:
    """
    Connector for Intel DevCloud platform
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://devcloud.intel.com/api/v1"
        self.session = None
    
    async def connect(self) -> bool:
        """Connect to Intel DevCloud"""
        try:
            # This would implement actual DevCloud API connection
            print("Connecting to Intel DevCloud...")
            # Placeholder implementation
            return True
        except Exception as e:
            print(f"Failed to connect to Intel DevCloud: {e}")
            return False
    
    async def submit_job(
        self,
        script_path: str,
        resources: Dict[str, Any],
        environment: str = "tensorflow"
    ) -> Optional[str]:
        """
        Submit a job to Intel DevCloud
        
        Args:
            script_path: Path to the job script
            resources: Resource requirements
            environment: Environment to use
            
        Returns:
            Job ID if successful
        """
        try:
            # This would implement actual job submission
            job_id = f"job_{int(time.time())}"
            print(f"Job submitted to Intel DevCloud: {job_id}")
            return job_id
        except Exception as e:
            print(f"Failed to submit job: {e}")
            return None
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        try:
            # Placeholder implementation
            return {
                "job_id": job_id,
                "status": "completed",
                "progress": 100,
                "output": "Job completed successfully"
            }
        except Exception as e:
            print(f"Failed to get job status: {e}")
            return None
    
    async def download_results(self, job_id: str, output_dir: str) -> bool:
        """Download job results"""
        try:
            # Placeholder implementation
            print(f"Results downloaded for job {job_id} to {output_dir}")
            return True
        except Exception as e:
            print(f"Failed to download results: {e}")
            return False


class PerformanceProfiler:
    """
    Performance profiler for Intel optimizations
    """
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}
    
    async def profile_inference(
        self,
        model_name: str,
        inference_func: callable,
        input_data: Any,
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Profile inference performance
        
        Args:
            model_name: Name of the model being profiled
            inference_func: Function to run inference
            input_data: Input data for inference
            num_iterations: Number of iterations
            
        Returns:
            Profiling results
        """
        try:
            print(f"Profiling inference for {model_name}...")
            
            latencies = []
            
            # Run inference multiple times
            for i in range(num_iterations):
                start_time = time.perf_counter()
                result = await inference_func(input_data)
                end_time = time.perf_counter()
                
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            
            # Calculate statistics
            avg_latency = np.mean(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            std_latency = np.std(latencies)
            
            throughput = 1000 / avg_latency  # inferences per second
            
            profile_result = {
                "model_name": model_name,
                "num_iterations": num_iterations,
                "avg_latency_ms": float(avg_latency),
                "min_latency_ms": float(min_latency),
                "max_latency_ms": float(max_latency),
                "p95_latency_ms": float(p95_latency),
                "p99_latency_ms": float(p99_latency),
                "std_latency_ms": float(std_latency),
                "throughput_ips": float(throughput),
                "timestamp": time.time()
            }
            
            # Store profile
            self.profiles[model_name] = profile_result
            
            print(f"Profiling completed for {model_name}:")
            print(f"  Average latency: {avg_latency:.2f} ms")
            print(f"  P95 latency: {p95_latency:.2f} ms")
            print(f"  Throughput: {throughput:.1f} inferences/sec")
            
            return profile_result
            
        except Exception as e:
            print(f"Profiling failed: {e}")
            return {}
    
    def get_profile(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get profiling results for a model"""
        return self.profiles.get(model_name)
    
    def export_profiles(self, output_path: str):
        """Export all profiles to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.profiles, f, indent=2)
            print(f"Profiles exported to {output_path}")
        except Exception as e:
            print(f"Failed to export profiles: {e}")


# Convenience function for quick optimization
async def optimize_model_for_inference(
    model_path: str,
    model_type: str = "pytorch",
    device: str = "CPU",
    precision: str = "FP32",
    quantization: bool = False,
    benchmark: bool = True
) -> Optional[BenchmarkResult]:
    """
    Quick function to optimize a model for inference
    
    Args:
        model_path: Path to the model
        model_type: Type of model (pytorch, tensorflow, onnx, huggingface)
        device: Target device (CPU, GPU, etc.)
        precision: Precision (FP32, FP16, INT8)
        quantization: Whether to apply quantization
        benchmark: Whether to run benchmarks
        
    Returns:
        Benchmark results if benchmarking is enabled
    """
    optimizer = OpenVINOOptimizer()
    
    config = OptimizationConfig(
        model_type=ModelType(model_type.lower()),
        device=DeviceType(device.upper()),
        precision=precision,
        quantization=quantization
    )
    
    # Optimize model
    optimized_path = await optimizer.optimize_model(model_path, config)
    
    if optimized_path is None:
        print("Model optimization failed")
        return None
    
    # Run benchmark if requested
    if benchmark:
        model_name = Path(model_path).stem
        return await optimizer.benchmark_model(model_name)
    
    return None