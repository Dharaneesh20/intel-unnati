"""
Intel optimizations for the AI Agent Framework

This module provides Intel-specific optimizations including:
- OpenVINO model optimization and inference
- Intel Extension for PyTorch optimizations
- Neural Compressor quantization and pruning
- Intel DevCloud integration for distributed computing
"""

from .openvino_optimizer import (
    OpenVINOOptimizer,
    OptimizationConfig,
    BenchmarkResult,
    ModelType,
    DeviceType,
    optimize_model_for_inference
)

from .pytorch_optimizer import (
    IntelPyTorchOptimizer,
    IntelPyTorchConfig,
    OptimizationLevel,
    DataType,
    optimize_pytorch_model
)

from .neural_compressor import (
    NeuralCompressorOptimizer,
    QuantizationConfig,
    PruningConfig,
    CompressionResult,
    quantize_model_auto,
    prune_model_magnitude
)

from .devcloud_integration import (
    DevCloudManager,
    DevCloudConfig,
    JobSubmission,
    JobResult,
    submit_optimization_job,
    run_distributed_benchmark
)

__all__ = [
    # OpenVINO
    'OpenVINOOptimizer',
    'OptimizationConfig',
    'BenchmarkResult',
    'ModelType',
    'DeviceType',
    'optimize_model_for_inference',
    
    # PyTorch
    'IntelPyTorchOptimizer',
    'IntelPyTorchConfig',
    'OptimizationLevel',
    'DataType',
    'optimize_pytorch_model',
    
    # Neural Compressor
    'NeuralCompressorOptimizer',
    'QuantizationConfig',
    'PruningConfig',
    'CompressionResult',
    'quantize_model_auto',
    'prune_model_magnitude',
    
    # DevCloud
    'DevCloudManager',
    'DevCloudConfig',
    'JobSubmission',
    'JobResult',
    'submit_optimization_job',
    'run_distributed_benchmark',
]