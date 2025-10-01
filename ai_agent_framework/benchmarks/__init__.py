"""
Benchmarks package initialization

This module provides the benchmark suite for the AI Agent Framework.
"""

from .benchmark_suite import (
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkConfig,
    BenchmarkType,
    MetricType,
    BenchmarkResult,
    BenchmarkMetrics,
    SystemMetrics,
    SystemMonitor,
    FrameworkBenchmarks,
    IntelOptimizationBenchmarks,
    AgentWorkflowBenchmarks,
    ScalabilityBenchmarks,
    run_quick_benchmark,
    benchmark_intel_optimizations
)

__all__ = [
    'BenchmarkRunner',
    'BenchmarkSuite',
    'BenchmarkConfig',
    'BenchmarkType',
    'MetricType',
    'BenchmarkResult',
    'BenchmarkMetrics',
    'SystemMetrics',
    'SystemMonitor',
    'FrameworkBenchmarks',
    'IntelOptimizationBenchmarks',
    'AgentWorkflowBenchmarks',
    'ScalabilityBenchmarks',
    'run_quick_benchmark',
    'benchmark_intel_optimizations'
]