"""
AI Agent Framework

A comprehensive framework for building and orchestrating intelligent agents
with support for DAG-based workflows, advanced memory management, Intel optimizations,
Apache component integration, benchmarking, and comprehensive monitoring capabilities.
"""

from . import core
from . import orchestration
from . import monitoring
from . import sdk
from . import intel_optimizations
from . import reference_agents
from . import apache_integration
from . import benchmarks

__version__ = "1.0.0"
__author__ = "AI Agent Framework Team"

__all__ = [
    'core',
    'orchestration', 
    'monitoring',
    'sdk',
    'intel_optimizations',
    'reference_agents',
    'apache_integration',
    'benchmarks'
]