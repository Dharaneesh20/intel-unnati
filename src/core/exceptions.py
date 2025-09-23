"""
Core exceptions for the AI Agent Framework.
"""

class FrameworkError(Exception):
    """Base exception for all framework errors."""
    pass


class WorkflowError(FrameworkError):
    """Exception raised for workflow-related errors."""
    pass


class ExecutionError(FrameworkError):
    """Exception raised during task or workflow execution."""
    pass


class ValidationError(FrameworkError):
    """Exception raised for validation errors."""
    pass


class ConfigurationError(FrameworkError):
    """Exception raised for configuration errors."""
    pass


class MemoryError(FrameworkError):
    """Exception raised for memory management errors."""
    pass


class GuardrailViolation(FrameworkError):
    """Exception raised when guardrails are violated."""
    pass


class ToolError(FrameworkError):
    """Exception raised for tool-related errors."""
    pass


class TimeoutError(ExecutionError):
    """Exception raised when operations timeout."""
    pass


class RetryExhaustedError(ExecutionError):
    """Exception raised when all retry attempts are exhausted."""
    pass
