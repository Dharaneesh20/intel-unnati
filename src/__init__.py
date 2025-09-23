"""
Core framework initialization and main entry point.
"""

__version__ = "1.0.0"
__author__ = "Intel AI Agent Framework Team"

from .core.workflow import Workflow, Task
from .core.agents import Agent
from .core.state_machine import StateMachine, State, Transition

__all__ = [
    "Workflow",
    "Task", 
    "Agent",
    "StateMachine",
    "State",
    "Transition"
]
