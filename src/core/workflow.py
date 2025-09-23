"""
Workflow definition and execution engine.
"""

import uuid
import asyncio
from typing import Dict, List, Any, Optional, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from .exceptions import WorkflowError, ValidationError, ExecutionError


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_id: str
    status: WorkflowStatus
    outputs: Dict[str, Any] = field(default_factory=dict)
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class Task:
    """Represents a single task in a workflow."""
    
    def __init__(
        self,
        name: str,
        tool: Any,
        inputs: Dict[str, Any] = None,
        outputs: List[str] = None,
        dependencies: List[str] = None,
        timeout: Optional[int] = None,
        retries: int = 3,
        condition: Optional[Callable] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.tool = tool
        self.inputs = inputs or {}
        self.outputs = outputs or []
        self.dependencies = dependencies or []
        self.timeout = timeout
        self.retries = retries
        self.condition = condition
        self.status = TaskStatus.PENDING
        
    def __rshift__(self, other: "Task") -> "Task":
        """Support for >> operator to define dependencies."""
        other.dependencies.append(self.name)
        return other
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task can be executed based on dependencies."""
        if self.condition and not self.condition():
            return False
        return all(dep in completed_tasks for dep in self.dependencies)
    
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute the task."""
        start_time = datetime.now(timezone.utc)
        self.status = TaskStatus.RUNNING
        
        try:
            # Resolve input variables from context
            resolved_inputs = self._resolve_inputs(context)
            
            # Execute the tool
            if hasattr(self.tool, 'execute'):
                result = await self.tool.execute(resolved_inputs)
            elif callable(self.tool):
                result = await self.tool(resolved_inputs)
            else:
                raise ExecutionError(f"Tool {self.tool} is not executable")
            
            self.status = TaskStatus.COMPLETED
            end_time = datetime.now(timezone.utc)
            
            return TaskResult(
                task_id=self.id,
                status=self.status,
                outputs=result,
                execution_time=(end_time - start_time).total_seconds(),
                started_at=start_time,
                completed_at=end_time
            )
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            end_time = datetime.now(timezone.utc)
            
            return TaskResult(
                task_id=self.id,
                status=self.status,
                error=str(e),
                execution_time=(end_time - start_time).total_seconds(),
                started_at=start_time,
                completed_at=end_time
            )
    
    def _resolve_inputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve input variables from context."""
        resolved = {}
        for key, value in self.inputs.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # Template variable
                var_path = value[2:-2].strip()
                resolved_value = self._get_nested_value(context, var_path)
                resolved[key] = resolved_value
            else:
                resolved[key] = value
        return resolved
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                raise ValidationError(f"Variable {path} not found in context")
        return current


class Workflow:
    """Represents a workflow composed of tasks."""
    
    def __init__(self, name: str, description: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.tasks: Dict[str, Task] = {}
        self.status = WorkflowStatus.PENDING
        self.created_at = datetime.now(timezone.utc)
        
    def add_task(self, task: Task) -> None:
        """Add a task to the workflow."""
        if task.name in self.tasks:
            raise WorkflowError(f"Task with name '{task.name}' already exists")
        self.tasks[task.name] = task
        
    def remove_task(self, task_name: str) -> None:
        """Remove a task from the workflow."""
        if task_name not in self.tasks:
            raise WorkflowError(f"Task '{task_name}' not found")
        
        # Check if other tasks depend on this task
        dependent_tasks = [
            name for name, task in self.tasks.items()
            if task_name in task.dependencies
        ]
        if dependent_tasks:
            raise WorkflowError(
                f"Cannot remove task '{task_name}' - "
                f"it is a dependency for: {', '.join(dependent_tasks)}"
            )
        
        del self.tasks[task_name]
        
    def set_dependencies(self, *dependencies) -> None:
        """Set dependencies between tasks."""
        # Dependencies are set using the >> operator in Task.__rshift__
        pass
        
    def validate(self) -> None:
        """Validate the workflow structure."""
        if not self.tasks:
            raise ValidationError("Workflow must contain at least one task")
        
        # Check for circular dependencies
        self._check_circular_dependencies()
        
        # Validate task dependencies exist
        for task_name, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    raise ValidationError(
                        f"Task '{task_name}' depends on non-existent task '{dep}'"
                    )
    
    def _check_circular_dependencies(self) -> None:
        """Check for circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for dep in self.tasks[node].dependencies:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for task_name in self.tasks:
            if task_name not in visited:
                if has_cycle(task_name):
                    raise ValidationError("Circular dependency detected in workflow")
    
    def get_execution_order(self) -> List[str]:
        """Get topological order of tasks for execution."""
        in_degree = {name: 0 for name in self.tasks}
        
        # Calculate in-degrees
        for task in self.tasks.values():
            for dep in task.dependencies:
                in_degree[task.name] += 1
        
        # Find tasks with no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees for dependent tasks
            for task_name, task in self.tasks.items():
                if current in task.dependencies:
                    in_degree[task_name] -= 1
                    if in_degree[task_name] == 0:
                        queue.append(task_name)
        
        if len(result) != len(self.tasks):
            raise WorkflowError("Unable to determine execution order - circular dependency")
        
        return result
    
    async def execute(self, inputs: Dict[str, Any] = None) -> WorkflowResult:
        """Execute the workflow."""
        start_time = datetime.now(timezone.utc)
        self.status = WorkflowStatus.RUNNING
        
        try:
            # Validate workflow before execution
            self.validate()
            
            # Initialize context with inputs
            context = inputs.copy() if inputs else {}
            context['workflow_id'] = self.id
            context['workflow_name'] = self.name
            
            # Get execution order
            execution_order = self.get_execution_order()
            
            # Execute tasks
            task_results = {}
            completed_tasks = set()
            
            for task_name in execution_order:
                task = self.tasks[task_name]
                
                # Check if task can be executed
                if not task.can_execute(completed_tasks):
                    task_results[task_name] = TaskResult(
                        task_id=task.id,
                        status=TaskStatus.SKIPPED
                    )
                    continue
                
                # Execute task
                result = await task.execute(context)
                task_results[task_name] = result
                
                if result.status == TaskStatus.COMPLETED:
                    # Add task outputs to context
                    context[task_name] = result.outputs
                    completed_tasks.add(task_name)
                elif result.status == TaskStatus.FAILED:
                    # Handle task failure
                    self.status = WorkflowStatus.FAILED
                    break
            
            # Determine final status
            if self.status != WorkflowStatus.FAILED:
                if all(r.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED] 
                       for r in task_results.values()):
                    self.status = WorkflowStatus.COMPLETED
                else:
                    self.status = WorkflowStatus.FAILED
            
            end_time = datetime.now(timezone.utc)
            
            return WorkflowResult(
                workflow_id=self.id,
                status=self.status,
                outputs=context,
                task_results=task_results,
                execution_time=(end_time - start_time).total_seconds(),
                started_at=start_time,
                completed_at=end_time
            )
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            end_time = datetime.now(timezone.utc)
            
            return WorkflowResult(
                workflow_id=self.id,
                status=self.status,
                error=str(e),
                execution_time=(end_time - start_time).total_seconds(),
                started_at=start_time,
                completed_at=end_time
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'tasks': {
                name: {
                    'id': task.id,
                    'name': task.name,
                    'inputs': task.inputs,
                    'outputs': task.outputs,
                    'dependencies': task.dependencies,
                    'timeout': task.timeout,
                    'retries': task.retries,
                    'status': task.status.value
                }
                for name, task in self.tasks.items()
            }
        }
