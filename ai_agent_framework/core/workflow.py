"""
Workflow engine for orchestrating tasks in DAG format
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict, deque

from .task import Task, TaskStatus, TaskResult


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class WorkflowConfig:
    """Workflow configuration"""
    name: str
    description: Optional[str] = None
    max_parallel_tasks: int = 10
    failure_strategy: str = "fail_fast"  # fail_fast, continue_on_failure, retry_failed
    timeout: float = 3600.0  # 1 hour
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Workflow execution result"""
    workflow_id: str
    status: WorkflowStatus
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    failed_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time": self.execution_time,
            "task_results": {k: v.to_dict() for k, v in self.task_results.items()},
            "failed_tasks": self.failed_tasks,
            "completed_tasks": self.completed_tasks,
        }


class Workflow:
    """
    Workflow class for defining and executing DAG-based task workflows.
    
    A workflow consists of tasks with dependencies that form a Directed
    Acyclic Graph (DAG). Tasks are executed in topological order with
    support for parallel execution where possible.
    """
    
    def __init__(self, config: WorkflowConfig):
        self.id = str(uuid4())
        self.config = config
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.dependents: Dict[str, Set[str]] = defaultdict(set)
        self.status = WorkflowStatus.PENDING
        self.result: Optional[WorkflowResult] = None
        self.created_at = datetime.utcnow()
        
    def add_task(self, task: Task, dependencies: Optional[List[str]] = None) -> str:
        """
        Add a task to the workflow with optional dependencies
        
        Args:
            task: Task to add
            dependencies: List of task IDs this task depends on
            
        Returns:
            Task ID
        """
        dependencies = dependencies or []
        
        # Validate dependencies exist
        for dep_id in dependencies:
            if dep_id not in self.tasks:
                raise ValueError(f"Dependency task {dep_id} not found")
        
        # Add task
        self.tasks[task.id] = task
        self.dependencies[task.id] = set(dependencies)
        
        # Update dependents
        for dep_id in dependencies:
            self.dependents[dep_id].add(task.id)
        
        return task.id
    
    def remove_task(self, task_id: str):
        """Remove a task from the workflow"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        # Remove dependencies and dependents
        for dep_id in self.dependencies[task_id]:
            self.dependents[dep_id].discard(task_id)
        
        for dependent_id in self.dependents[task_id]:
            self.dependencies[dependent_id].discard(task_id)
        
        # Remove task
        del self.tasks[task_id]
        del self.dependencies[task_id]
        del self.dependents[task_id]
    
    def validate_dag(self) -> bool:
        """
        Validate that the workflow forms a valid DAG (no cycles)
        
        Returns:
            True if valid DAG, False otherwise
        """
        # Use Kahn's algorithm for cycle detection
        in_degree = {task_id: len(deps) for task_id, deps in self.dependencies.items()}
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        processed = 0
        
        while queue:
            task_id = queue.popleft()
            processed += 1
            
            for dependent_id in self.dependents[task_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
        
        return processed == len(self.tasks)
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get tasks in topological order grouped by execution level
        
        Returns:
            List of lists, where each inner list contains tasks that can
            be executed in parallel
        """
        if not self.validate_dag():
            raise ValueError("Workflow contains cycles")
        
        in_degree = {task_id: len(deps) for task_id, deps in self.dependencies.items()}
        levels = []
        
        while any(degree == 0 for degree in in_degree.values()):
            # Find all tasks with no dependencies
            current_level = [
                task_id for task_id, degree in in_degree.items() 
                if degree == 0
            ]
            
            if not current_level:
                break
                
            levels.append(current_level)
            
            # Remove these tasks and update dependencies
            for task_id in current_level:
                in_degree[task_id] = -1  # Mark as processed
                for dependent_id in self.dependents[task_id]:
                    if in_degree[dependent_id] > 0:
                        in_degree[dependent_id] -= 1
        
        return levels
    
    async def execute(self, input_data: Any = None, context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute the workflow
        
        Args:
            input_data: Initial input data for the workflow
            context: Execution context
            
        Returns:
            WorkflowResult with execution details
        """
        context = context or {}
        
        self.status = WorkflowStatus.RUNNING
        self.result = WorkflowResult(
            workflow_id=self.id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        
        try:
            # Get execution order
            execution_levels = self.get_execution_order()
            
            # Execute tasks level by level
            for level in execution_levels:
                if self.status != WorkflowStatus.RUNNING:
                    break
                
                # Execute tasks in parallel within each level
                semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)
                tasks_to_run = []
                
                for task_id in level:
                    if self.status != WorkflowStatus.RUNNING:
                        break
                    
                    task = self.tasks[task_id]
                    tasks_to_run.append(
                        self._execute_task_with_semaphore(
                            semaphore, task, input_data, context
                        )
                    )
                
                # Wait for all tasks in this level to complete
                if tasks_to_run:
                    results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
                    
                    # Process results
                    for i, result in enumerate(results):
                        task_id = level[i]
                        
                        if isinstance(result, Exception):
                            self.result.failed_tasks.append(task_id)
                            if self.config.failure_strategy == "fail_fast":
                                self.status = WorkflowStatus.FAILED
                                break
                        else:
                            self.result.task_results[task_id] = result
                            self.result.completed_tasks.append(task_id)
            
            # Determine final status
            if self.status == WorkflowStatus.RUNNING:
                if self.result.failed_tasks and self.config.failure_strategy == "fail_fast":
                    self.status = WorkflowStatus.FAILED
                else:
                    self.status = WorkflowStatus.COMPLETED
            
            self.result.status = self.status
            self.result.end_time = datetime.utcnow()
            self.result.execution_time = (
                self.result.end_time - self.result.start_time
            ).total_seconds()
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            self.result.status = WorkflowStatus.FAILED
            self.result.end_time = datetime.utcnow()
            self.result.execution_time = (
                self.result.end_time - self.result.start_time
            ).total_seconds()
            raise e
        
        return self.result
    
    async def _execute_task_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        task: Task, 
        input_data: Any,
        context: Dict[str, Any]
    ) -> TaskResult:
        """Execute a task with semaphore control"""
        async with semaphore:
            return await task.run(input_data, context)
    
    def pause(self):
        """Pause workflow execution"""
        if self.status == WorkflowStatus.RUNNING:
            self.status = WorkflowStatus.PAUSED
    
    def resume(self):
        """Resume workflow execution"""
        if self.status == WorkflowStatus.PAUSED:
            self.status = WorkflowStatus.RUNNING
    
    def cancel(self):
        """Cancel workflow execution"""
        if self.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]:
            self.status = WorkflowStatus.CANCELLED
            # Cancel all running tasks
            for task in self.tasks.values():
                task.cancel()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "workflow_id": self.id,
            "name": self.config.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "task_count": len(self.tasks),
            "completed_tasks": len(self.result.completed_tasks) if self.result else 0,
            "failed_tasks": len(self.result.failed_tasks) if self.result else 0,
            "result": self.result.to_dict() if self.result else None,
        }


class WorkflowEngine:
    """
    Workflow execution engine that manages multiple workflows
    """
    
    def __init__(self, max_concurrent_workflows: int = 5):
        self.workflows: Dict[str, Workflow] = {}
        self.max_concurrent_workflows = max_concurrent_workflows
        self._semaphore = asyncio.Semaphore(max_concurrent_workflows)
    
    def register_workflow(self, workflow: Workflow) -> str:
        """Register a workflow for execution"""
        self.workflows[workflow.id] = workflow
        return workflow.id
    
    async def execute_workflow(
        self, 
        workflow_id: str, 
        input_data: Any = None,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """Execute a registered workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        async with self._semaphore:
            return await workflow.execute(input_data, context)
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        return self.workflows[workflow_id].get_status()
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all registered workflows"""
        return [workflow.get_status() for workflow in self.workflows.values()]