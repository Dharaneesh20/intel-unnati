"""
Task implementation for the AI Agent Framework
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from uuid import uuid4
from datetime import datetime
from dataclasses import dataclass, field
import asyncio
import json


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskConfig:
    """Task configuration"""
    name: str
    description: Optional[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    timeout: float = 60.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time": self.execution_time,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }


class Task(ABC):
    """
    Base Task class for the AI Agent Framework.
    
    Tasks are the basic units of work that can be executed independently
    or as part of a workflow. They support retries, timeouts, dependencies,
    and provide detailed execution tracking.
    """
    
    def __init__(self, config: TaskConfig):
        self.id = str(uuid4())
        self.config = config
        self.status = TaskStatus.PENDING
        self.result: Optional[TaskResult] = None
        self.retry_count = 0
        self.created_at = datetime.utcnow()
        
    async def run(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        """
        Execute the task with retry logic and timeout handling
        """
        context = context or {}
        
        self.result = TaskResult(
            task_id=self.id,
            status=TaskStatus.RUNNING,
            start_time=datetime.utcnow(),
            metadata=context
        )
        
        for attempt in range(self.config.max_retries + 1):
            try:
                self.status = TaskStatus.RUNNING
                if attempt > 0:
                    self.status = TaskStatus.RETRYING
                    self.retry_count = attempt
                    
                # Execute with timeout
                result = await asyncio.wait_for(
                    self.execute(input_data, context),
                    timeout=self.config.timeout
                )
                
                # Success
                self.status = TaskStatus.COMPLETED
                self.result.status = TaskStatus.COMPLETED
                self.result.result = result
                self.result.end_time = datetime.utcnow()
                self.result.execution_time = (
                    self.result.end_time - self.result.start_time
                ).total_seconds()
                self.result.retry_count = self.retry_count
                
                return self.result
                
            except asyncio.TimeoutError as e:
                error_msg = f"Task timed out after {self.config.timeout}s"
                if attempt < self.config.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    self._handle_failure(error_msg)
                    break
                    
            except Exception as e:
                error_msg = str(e)
                if attempt < self.config.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    self._handle_failure(error_msg)
                    break
        
        return self.result
    
    def _handle_failure(self, error_msg: str):
        """Handle task failure"""
        self.status = TaskStatus.FAILED
        self.result.status = TaskStatus.FAILED
        self.result.error = error_msg
        self.result.end_time = datetime.utcnow()
        self.result.execution_time = (
            self.result.end_time - self.result.start_time
        ).total_seconds()
        self.result.retry_count = self.retry_count
    
    @abstractmethod
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """
        Core task execution logic - must be implemented by subclasses
        """
        pass
    
    def cancel(self):
        """Cancel task execution"""
        if self.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            self.status = TaskStatus.CANCELLED
            if self.result:
                self.result.status = TaskStatus.CANCELLED
    
    def get_status(self) -> Dict[str, Any]:
        """Get current task status"""
        return {
            "task_id": self.id,
            "name": self.config.name,
            "status": self.status.value,
            "priority": self.config.priority.value,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "dependencies": self.config.dependencies,
            "result": self.result.to_dict() if self.result else None,
        }


class FunctionTask(Task):
    """
    Task that wraps a function for execution
    """
    
    def __init__(self, name: str, func: Callable, **kwargs):
        config = TaskConfig(name=name, **kwargs)
        super().__init__(config)
        self.func = func
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute the wrapped function"""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(input_data, context)
        else:
            return self.func(input_data, context)


class ShellTask(Task):
    """
    Task that executes shell commands
    """
    
    def __init__(self, name: str, command: str, **kwargs):
        config = TaskConfig(name=name, **kwargs)
        super().__init__(config)
        self.command = command
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute shell command"""
        process = await asyncio.create_subprocess_shell(
            self.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {stderr.decode()}")
        
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
        }


class HTTPTask(Task):
    """
    Task that makes HTTP requests
    """
    
    def __init__(self, name: str, url: str, method: str = "GET", **kwargs):
        config = TaskConfig(name=name, **kwargs)
        super().__init__(config)
        self.url = url
        self.method = method.upper()
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Make HTTP request"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                self.method, 
                self.url, 
                json=input_data if self.method in ["POST", "PUT", "PATCH"] else None
            ) as response:
                result = {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "url": str(response.url),
                }
                
                try:
                    result["data"] = await response.json()
                except:
                    result["data"] = await response.text()
                
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {result['data']}")
                
                return result