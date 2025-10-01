"""
Advanced scheduler for task and workflow execution
"""

from typing import Any, Dict, List, Optional, Callable, Set
from uuid import uuid4
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import heapq
from collections import defaultdict, deque
import json

from ..core.task import Task, TaskStatus, TaskResult, TaskPriority
from ..core.workflow import Workflow, WorkflowStatus


class ScheduleType(Enum):
    """Types of scheduling"""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    CRON = "cron"
    INTERVAL = "interval"
    EVENT_DRIVEN = "event_driven"


class ResourceType(Enum):
    """Types of resources"""
    CPU = "cpu"
    MEMORY = "memory" 
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class ResourceRequirements:
    """Resource requirements for a task"""
    cpu_cores: float = 1.0
    memory_mb: int = 512
    gpu_count: int = 0
    network_bandwidth_mbps: float = 100.0
    storage_gb: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_count": self.gpu_count,
            "network_bandwidth_mbps": self.network_bandwidth_mbps,
            "storage_gb": self.storage_gb,
        }


@dataclass
class ScheduledItem:
    """Item in the scheduler queue"""
    id: str
    item: Any  # Task or Workflow
    priority: int
    scheduled_time: datetime
    resource_requirements: ResourceRequirements
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __lt__(self, other):
        """Comparison for priority queue"""
        if self.scheduled_time == other.scheduled_time:
            return self.priority > other.priority  # Higher priority first
        return self.scheduled_time < other.scheduled_time


@dataclass
class ExecutorNode:
    """Executor node for distributed execution"""
    id: str
    name: str
    address: str
    available_resources: ResourceRequirements
    used_resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    status: str = "active"  # active, inactive, failed
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    
    def has_available_resources(self, required: ResourceRequirements) -> bool:
        """Check if node has required resources available"""
        available_cpu = self.available_resources.cpu_cores - self.used_resources.cpu_cores
        available_memory = self.available_resources.memory_mb - self.used_resources.memory_mb
        available_gpu = self.available_resources.gpu_count - self.used_resources.gpu_count
        
        return (
            available_cpu >= required.cpu_cores and
            available_memory >= required.memory_mb and
            available_gpu >= required.gpu_count
        )
    
    def allocate_resources(self, required: ResourceRequirements):
        """Allocate resources on this node"""
        self.used_resources.cpu_cores += required.cpu_cores
        self.used_resources.memory_mb += required.memory_mb
        self.used_resources.gpu_count += required.gpu_count
        self.used_resources.network_bandwidth_mbps += required.network_bandwidth_mbps
        self.used_resources.storage_gb += required.storage_gb
    
    def release_resources(self, required: ResourceRequirements):
        """Release resources on this node"""
        self.used_resources.cpu_cores = max(0, self.used_resources.cpu_cores - required.cpu_cores)
        self.used_resources.memory_mb = max(0, self.used_resources.memory_mb - required.memory_mb)
        self.used_resources.gpu_count = max(0, self.used_resources.gpu_count - required.gpu_count)
        self.used_resources.network_bandwidth_mbps = max(0, self.used_resources.network_bandwidth_mbps - required.network_bandwidth_mbps)
        self.used_resources.storage_gb = max(0, self.used_resources.storage_gb - required.storage_gb)


class Scheduler:
    """
    Advanced scheduler for managing task and workflow execution with
    resource management, priority scheduling, and distributed execution.
    """
    
    def __init__(self, max_concurrent_executions: int = 10):
        self.max_concurrent_executions = max_concurrent_executions
        self.scheduled_items: List[ScheduledItem] = []  # Priority queue
        self.running_items: Dict[str, ScheduledItem] = {}
        self.completed_items: Dict[str, Dict[str, Any]] = {}
        self.failed_items: Dict[str, Dict[str, Any]] = {}
        
        # Resource management
        self.executor_nodes: Dict[str, ExecutorNode] = {}
        self.resource_allocations: Dict[str, str] = {}  # item_id -> node_id
        
        # Scheduling state
        self._scheduler_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        
        # Event hooks
        self.on_item_scheduled: Optional[Callable] = None
        self.on_item_started: Optional[Callable] = None
        self.on_item_completed: Optional[Callable] = None
        self.on_item_failed: Optional[Callable] = None
    
    async def start(self):
        """Start the scheduler"""
        if self._is_running:
            return
        
        self._is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop(self):
        """Stop the scheduler"""
        self._is_running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
    
    def schedule_task(
        self,
        task: Task,
        schedule_time: Optional[datetime] = None,
        priority: Optional[TaskPriority] = None,
        resource_requirements: Optional[ResourceRequirements] = None,
        schedule_type: ScheduleType = ScheduleType.IMMEDIATE,
        schedule_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Schedule a task for execution"""
        
        scheduled_item = ScheduledItem(
            id=str(uuid4()),
            item=task,
            priority=priority.value if priority else task.config.priority.value,
            scheduled_time=schedule_time or datetime.utcnow(),
            resource_requirements=resource_requirements or ResourceRequirements(),
            schedule_type=schedule_type,
            schedule_config=schedule_config or {},
            max_retries=task.config.max_retries
        )
        
        heapq.heappush(self.scheduled_items, scheduled_item)
        
        if self.on_item_scheduled:
            asyncio.create_task(self.on_item_scheduled(scheduled_item))
        
        return scheduled_item.id
    
    def schedule_workflow(
        self,
        workflow: Workflow,
        schedule_time: Optional[datetime] = None,
        priority: int = 2,
        resource_requirements: Optional[ResourceRequirements] = None,
        schedule_type: ScheduleType = ScheduleType.IMMEDIATE,
        schedule_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Schedule a workflow for execution"""
        
        scheduled_item = ScheduledItem(
            id=str(uuid4()),
            item=workflow,
            priority=priority,
            scheduled_time=schedule_time or datetime.utcnow(),
            resource_requirements=resource_requirements or ResourceRequirements(),
            schedule_type=schedule_type,
            schedule_config=schedule_config or {}
        )
        
        heapq.heappush(self.scheduled_items, scheduled_item)
        
        if self.on_item_scheduled:
            asyncio.create_task(self.on_item_scheduled(scheduled_item))
        
        return scheduled_item.id
    
    def cancel_scheduled_item(self, item_id: str) -> bool:
        """Cancel a scheduled item"""
        # Remove from scheduled items
        for i, item in enumerate(self.scheduled_items):
            if item.id == item_id:
                del self.scheduled_items[i]
                heapq.heapify(self.scheduled_items)
                return True
        
        # Cancel running item
        if item_id in self.running_items:
            item = self.running_items[item_id]
            if isinstance(item.item, Task):
                item.item.cancel()
            elif isinstance(item.item, Workflow):
                item.item.cancel()
            return True
        
        return False
    
    def add_executor_node(self, node: ExecutorNode):
        """Add an executor node"""
        self.executor_nodes[node.id] = node
    
    def remove_executor_node(self, node_id: str):
        """Remove an executor node"""
        if node_id in self.executor_nodes:
            # Release any allocated resources
            items_to_reschedule = []
            for item_id, allocated_node_id in self.resource_allocations.items():
                if allocated_node_id == node_id:
                    items_to_reschedule.append(item_id)
            
            # TODO: Implement rescheduling logic
            
            del self.executor_nodes[node_id]
    
    def _find_suitable_executor(self, requirements: ResourceRequirements) -> Optional[ExecutorNode]:
        """Find a suitable executor node for the given resource requirements"""
        suitable_nodes = []
        
        for node in self.executor_nodes.values():
            if node.status == "active" and node.has_available_resources(requirements):
                suitable_nodes.append(node)
        
        if not suitable_nodes:
            return None
        
        # Choose node with most available resources (best fit)
        return max(suitable_nodes, key=lambda n: (
            n.available_resources.cpu_cores - n.used_resources.cpu_cores,
            n.available_resources.memory_mb - n.used_resources.memory_mb
        ))
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._is_running:
            try:
                await self._process_scheduled_items()
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Scheduler error: {e}")
                await asyncio.sleep(5)
    
    async def _process_scheduled_items(self):
        """Process items ready for execution"""
        now = datetime.utcnow()
        ready_items = []
        
        # Find items ready for execution
        while (self.scheduled_items and 
               self.scheduled_items[0].scheduled_time <= now and
               len(self.running_items) < self.max_concurrent_executions):
            
            item = heapq.heappop(self.scheduled_items)
            
            # Check resource availability
            executor_node = self._find_suitable_executor(item.resource_requirements)
            
            if executor_node:
                # Allocate resources
                executor_node.allocate_resources(item.resource_requirements)
                self.resource_allocations[item.id] = executor_node.id
                ready_items.append(item)
            else:
                # No resources available, put back in queue with delay
                item.scheduled_time = now + timedelta(seconds=30)
                heapq.heappush(self.scheduled_items, item)
                break
        
        # Execute ready items
        for item in ready_items:
            self.running_items[item.id] = item
            asyncio.create_task(self._execute_item(item))
    
    async def _execute_item(self, scheduled_item: ScheduledItem):
        """Execute a scheduled item"""
        try:
            async with self._execution_semaphore:
                if self.on_item_started:
                    await self.on_item_started(scheduled_item)
                
                # Execute the item
                if isinstance(scheduled_item.item, Task):
                    result = await scheduled_item.item.run({})
                elif isinstance(scheduled_item.item, Workflow):
                    result = await scheduled_item.item.execute()
                else:
                    raise ValueError(f"Unknown item type: {type(scheduled_item.item)}")
                
                # Handle success
                self.completed_items[scheduled_item.id] = {
                    "item": scheduled_item,
                    "result": result,
                    "completed_at": datetime.utcnow()
                }
                
                if self.on_item_completed:
                    await self.on_item_completed(scheduled_item, result)
        
        except Exception as e:
            # Handle failure
            scheduled_item.retry_count += 1
            
            if scheduled_item.retry_count <= scheduled_item.max_retries:
                # Reschedule for retry
                scheduled_item.scheduled_time = datetime.utcnow() + timedelta(
                    seconds=2 ** scheduled_item.retry_count
                )
                heapq.heappush(self.scheduled_items, scheduled_item)
            else:
                # Max retries exceeded
                self.failed_items[scheduled_item.id] = {
                    "item": scheduled_item,
                    "error": str(e),
                    "failed_at": datetime.utcnow()
                }
                
                if self.on_item_failed:
                    await self.on_item_failed(scheduled_item, e)
        
        finally:
            # Clean up
            if scheduled_item.id in self.running_items:
                del self.running_items[scheduled_item.id]
            
            # Release resources
            if scheduled_item.id in self.resource_allocations:
                node_id = self.resource_allocations[scheduled_item.id]
                if node_id in self.executor_nodes:
                    self.executor_nodes[node_id].release_resources(
                        scheduled_item.resource_requirements
                    )
                del self.resource_allocations[scheduled_item.id]
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "is_running": self._is_running,
            "scheduled_items": len(self.scheduled_items),
            "running_items": len(self.running_items),
            "completed_items": len(self.completed_items),
            "failed_items": len(self.failed_items),
            "executor_nodes": len(self.executor_nodes),
            "active_nodes": len([
                n for n in self.executor_nodes.values() 
                if n.status == "active"
            ]),
        }
    
    def get_scheduled_items(self) -> List[Dict[str, Any]]:
        """Get list of scheduled items"""
        return [
            {
                "id": item.id,
                "type": type(item.item).__name__,
                "priority": item.priority,
                "scheduled_time": item.scheduled_time.isoformat(),
                "schedule_type": item.schedule_type.value,
                "retry_count": item.retry_count,
                "resource_requirements": item.resource_requirements.to_dict(),
            }
            for item in self.scheduled_items
        ]
    
    def get_running_items(self) -> List[Dict[str, Any]]:
        """Get list of running items"""
        return [
            {
                "id": item.id,
                "type": type(item.item).__name__,
                "priority": item.priority,
                "started_at": item.scheduled_time.isoformat(),
                "resource_requirements": item.resource_requirements.to_dict(),
                "allocated_node": self.resource_allocations.get(item.id),
            }
            for item in self.running_items.values()
        ]
    
    def get_executor_nodes(self) -> List[Dict[str, Any]]:
        """Get list of executor nodes"""
        return [
            {
                "id": node.id,
                "name": node.name,
                "address": node.address,
                "status": node.status,
                "available_resources": node.available_resources.to_dict(),
                "used_resources": node.used_resources.to_dict(),
                "last_heartbeat": node.last_heartbeat.isoformat(),
            }
            for node in self.executor_nodes.values()
        ]