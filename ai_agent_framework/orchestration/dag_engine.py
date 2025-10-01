"""
DAG (Directed Acyclic Graph) Engine for workflow orchestration
"""

from typing import Any, Dict, List, Optional, Set, Callable, Awaitable
from uuid import uuid4
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict, deque
import json

from ..core.task import Task, TaskStatus, TaskResult
from ..core.workflow import Workflow, WorkflowStatus, WorkflowResult


class DAGNodeType(Enum):
    """Types of nodes in the DAG"""
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENCE = "sequence"
    LOOP = "loop"


@dataclass
class DAGNode:
    """A node in the DAG"""
    id: str
    name: str
    node_type: DAGNodeType
    task: Optional[Task] = None
    condition: Optional[Callable[[Any], Awaitable[bool]]] = None
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAGExecution:
    """DAG execution state"""
    dag_id: str
    execution_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    node_results: Dict[str, Any] = field(default_factory=dict)
    failed_nodes: List[str] = field(default_factory=list)
    completed_nodes: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class DAGEngine:
    """
    Advanced DAG execution engine with support for conditional flows,
    parallel execution, loops, and complex orchestration patterns.
    """
    
    def __init__(self, max_concurrent_nodes: int = 10):
        self.max_concurrent_nodes = max_concurrent_nodes
        self.dags: Dict[str, "DAG"] = {}
        self.executions: Dict[str, DAGExecution] = {}
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_nodes)
    
    def register_dag(self, dag: "DAG") -> str:
        """Register a DAG for execution"""
        self.dags[dag.id] = dag
        return dag.id
    
    async def execute_dag(
        self, 
        dag_id: str, 
        input_data: Any = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DAGExecution:
        """Execute a registered DAG"""
        if dag_id not in self.dags:
            raise ValueError(f"DAG {dag_id} not found")
        
        dag = self.dags[dag_id]
        execution_id = str(uuid4())
        
        execution = DAGExecution(
            dag_id=dag_id,
            execution_id=execution_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.utcnow(),
            context=context or {}
        )
        
        self.executions[execution_id] = execution
        
        try:
            await self._execute_dag_nodes(dag, execution, input_data)
            
            if execution.failed_nodes:
                execution.status = WorkflowStatus.FAILED
            else:
                execution.status = WorkflowStatus.COMPLETED
                
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.context["error"] = str(e)
        
        execution.end_time = datetime.utcnow()
        return execution
    
    async def _execute_dag_nodes(
        self, 
        dag: "DAG", 
        execution: DAGExecution, 
        input_data: Any
    ):
        """Execute DAG nodes in topological order"""
        # Get execution order
        execution_levels = dag.get_execution_order()
        current_data = input_data
        
        for level in execution_levels:
            if execution.status != WorkflowStatus.RUNNING:
                break
            
            # Execute nodes in parallel within each level
            tasks_to_run = []
            
            for node_id in level:
                if execution.status != WorkflowStatus.RUNNING:
                    break
                
                node = dag.nodes[node_id]
                tasks_to_run.append(
                    self._execute_node_with_semaphore(
                        node, execution, current_data
                    )
                )
            
            # Wait for all nodes in this level to complete
            if tasks_to_run:
                results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
                
                # Process results
                level_results = {}
                for i, result in enumerate(results):
                    node_id = level[i]
                    
                    if isinstance(result, Exception):
                        execution.failed_nodes.append(node_id)
                        execution.node_results[node_id] = {
                            "error": str(result),
                            "status": "failed"
                        }
                    else:
                        execution.completed_nodes.append(node_id)
                        execution.node_results[node_id] = result
                        level_results[node_id] = result
                
                # Update current_data for next level
                if level_results:
                    current_data = level_results
    
    async def _execute_node_with_semaphore(
        self, 
        node: DAGNode, 
        execution: DAGExecution,
        input_data: Any
    ) -> Any:
        """Execute a single DAG node with semaphore control"""
        async with self._execution_semaphore:
            return await self._execute_node(node, execution, input_data)
    
    async def _execute_node(
        self, 
        node: DAGNode, 
        execution: DAGExecution,
        input_data: Any
    ) -> Any:
        """Execute a single DAG node"""
        if node.node_type == DAGNodeType.TASK and node.task:
            # Execute task
            result = await node.task.run(input_data, execution.context)
            return result
        
        elif node.node_type == DAGNodeType.CONDITION and node.condition:
            # Evaluate condition
            result = await node.condition(input_data)
            return {"condition_result": result, "data": input_data}
        
        elif node.node_type == DAGNodeType.PARALLEL:
            # Execute parallel branches (placeholder implementation)
            return {"parallel_result": input_data}
        
        elif node.node_type == DAGNodeType.SEQUENCE:
            # Execute sequence (placeholder implementation)
            return {"sequence_result": input_data}
        
        elif node.node_type == DAGNodeType.LOOP:
            # Execute loop (placeholder implementation)
            return {"loop_result": input_data}
        
        else:
            # Default: pass through data
            return input_data
    
    def get_execution_status(self, execution_id: str) -> Optional[DAGExecution]:
        """Get execution status"""
        return self.executions.get(execution_id)
    
    def list_executions(self, dag_id: Optional[str] = None) -> List[DAGExecution]:
        """List executions, optionally filtered by DAG ID"""
        executions = list(self.executions.values())
        
        if dag_id:
            executions = [e for e in executions if e.dag_id == dag_id]
        
        return executions


class DAG:
    """
    DAG (Directed Acyclic Graph) definition for complex workflows
    """
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.id = str(uuid4())
        self.name = name
        self.description = description
        self.nodes: Dict[str, DAGNode] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self.created_at = datetime.utcnow()
    
    def add_task_node(
        self, 
        name: str, 
        task: Task, 
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Add a task node to the DAG"""
        node_id = str(uuid4())
        dependencies = dependencies or []
        
        node = DAGNode(
            id=node_id,
            name=name,
            node_type=DAGNodeType.TASK,
            task=task,
            dependencies=set(dependencies)
        )
        
        self.nodes[node_id] = node
        
        # Add edges
        for dep_id in dependencies:
            if dep_id not in self.nodes:
                raise ValueError(f"Dependency node {dep_id} not found")
            self.edges[dep_id].add(node_id)
        
        return node_id
    
    def add_condition_node(
        self,
        name: str,
        condition: Callable[[Any], Awaitable[bool]],
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Add a condition node to the DAG"""
        node_id = str(uuid4())
        dependencies = dependencies or []
        
        node = DAGNode(
            id=node_id,
            name=name,
            node_type=DAGNodeType.CONDITION,
            condition=condition,
            dependencies=set(dependencies)
        )
        
        self.nodes[node_id] = node
        
        # Add edges
        for dep_id in dependencies:
            if dep_id not in self.nodes:
                raise ValueError(f"Dependency node {dep_id} not found")
            self.edges[dep_id].add(node_id)
        
        return node_id
    
    def add_parallel_node(
        self,
        name: str,
        parallel_nodes: List[str],
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Add a parallel execution node"""
        node_id = str(uuid4())
        dependencies = dependencies or []
        
        node = DAGNode(
            id=node_id,
            name=name,
            node_type=DAGNodeType.PARALLEL,
            dependencies=set(dependencies),
            metadata={"parallel_nodes": parallel_nodes}
        )
        
        self.nodes[node_id] = node
        
        # Add edges
        for dep_id in dependencies:
            if dep_id not in self.nodes:
                raise ValueError(f"Dependency node {dep_id} not found")
            self.edges[dep_id].add(node_id)
        
        return node_id
    
    def remove_node(self, node_id: str):
        """Remove a node from the DAG"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        # Remove from edges
        for source, targets in self.edges.items():
            targets.discard(node_id)
        
        if node_id in self.edges:
            del self.edges[node_id]
        
        # Remove node
        del self.nodes[node_id]
    
    def validate_dag(self) -> bool:
        """Validate that the DAG has no cycles"""
        # Use Kahn's algorithm for cycle detection
        in_degree = {node_id: len(node.dependencies) for node_id, node in self.nodes.items()}
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        processed = 0
        
        while queue:
            node_id = queue.popleft()
            processed += 1
            
            for target_id in self.edges[node_id]:
                in_degree[target_id] -= 1
                if in_degree[target_id] == 0:
                    queue.append(target_id)
        
        return processed == len(self.nodes)
    
    def get_execution_order(self) -> List[List[str]]:
        """Get nodes in topological order grouped by execution level"""
        if not self.validate_dag():
            raise ValueError("DAG contains cycles")
        
        in_degree = {node_id: len(node.dependencies) for node_id, node in self.nodes.items()}
        levels = []
        
        while any(degree == 0 for degree in in_degree.values()):
            # Find all nodes with no dependencies
            current_level = [
                node_id for node_id, degree in in_degree.items() 
                if degree == 0
            ]
            
            if not current_level:
                break
                
            levels.append(current_level)
            
            # Remove these nodes and update dependencies
            for node_id in current_level:
                in_degree[node_id] = -1  # Mark as processed
                for target_id in self.edges[node_id]:
                    if in_degree[target_id] > 0:
                        in_degree[target_id] -= 1
        
        return levels
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DAG to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "nodes": {
                node_id: {
                    "id": node.id,
                    "name": node.name,
                    "type": node.node_type.value,
                    "dependencies": list(node.dependencies),
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": {
                source: list(targets) 
                for source, targets in self.edges.items()
            }
        }
    
    def export_mermaid(self) -> str:
        """Export DAG as Mermaid diagram"""
        lines = ["graph TD"]
        
        # Add nodes
        for node_id, node in self.nodes.items():
            node_label = f"{node.name} ({node.node_type.value})"
            lines.append(f"    {node_id}[\"{node_label}\"]")
        
        # Add edges
        for source, targets in self.edges.items():
            for target in targets:
                lines.append(f"    {source} --> {target}")
        
        return "\n".join(lines)