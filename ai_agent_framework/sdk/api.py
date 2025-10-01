"""
RESTful API for the AI Agent Framework
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from uuid import uuid4

from ..core.agent import Agent, AgentConfig, AgentContext
from ..core.task import Task, TaskConfig, FunctionTask
from ..core.workflow import Workflow, WorkflowConfig, WorkflowEngine
from ..core.memory import MemoryManager, InMemoryStorage
from ..orchestration.scheduler import Scheduler
from ..orchestration.dag_engine import DAGEngine, DAG
from ..monitoring.metrics import MetricsCollector
from ..monitoring.logger import LoggerFactory


class APIResponse:
    """Standard API response format"""
    
    def __init__(
        self, 
        success: bool, 
        data: Any = None, 
        error: Optional[str] = None,
        message: Optional[str] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.message = message
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        response = {
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.data is not None:
            response["data"] = self.data
        if self.error:
            response["error"] = self.error
        if self.message:
            response["message"] = self.message
        
        return response


class FrameworkAPI:
    """
    Main API class for the AI Agent Framework providing RESTful endpoints
    for managing agents, tasks, workflows, and monitoring.
    """
    
    def __init__(self):
        # Core components
        self.workflow_engine = WorkflowEngine()
        self.scheduler = Scheduler()
        self.dag_engine = DAGEngine()
        self.memory_manager = MemoryManager(InMemoryStorage())
        self.metrics_collector = MetricsCollector()
        
        # State management
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.dags: Dict[str, DAG] = {}
        
        # Logging
        self.logger = LoggerFactory.get_logger("api")
        
        # Web application (will be set by web server)
        self.app = None
    
    async def start(self):
        """Start the API server and framework components"""
        await self.scheduler.start()
        await self.memory_manager.start_cleanup_task()
        await self.metrics_collector.start_export_loop()
        await self.logger.info("Framework API started")
    
    async def stop(self):
        """Stop the API server and framework components"""
        await self.scheduler.stop()
        await self.memory_manager.stop_cleanup_task()
        await self.metrics_collector.stop_export_loop()
        await self.logger.info("Framework API stopped")
    
    # Agent Management Endpoints
    
    async def create_agent(self, agent_data: Dict[str, Any]) -> APIResponse:
        """Create a new agent"""
        try:
            # Validate input
            if "name" not in agent_data:
                return APIResponse(False, error="Agent name is required")
            
            # Create agent config
            config = AgentConfig(
                name=agent_data["name"],
                description=agent_data.get("description"),
                max_retries=agent_data.get("max_retries", 3),
                timeout=agent_data.get("timeout", 300.0),
                memory_enabled=agent_data.get("memory_enabled", True),
                guardrails_enabled=agent_data.get("guardrails_enabled", True),
                metrics_enabled=agent_data.get("metrics_enabled", True)
            )
            
            # Create agent (this would typically be a custom implementation)
            from ..core.agent import SimpleAgent
            agent = SimpleAgent(config.name)
            
            # Register agent
            self.agents[agent.id] = agent
            
            await self.logger.info(f"Agent created: {agent.id}", agent_id=agent.id)
            
            return APIResponse(
                True, 
                data={
                    "agent_id": agent.id,
                    "name": config.name,
                    "config": config.__dict__
                },
                message="Agent created successfully"
            )
            
        except Exception as e:
            await self.logger.error(f"Failed to create agent: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    async def get_agent(self, agent_id: str) -> APIResponse:
        """Get agent details"""
        try:
            if agent_id not in self.agents:
                return APIResponse(False, error="Agent not found")
            
            agent = self.agents[agent_id]
            return APIResponse(
                True,
                data=agent.get_status()
            )
            
        except Exception as e:
            await self.logger.error(f"Failed to get agent {agent_id}: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    async def list_agents(self) -> APIResponse:
        """List all agents"""
        try:
            agents_data = []
            for agent in self.agents.values():
                agents_data.append(agent.get_status())
            
            return APIResponse(True, data=agents_data)
            
        except Exception as e:
            await self.logger.error(f"Failed to list agents: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    async def execute_agent(self, agent_id: str, input_data: Any, context: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Execute an agent"""
        try:
            if agent_id not in self.agents:
                return APIResponse(False, error="Agent not found")
            
            agent = self.agents[agent_id]
            
            # Create execution context
            agent_context = AgentContext(
                agent_id=agent_id,
                session_id=str(uuid4()),
                user_id=context.get("user_id") if context else None,
                metadata=context or {}
            )
            
            # Execute agent
            result = await agent.run(agent_context, input_data)
            
            # Record metrics
            execution_time = (agent.end_time - agent.start_time).total_seconds() if agent.start_time and agent.end_time else 0
            self.metrics_collector.record_agent_execution(
                agent.config.name,
                execution_time,
                agent.state.value == "completed"
            )
            
            return APIResponse(
                True,
                data={
                    "result": result,
                    "status": agent.get_status()
                },
                message="Agent executed successfully"
            )
            
        except Exception as e:
            await self.logger.error(f"Failed to execute agent {agent_id}: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    # Task Management Endpoints
    
    async def create_task(self, task_data: Dict[str, Any]) -> APIResponse:
        """Create a new task"""
        try:
            # Validate input
            if "name" not in task_data:
                return APIResponse(False, error="Task name is required")
            
            # Create task config
            config = TaskConfig(
                name=task_data["name"],
                description=task_data.get("description"),
                max_retries=task_data.get("max_retries", 3),
                timeout=task_data.get("timeout", 60.0),
                dependencies=task_data.get("dependencies", [])
            )
            
            # Create task (placeholder - would be custom implementation)
            task = FunctionTask(config.name, lambda x, ctx: x)
            
            # Register task
            self.tasks[task.id] = task
            
            await self.logger.info(f"Task created: {task.id}", task_id=task.id)
            
            return APIResponse(
                True,
                data={
                    "task_id": task.id,
                    "name": config.name,
                    "config": config.__dict__
                },
                message="Task created successfully"
            )
            
        except Exception as e:
            await self.logger.error(f"Failed to create task: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    async def get_task(self, task_id: str) -> APIResponse:
        """Get task details"""
        try:
            if task_id not in self.tasks:
                return APIResponse(False, error="Task not found")
            
            task = self.tasks[task_id]
            return APIResponse(True, data=task.get_status())
            
        except Exception as e:
            await self.logger.error(f"Failed to get task {task_id}: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    async def execute_task(self, task_id: str, input_data: Any, context: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Execute a task"""
        try:
            if task_id not in self.tasks:
                return APIResponse(False, error="Task not found")
            
            task = self.tasks[task_id]
            result = await task.run(input_data, context or {})
            
            return APIResponse(
                True,
                data={
                    "result": result.to_dict(),
                    "status": task.get_status()
                },
                message="Task executed successfully"
            )
            
        except Exception as e:
            await self.logger.error(f"Failed to execute task {task_id}: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    # Workflow Management Endpoints
    
    async def create_workflow(self, workflow_data: Dict[str, Any]) -> APIResponse:
        """Create a new workflow"""
        try:
            # Validate input
            if "name" not in workflow_data:
                return APIResponse(False, error="Workflow name is required")
            
            # Create workflow config
            config = WorkflowConfig(
                name=workflow_data["name"],
                description=workflow_data.get("description"),
                max_parallel_tasks=workflow_data.get("max_parallel_tasks", 10),
                failure_strategy=workflow_data.get("failure_strategy", "fail_fast"),
                timeout=workflow_data.get("timeout", 3600.0)
            )
            
            # Create workflow
            workflow = Workflow(config)
            
            # Register workflow
            self.workflows[workflow.id] = workflow
            self.workflow_engine.register_workflow(workflow)
            
            await self.logger.info(f"Workflow created: {workflow.id}", workflow_id=workflow.id)
            
            return APIResponse(
                True,
                data={
                    "workflow_id": workflow.id,
                    "name": config.name,
                    "config": config.__dict__
                },
                message="Workflow created successfully"
            )
            
        except Exception as e:
            await self.logger.error(f"Failed to create workflow: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    async def add_task_to_workflow(self, workflow_id: str, task_id: str, dependencies: Optional[List[str]] = None) -> APIResponse:
        """Add a task to a workflow"""
        try:
            if workflow_id not in self.workflows:
                return APIResponse(False, error="Workflow not found")
            
            if task_id not in self.tasks:
                return APIResponse(False, error="Task not found")
            
            workflow = self.workflows[workflow_id]
            task = self.tasks[task_id]
            
            workflow.add_task(task, dependencies)
            
            return APIResponse(
                True,
                message="Task added to workflow successfully"
            )
            
        except Exception as e:
            await self.logger.error(f"Failed to add task to workflow: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    async def execute_workflow(self, workflow_id: str, input_data: Any = None, context: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Execute a workflow"""
        try:
            if workflow_id not in self.workflows:
                return APIResponse(False, error="Workflow not found")
            
            result = await self.workflow_engine.execute_workflow(workflow_id, input_data, context)
            
            return APIResponse(
                True,
                data=result.to_dict(),
                message="Workflow executed successfully"
            )
            
        except Exception as e:
            await self.logger.error(f"Failed to execute workflow {workflow_id}: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    # Memory Management Endpoints
    
    async def store_memory(self, key: str, value: Any, memory_type: str = "working", expires_at: Optional[str] = None) -> APIResponse:
        """Store a value in memory"""
        try:
            from ..core.memory import MemoryType
            
            # Parse memory type
            memory_type_enum = MemoryType(memory_type)
            
            # Parse expiration
            expires_at_dt = None
            if expires_at:
                expires_at_dt = datetime.fromisoformat(expires_at)
            
            entry_id = await self.memory_manager.store(
                key, value, memory_type_enum, expires_at_dt
            )
            
            return APIResponse(
                True,
                data={"entry_id": entry_id},
                message="Value stored in memory successfully"
            )
            
        except Exception as e:
            await self.logger.error(f"Failed to store memory: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    async def retrieve_memory(self, key: str) -> APIResponse:
        """Retrieve a value from memory"""
        try:
            entry = await self.memory_manager.retrieve(key)
            
            if entry is None:
                return APIResponse(False, error="Key not found in memory")
            
            return APIResponse(True, data=entry.to_dict())
            
        except Exception as e:
            await self.logger.error(f"Failed to retrieve memory: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    # Monitoring Endpoints
    
    async def get_metrics(self) -> APIResponse:
        """Get all metrics"""
        try:
            metrics_data = self.metrics_collector.export_metrics()
            return APIResponse(True, data=metrics_data)
            
        except Exception as e:
            await self.logger.error(f"Failed to get metrics: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    async def get_metric(self, metric_name: str) -> APIResponse:
        """Get a specific metric"""
        try:
            metric = self.metrics_collector.get_metric(metric_name)
            
            if metric is None:
                return APIResponse(False, error="Metric not found")
            
            return APIResponse(True, data=metric.to_dict())
            
        except Exception as e:
            await self.logger.error(f"Failed to get metric {metric_name}: {e}", exception=e)
            return APIResponse(False, error=str(e))
    
    async def get_system_status(self) -> APIResponse:
        """Get overall system status"""
        try:
            status = {
                "agents": {
                    "total": len(self.agents),
                    "running": len([a for a in self.agents.values() if a.state.value == "running"]),
                    "completed": len([a for a in self.agents.values() if a.state.value == "completed"]),
                    "failed": len([a for a in self.agents.values() if a.state.value == "failed"]),
                },
                "tasks": {
                    "total": len(self.tasks),
                },
                "workflows": {
                    "total": len(self.workflows),
                },
                "scheduler": self.scheduler.get_status(),
                "memory": {
                    "total_keys": len(await self.memory_manager.list_keys()),
                },
                "uptime": (datetime.utcnow() - datetime.utcnow()).total_seconds(),  # Placeholder
            }
            
            return APIResponse(True, data=status)
            
        except Exception as e:
            await self.logger.error(f"Failed to get system status: {e}", exception=e)
            return APIResponse(False, error=str(e))


# Web server integration (FastAPI example)
def create_fastapi_app(api: FrameworkAPI):
    """Create FastAPI application with framework endpoints"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        from typing import Optional, Any
        
        app = FastAPI(
            title="AI Agent Framework API",
            description="RESTful API for the AI Agent Framework",
            version="0.1.0"
        )
        
        # Request models
        class AgentCreateRequest(BaseModel):
            name: str
            description: Optional[str] = None
            max_retries: int = 3
            timeout: float = 300.0
            memory_enabled: bool = True
            guardrails_enabled: bool = True
            metrics_enabled: bool = True
        
        class TaskCreateRequest(BaseModel):
            name: str
            description: Optional[str] = None
            max_retries: int = 3
            timeout: float = 60.0
            dependencies: List[str] = []
        
        class WorkflowCreateRequest(BaseModel):
            name: str
            description: Optional[str] = None
            max_parallel_tasks: int = 10
            failure_strategy: str = "fail_fast"
            timeout: float = 3600.0
        
        class ExecuteRequest(BaseModel):
            input_data: Any
            context: Optional[Dict[str, Any]] = None
        
        class MemoryStoreRequest(BaseModel):
            key: str
            value: Any
            memory_type: str = "working"
            expires_at: Optional[str] = None
        
        # Agent endpoints
        @app.post("/agents")
        async def create_agent(request: AgentCreateRequest):
            response = await api.create_agent(request.dict())
            if not response.success:
                raise HTTPException(status_code=400, detail=response.error)
            return response.to_dict()
        
        @app.get("/agents/{agent_id}")
        async def get_agent(agent_id: str):
            response = await api.get_agent(agent_id)
            if not response.success:
                raise HTTPException(status_code=404, detail=response.error)
            return response.to_dict()
        
        @app.get("/agents")
        async def list_agents():
            response = await api.list_agents()
            return response.to_dict()
        
        @app.post("/agents/{agent_id}/execute")
        async def execute_agent(agent_id: str, request: ExecuteRequest):
            response = await api.execute_agent(agent_id, request.input_data, request.context)
            if not response.success:
                raise HTTPException(status_code=400, detail=response.error)
            return response.to_dict()
        
        # Task endpoints
        @app.post("/tasks")
        async def create_task(request: TaskCreateRequest):
            response = await api.create_task(request.dict())
            if not response.success:
                raise HTTPException(status_code=400, detail=response.error)
            return response.to_dict()
        
        @app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            response = await api.get_task(task_id)
            if not response.success:
                raise HTTPException(status_code=404, detail=response.error)
            return response.to_dict()
        
        @app.post("/tasks/{task_id}/execute")
        async def execute_task(task_id: str, request: ExecuteRequest):
            response = await api.execute_task(task_id, request.input_data, request.context)
            if not response.success:
                raise HTTPException(status_code=400, detail=response.error)
            return response.to_dict()
        
        # Workflow endpoints
        @app.post("/workflows")
        async def create_workflow(request: WorkflowCreateRequest):
            response = await api.create_workflow(request.dict())
            if not response.success:
                raise HTTPException(status_code=400, detail=response.error)
            return response.to_dict()
        
        @app.post("/workflows/{workflow_id}/tasks/{task_id}")
        async def add_task_to_workflow(workflow_id: str, task_id: str, dependencies: Optional[List[str]] = None):
            response = await api.add_task_to_workflow(workflow_id, task_id, dependencies)
            if not response.success:
                raise HTTPException(status_code=400, detail=response.error)
            return response.to_dict()
        
        @app.post("/workflows/{workflow_id}/execute")
        async def execute_workflow(workflow_id: str, request: ExecuteRequest):
            response = await api.execute_workflow(workflow_id, request.input_data, request.context)
            if not response.success:
                raise HTTPException(status_code=400, detail=response.error)
            return response.to_dict()
        
        # Memory endpoints
        @app.post("/memory")
        async def store_memory(request: MemoryStoreRequest):
            response = await api.store_memory(
                request.key, request.value, request.memory_type, request.expires_at
            )
            if not response.success:
                raise HTTPException(status_code=400, detail=response.error)
            return response.to_dict()
        
        @app.get("/memory/{key}")
        async def retrieve_memory(key: str):
            response = await api.retrieve_memory(key)
            if not response.success:
                raise HTTPException(status_code=404, detail=response.error)
            return response.to_dict()
        
        # Monitoring endpoints
        @app.get("/metrics")
        async def get_metrics():
            response = await api.get_metrics()
            return response.to_dict()
        
        @app.get("/metrics/{metric_name}")
        async def get_metric(metric_name: str):
            response = await api.get_metric(metric_name)
            if not response.success:
                raise HTTPException(status_code=404, detail=response.error)
            return response.to_dict()
        
        @app.get("/status")
        async def get_system_status():
            response = await api.get_system_status()
            return response.to_dict()
        
        # Health check
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        api.app = app
        return app
        
    except ImportError:
        raise ImportError("FastAPI and pydantic required for web API")