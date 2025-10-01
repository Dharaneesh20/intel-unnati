"""
Core Agent implementation
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio
import structlog

logger = structlog.get_logger(__name__)


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentConfig(BaseModel):
    """Agent configuration"""
    name: str
    description: Optional[str] = None
    max_retries: int = 3
    timeout: float = 300.0  # 5 minutes
    memory_enabled: bool = True
    guardrails_enabled: bool = True
    metrics_enabled: bool = True


class AgentContext(BaseModel):
    """Agent execution context"""
    agent_id: str
    session_id: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Agent(ABC):
    """
    Base Agent class for the AI Agent Framework.
    
    All agents must inherit from this class and implement the execute method.
    Provides lifecycle management, state tracking, and integration with
    the framework's monitoring and orchestration systems.
    """
    
    def __init__(self, config: AgentConfig):
        self.id = str(uuid4())
        self.config = config
        self.state = AgentState.IDLE
        self.context: Optional[AgentContext] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error: Optional[Exception] = None
        self.result: Any = None
        self._logger = logger.bind(agent_id=self.id, agent_name=config.name)
        
    async def run(self, context: AgentContext, input_data: Any) -> Any:
        """
        Main execution method with lifecycle management
        """
        self.context = context
        self.state = AgentState.RUNNING
        self.start_time = datetime.utcnow()
        
        self._logger.info("Agent starting execution", 
                         context=context.dict(), 
                         input_type=type(input_data).__name__)
        
        try:
            for attempt in range(self.config.max_retries + 1):
                try:
                    # Execute with timeout
                    self.result = await asyncio.wait_for(
                        self.execute(input_data),
                        timeout=self.config.timeout
                    )
                    
                    self.state = AgentState.COMPLETED
                    self.end_time = datetime.utcnow()
                    
                    self._logger.info("Agent completed successfully",
                                    execution_time=(self.end_time - self.start_time).total_seconds(),
                                    result_type=type(self.result).__name__)
                    
                    return self.result
                    
                except asyncio.TimeoutError:
                    self.error = TimeoutError(f"Agent execution timed out after {self.config.timeout}s")
                    if attempt < self.config.max_retries:
                        self._logger.warning(f"Agent timeout, retrying (attempt {attempt + 1})")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        break
                        
                except Exception as e:
                    self.error = e
                    if attempt < self.config.max_retries:
                        self._logger.warning(f"Agent execution failed, retrying (attempt {attempt + 1})",
                                           error=str(e))
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        break
                        
        except Exception as e:
            self.error = e
            
        # Execution failed
        self.state = AgentState.FAILED
        self.end_time = datetime.utcnow()
        
        self._logger.error("Agent execution failed",
                          error=str(self.error),
                          execution_time=(self.end_time - self.start_time).total_seconds())
        
        raise self.error
    
    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        """
        Core execution logic - must be implemented by subclasses
        """
        pass
    
    async def pause(self) -> None:
        """Pause agent execution"""
        if self.state == AgentState.RUNNING:
            self.state = AgentState.PAUSED
            self._logger.info("Agent paused")
    
    async def resume(self) -> None:
        """Resume agent execution"""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.RUNNING
            self._logger.info("Agent resumed")
    
    async def cancel(self) -> None:
        """Cancel agent execution"""
        if self.state in [AgentState.RUNNING, AgentState.PAUSED]:
            self.state = AgentState.CANCELLED
            self._logger.info("Agent cancelled")
    
    async def initialize(self) -> None:
        """Initialize the agent"""
        self._logger.info("Agent initialized")
    
    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        self._logger.info("Agent cleaned up")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.id,
            "name": self.config.name,
            "state": self.state.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time": (
                (self.end_time - self.start_time).total_seconds() 
                if self.start_time and self.end_time else None
            ),
            "error": str(self.error) if self.error else None,
            "has_result": self.result is not None,
        }


class SimpleAgent(Agent):
    """
    Simple agent implementation for testing and examples
    """
    
    def __init__(self, name: str, execution_func=None):
        config = AgentConfig(name=name)
        super().__init__(config)
        self.execution_func = execution_func
    
    async def execute(self, input_data: Any) -> Any:
        """Execute the provided function or return input as-is"""
        if self.execution_func:
            if asyncio.iscoroutinefunction(self.execution_func):
                return await self.execution_func(input_data)
            else:
                return self.execution_func(input_data)
        return input_data