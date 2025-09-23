"""
Agent implementation for executing workflows and state machines.
"""

import uuid
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field

from .workflow import Workflow, WorkflowResult, WorkflowStatus
from .state_machine import StateMachine
from .config import get_config
from .exceptions import ExecutionError, ValidationError


@dataclass
class AgentResult:
    """Result of agent execution."""
    agent_id: str
    status: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    workflow_result: Optional[WorkflowResult] = None
    state_machine_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class Agent:
    """AI Agent that can execute workflows or state machines."""
    
    def __init__(
        self,
        name: str,
        workflow: Optional[Workflow] = None,
        state_machine: Optional[StateMachine] = None,
        description: str = "",
        max_retries: int = 3,
        timeout: Optional[int] = None
    ):
        if not workflow and not state_machine:
            raise ValidationError("Agent must have either a workflow or state machine")
        
        if workflow and state_machine:
            raise ValidationError("Agent cannot have both workflow and state machine")
        
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.workflow = workflow
        self.state_machine = state_machine
        self.max_retries = max_retries
        self.timeout = timeout
        self.created_at = datetime.now(timezone.utc)
        self.execution_history: List[AgentResult] = []
        
        # Configuration
        self.config = get_config()
        
    @property
    def execution_type(self) -> str:
        """Get the type of execution (workflow or state_machine)."""
        return "workflow" if self.workflow else "state_machine"
    
    async def execute(
        self,
        inputs: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> AgentResult:
        """Execute the agent's workflow or state machine."""
        start_time = datetime.now(timezone.utc)
        
        # Prepare execution context
        execution_context = self._prepare_context(inputs, context)
        
        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.workflow:
                    result = await self._execute_workflow(execution_context)
                else:
                    result = await self._execute_state_machine(execution_context)
                
                # Record successful execution
                self.execution_history.append(result)
                return result
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # Wait before retry with exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed
                    break
        
        # All attempts failed
        end_time = datetime.now(timezone.utc)
        error_result = AgentResult(
            agent_id=self.id,
            status="failed",
            error=f"Execution failed after {self.max_retries + 1} attempts: {str(last_error)}",
            execution_time=(end_time - start_time).total_seconds(),
            started_at=start_time,
            completed_at=end_time
        )
        
        self.execution_history.append(error_result)
        return error_result
    
    def _prepare_context(
        self,
        inputs: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Prepare execution context."""
        execution_context = {
            'agent_id': self.id,
            'agent_name': self.name,
            'execution_type': self.execution_type,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if inputs:
            execution_context.update(inputs)
        
        if context:
            execution_context.update(context)
        
        return execution_context
    
    async def _execute_workflow(self, context: Dict[str, Any]) -> AgentResult:
        """Execute workflow."""
        start_time = datetime.now(timezone.utc)
        
        try:
            workflow_result = await self.workflow.execute(context)
            end_time = datetime.now(timezone.utc)
            
            status = "completed" if workflow_result.status == WorkflowStatus.COMPLETED else "failed"
            
            return AgentResult(
                agent_id=self.id,
                status=status,
                outputs=workflow_result.outputs,
                workflow_result=workflow_result,
                execution_time=(end_time - start_time).total_seconds(),
                started_at=start_time,
                completed_at=end_time
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            
            return AgentResult(
                agent_id=self.id,
                status="failed",
                error=str(e),
                execution_time=(end_time - start_time).total_seconds(),
                started_at=start_time,
                completed_at=end_time
            )
    
    async def _execute_state_machine(self, context: Dict[str, Any]) -> AgentResult:
        """Execute state machine."""
        start_time = datetime.now(timezone.utc)
        
        try:
            sm_result = await self.state_machine.execute(context)
            end_time = datetime.now(timezone.utc)
            
            execution_result = sm_result.get('execution_result', {})
            status = execution_result.get('status', 'unknown')
            
            return AgentResult(
                agent_id=self.id,
                status=status,
                outputs=sm_result,
                state_machine_result=sm_result,
                execution_time=(end_time - start_time).total_seconds(),
                started_at=start_time,
                completed_at=end_time
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            
            return AgentResult(
                agent_id=self.id,
                status="failed",
                error=str(e),
                execution_time=(end_time - start_time).total_seconds(),
                started_at=start_time,
                completed_at=end_time
            )
    
    def get_execution_history(self) -> List[AgentResult]:
        """Get the agent's execution history."""
        return self.execution_history.copy()
    
    def get_last_execution(self) -> Optional[AgentResult]:
        """Get the result of the last execution."""
        return self.execution_history[-1] if self.execution_history else None
    
    def clear_history(self) -> None:
        """Clear the execution history."""
        self.execution_history.clear()
    
    def validate(self) -> None:
        """Validate the agent configuration."""
        if self.workflow:
            self.workflow.validate()
        elif self.state_machine:
            self.state_machine.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'execution_type': self.execution_type,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'created_at': self.created_at.isoformat(),
            'workflow': self.workflow.to_dict() if self.workflow else None,
            'state_machine': self.state_machine.to_dict() if self.state_machine else None,
            'execution_count': len(self.execution_history)
        }


class AgentManager:
    """Manager for multiple agents."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        
    def register_agent(self, agent: Agent) -> None:
        """Register an agent."""
        if agent.id in self.agents:
            raise ValidationError(f"Agent with ID {agent.id} already registered")
        
        # Validate agent before registration
        agent.validate()
        
        self.agents[agent.id] = agent
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id not in self.agents:
            raise ValidationError(f"Agent with ID {agent_id} not found")
        
        del self.agents[agent_id]
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        for agent in self.agents.values():
            if agent.name == name:
                return agent
        return None
    
    def list_agents(self) -> List[Agent]:
        """List all registered agents."""
        return list(self.agents.values())
    
    async def execute_agent(
        self,
        agent_id: str,
        inputs: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> AgentResult:
        """Execute an agent by ID."""
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValidationError(f"Agent with ID {agent_id} not found")
        
        return await agent.execute(inputs, context)
    
    async def execute_multiple_agents(
        self,
        agent_ids: List[str],
        inputs: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
        parallel: bool = True
    ) -> Dict[str, AgentResult]:
        """Execute multiple agents."""
        if parallel:
            # Execute agents in parallel
            tasks = []
            for agent_id in agent_ids:
                agent = self.get_agent(agent_id)
                if agent:
                    tasks.append(agent.execute(inputs, context))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Map results back to agent IDs
            result_dict = {}
            for i, agent_id in enumerate(agent_ids):
                if i < len(results):
                    if isinstance(results[i], Exception):
                        # Handle execution exception
                        result_dict[agent_id] = AgentResult(
                            agent_id=agent_id,
                            status="failed",
                            error=str(results[i])
                        )
                    else:
                        result_dict[agent_id] = results[i]
            
            return result_dict
        
        else:
            # Execute agents sequentially
            results = {}
            for agent_id in agent_ids:
                try:
                    result = await self.execute_agent(agent_id, inputs, context)
                    results[agent_id] = result
                except Exception as e:
                    results[agent_id] = AgentResult(
                        agent_id=agent_id,
                        status="failed",
                        error=str(e)
                    )
            
            return results
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for all agents."""
        total_executions = 0
        successful_executions = 0
        failed_executions = 0
        
        for agent in self.agents.values():
            history = agent.get_execution_history()
            total_executions += len(history)
            
            for result in history:
                if result.status == "completed":
                    successful_executions += 1
                else:
                    failed_executions += 1
        
        success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
        
        return {
            'total_agents': len(self.agents),
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failed_executions': failed_executions,
            'success_rate': round(success_rate, 2)
        }
