"""
State machine implementation for complex workflow control.
"""

import uuid
import asyncio
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .exceptions import WorkflowError, ValidationError, ExecutionError


class StateType(Enum):
    """Types of states in the state machine."""
    INITIAL = "initial"
    NORMAL = "normal"
    FINAL = "final"
    ERROR = "error"


class TransitionType(Enum):
    """Types of transitions."""
    AUTOMATIC = "automatic"
    CONDITIONAL = "conditional"
    MANUAL = "manual"


@dataclass
class StateResult:
    """Result of state execution."""
    state_id: str
    status: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class State:
    """Represents a state in the state machine."""
    
    def __init__(
        self,
        name: str,
        state_type: StateType = StateType.NORMAL,
        action: Optional[Callable] = None,
        timeout: Optional[int] = None,
        retries: int = 3
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.state_type = state_type
        self.action = action
        self.timeout = timeout
        self.retries = retries
        self.entry_actions: List[Callable] = []
        self.exit_actions: List[Callable] = []
        
    def add_entry_action(self, action: Callable) -> None:
        """Add an action to execute when entering this state."""
        self.entry_actions.append(action)
        
    def add_exit_action(self, action: Callable) -> None:
        """Add an action to execute when exiting this state."""
        self.exit_actions.append(action)
        
    async def enter(self, context: Dict[str, Any]) -> None:
        """Execute entry actions."""
        for action in self.entry_actions:
            await self._execute_action(action, context)
            
    async def execute(self, context: Dict[str, Any]) -> StateResult:
        """Execute the state's main action."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Execute entry actions
            await self.enter(context)
            
            outputs = {}
            if self.action:
                if asyncio.iscoroutinefunction(self.action):
                    outputs = await self.action(context)
                else:
                    outputs = self.action(context)
            
            # Execute exit actions
            await self.exit(context)
            
            end_time = datetime.now(timezone.utc)
            
            return StateResult(
                state_id=self.id,
                status="completed",
                outputs=outputs or {},
                execution_time=(end_time - start_time).total_seconds(),
                started_at=start_time,
                completed_at=end_time
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            return StateResult(
                state_id=self.id,
                status="failed",
                error=str(e),
                execution_time=(end_time - start_time).total_seconds(),
                started_at=start_time,
                completed_at=end_time
            )
    
    async def exit(self, context: Dict[str, Any]) -> None:
        """Execute exit actions."""
        for action in self.exit_actions:
            await self._execute_action(action, context)
    
    async def _execute_action(self, action: Callable, context: Dict[str, Any]) -> Any:
        """Execute a single action."""
        if asyncio.iscoroutinefunction(action):
            return await action(context)
        else:
            return action(context)


class Transition:
    """Represents a transition between states."""
    
    def __init__(
        self,
        from_state: State,
        to_state: State,
        condition: Optional[Callable] = None,
        transition_type: TransitionType = TransitionType.AUTOMATIC,
        action: Optional[Callable] = None,
        priority: int = 0
    ):
        self.id = str(uuid.uuid4())
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition
        self.transition_type = transition_type
        self.action = action
        self.priority = priority
        
    def can_execute(self, context: Dict[str, Any]) -> bool:
        """Check if the transition can be executed."""
        if self.condition:
            try:
                if asyncio.iscoroutinefunction(self.condition):
                    # For async conditions, we need to handle this differently
                    # For now, assume sync conditions
                    return False
                else:
                    return bool(self.condition(context))
            except Exception:
                return False
        return True
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the transition action."""
        if self.action:
            if asyncio.iscoroutinefunction(self.action):
                result = await self.action(context)
            else:
                result = self.action(context)
            return result or {}
        return {}


class StateMachine:
    """State machine for complex workflow control."""
    
    def __init__(self, name: str = "StateMachine"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.states: Dict[str, State] = {}
        self.transitions: List[Transition] = []
        self.initial_state: Optional[State] = None
        self.current_state: Optional[State] = None
        self.final_states: Set[State] = set()
        self.error_states: Set[State] = set()
        
    def add_state(self, state: State) -> None:
        """Add a state to the state machine."""
        if state.name in self.states:
            raise WorkflowError(f"State '{state.name}' already exists")
        
        self.states[state.name] = state
        
        if state.state_type == StateType.INITIAL:
            if self.initial_state:
                raise WorkflowError("State machine can only have one initial state")
            self.initial_state = state
        elif state.state_type == StateType.FINAL:
            self.final_states.add(state)
        elif state.state_type == StateType.ERROR:
            self.error_states.add(state)
    
    def remove_state(self, state_name: str) -> None:
        """Remove a state from the state machine."""
        if state_name not in self.states:
            raise WorkflowError(f"State '{state_name}' not found")
        
        state = self.states[state_name]
        
        # Remove all transitions involving this state
        self.transitions = [
            t for t in self.transitions
            if t.from_state != state and t.to_state != state
        ]
        
        # Update special state sets
        self.final_states.discard(state)
        self.error_states.discard(state)
        
        if self.initial_state == state:
            self.initial_state = None
        
        del self.states[state_name]
    
    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the state machine."""
        # Validate that states exist
        if transition.from_state.name not in self.states:
            raise WorkflowError(f"From state '{transition.from_state.name}' not found")
        if transition.to_state.name not in self.states:
            raise WorkflowError(f"To state '{transition.to_state.name}' not found")
        
        self.transitions.append(transition)
        
        # Sort transitions by priority (higher priority first)
        self.transitions.sort(key=lambda t: t.priority, reverse=True)
    
    def get_transitions_from_state(self, state: State) -> List[Transition]:
        """Get all transitions from a given state."""
        return [t for t in self.transitions if t.from_state == state]
    
    def validate(self) -> None:
        """Validate the state machine structure."""
        if not self.states:
            raise ValidationError("State machine must contain at least one state")
        
        if not self.initial_state:
            raise ValidationError("State machine must have an initial state")
        
        if not self.final_states:
            raise ValidationError("State machine must have at least one final state")
        
        # Check that all states are reachable
        reachable = self._get_reachable_states()
        unreachable = set(self.states.values()) - reachable
        if unreachable:
            unreachable_names = [s.name for s in unreachable]
            raise ValidationError(f"Unreachable states: {', '.join(unreachable_names)}")
    
    def _get_reachable_states(self) -> Set[State]:
        """Get all states reachable from the initial state."""
        if not self.initial_state:
            return set()
        
        visited = set()
        stack = [self.initial_state]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            
            # Add all states reachable from current state
            for transition in self.get_transitions_from_state(current):
                if transition.to_state not in visited:
                    stack.append(transition.to_state)
        
        return visited
    
    async def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the state machine."""
        if not context:
            context = {}
        
        # Validate before execution
        self.validate()
        
        # Initialize execution
        self.current_state = self.initial_state
        context['state_machine_id'] = self.id
        context['execution_path'] = []
        
        start_time = datetime.now(timezone.utc)
        
        try:
            while self.current_state:
                # Record execution path
                context['execution_path'].append({
                    'state': self.current_state.name,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                # Execute current state
                state_result = await self.current_state.execute(context)
                
                # Update context with state outputs
                context[f'state_{self.current_state.name}'] = state_result.outputs
                
                # Check if we've reached a final state
                if self.current_state in self.final_states:
                    break
                
                # Check if we've reached an error state
                if self.current_state in self.error_states:
                    raise ExecutionError(f"Reached error state: {self.current_state.name}")
                
                # Find next state based on transitions
                next_state = await self._find_next_state(context)
                if not next_state:
                    raise ExecutionError(f"No valid transition from state: {self.current_state.name}")
                
                self.current_state = next_state
            
            end_time = datetime.now(timezone.utc)
            
            context['execution_result'] = {
                'status': 'completed',
                'execution_time': (end_time - start_time).total_seconds(),
                'final_state': self.current_state.name if self.current_state else None
            }
            
            return context
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            
            context['execution_result'] = {
                'status': 'failed',
                'error': str(e),
                'execution_time': (end_time - start_time).total_seconds(),
                'final_state': self.current_state.name if self.current_state else None
            }
            
            raise ExecutionError(f"State machine execution failed: {str(e)}")
    
    async def _find_next_state(self, context: Dict[str, Any]) -> Optional[State]:
        """Find the next state to transition to."""
        available_transitions = self.get_transitions_from_state(self.current_state)
        
        for transition in available_transitions:
            if transition.can_execute(context):
                # Execute transition action
                await transition.execute(context)
                return transition.to_state
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state machine to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'states': {
                name: {
                    'id': state.id,
                    'name': state.name,
                    'type': state.state_type.value,
                    'timeout': state.timeout,
                    'retries': state.retries
                }
                for name, state in self.states.items()
            },
            'transitions': [
                {
                    'id': t.id,
                    'from_state': t.from_state.name,
                    'to_state': t.to_state.name,
                    'type': t.transition_type.value,
                    'priority': t.priority
                }
                for t in self.transitions
            ],
            'initial_state': self.initial_state.name if self.initial_state else None,
            'final_states': [s.name for s in self.final_states],
            'error_states': [s.name for s in self.error_states]
        }
