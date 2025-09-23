"""
Unit tests for the core agent functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.agents import Agent
from core.workflow import Workflow, Task


class TestAgent:
    """Test cases for Agent class."""
    
    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing."""
        tool = Mock()
        tool.execute = AsyncMock(return_value={"result": "success"})
        return tool
    
    @pytest.fixture
    def simple_workflow(self, mock_tool):
        """Create a simple workflow for testing."""
        workflow = Workflow(name="test_workflow")
        task = Task(
            name="test_task",
            tool=mock_tool,
            inputs={"input": "test"},
            outputs=["result"]
        )
        workflow.add_task(task)
        return workflow
    
    @pytest.fixture
    def agent(self, simple_workflow):
        """Create an agent for testing."""
        return Agent(
            name="test_agent",
            workflow=simple_workflow,
            description="Test agent"
        )
    
    def test_agent_creation(self, agent):
        """Test agent creation."""
        assert agent.name == "test_agent"
        assert agent.description == "Test agent"
        assert agent.workflow is not None
    
    @pytest.mark.asyncio
    async def test_agent_execution(self, agent):
        """Test agent execution."""
        result = await agent.execute({"input": "test_data"})
        
        assert result.status == "completed"
        assert result.agent_id == agent.id
    
    def test_agent_validation(self):
        """Test agent validation."""
        with pytest.raises(Exception):
            # Should raise error when no workflow or state machine
            Agent(name="invalid_agent")
    
    @pytest.mark.asyncio
    async def test_agent_retry_mechanism(self, simple_workflow):
        """Test agent retry mechanism."""
        # Create a tool that fails first time, succeeds second time
        failing_tool = Mock()
        call_count = 0
        
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return {"result": "success"}
        
        failing_tool.execute = mock_execute
        
        # Replace tool in workflow
        simple_workflow.tasks[0].tool = failing_tool
        
        agent = Agent(
            name="retry_agent",
            workflow=simple_workflow,
            max_retries=2
        )
        
        result = await agent.execute({"input": "test"})
        assert result.status == "completed"
        assert call_count == 2  # Failed once, succeeded on retry
