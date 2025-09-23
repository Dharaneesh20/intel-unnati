"""
Simple example of creating and running an AI agent.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.workflow import Workflow, Task
from src.core.agents import Agent


# Mock tool for demonstration
class SimpleTextTool:
    """Simple tool that processes text."""
    
    async def execute(self, inputs: dict) -> dict:
        text = inputs.get("text", "")
        return {
            "processed_text": f"Processed: {text}",
            "word_count": len(text.split())
        }


async def main():
    """Run a simple agent example."""
    
    # Create a workflow
    workflow = Workflow(
        name="simple_text_processing",
        description="A simple text processing workflow"
    )
    
    # Create a task
    process_task = Task(
        name="process_text",
        tool=SimpleTextTool(),
        inputs={"text": "{{user_input}}"},
        outputs=["processed_text", "word_count"]
    )
    
    # Add task to workflow
    workflow.add_task(process_task)
    
    # Create agent
    agent = Agent(
        name="simple_agent",
        workflow=workflow,
        description="A simple text processing agent"
    )
    
    # Execute agent
    print("🤖 Starting simple agent...")
    result = await agent.execute({
        "user_input": "Hello from Intel AI Agent Framework!"
    })
    
    print(f"✅ Agent execution completed!")
    print(f"Status: {result.status}")
    print(f"Outputs: {result.outputs}")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
