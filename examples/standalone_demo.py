"""
Standalone example that demonstrates the framework without external dependencies.
"""

import asyncio
import time
from typing import Dict, Any


class MockTool:
    """Mock tool for demonstration purposes."""
    
    def __init__(self, name: str, processing_time: float = 0.1):
        self.name = name
        self.processing_time = processing_time
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with mock processing."""
        print(f"  🔧 {self.name} processing: {inputs}")
        await asyncio.sleep(self.processing_time)  # Simulate processing time
        
        # Mock processing logic
        result = {
            "tool_name": self.name,
            "processed_inputs": inputs,
            "timestamp": time.time(),
            "status": "completed"
        }
        
        print(f"  ✅ {self.name} completed: {result}")
        return result


class SimpleAgent:
    """Simplified agent for demonstration."""
    
    def __init__(self, name: str, tools: list):
        self.name = name
        self.tools = tools
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all tools in sequence."""
        print(f"🤖 Starting agent: {self.name}")
        
        results = {}
        current_inputs = inputs
        
        for tool in self.tools:
            tool_result = await tool.execute(current_inputs)
            results[tool.name] = tool_result
            # Pass outputs to next tool
            current_inputs.update(tool_result)
        
        print(f"✅ Agent {self.name} completed successfully!")
        return {
            "agent_name": self.name,
            "final_results": results,
            "status": "completed"
        }


async def main():
    """Run the demonstration."""
    print("🚀 Intel AI Agent Framework - Simple Demo")
    print("=" * 50)
    
    # Create tools
    tools = [
        MockTool("text_analyzer", 0.2),
        MockTool("sentiment_processor", 0.3),
        MockTool("summary_generator", 0.1)
    ]
    
    # Create agent
    agent = SimpleAgent("demo_agent", tools)
    
    # Execute agent
    inputs = {
        "text": "This is a sample text for the Intel AI Agent Framework demonstration!",
        "user_id": "demo_user",
        "task_type": "text_processing"
    }
    
    result = await agent.execute(inputs)
    
    print("\n📊 Final Results:")
    print("=" * 50)
    for key, value in result.items():
        print(f"{key}: {value}")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
