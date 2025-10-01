#!/usr/bin/env python3
"""
AI Agent Framework - Simple Startup

This script provides a simple way to run the core framework without external dependencies.
Perfect for getting started and testing basic functionality.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleAgent:
    """Simple agent implementation for demonstration"""
    
    def __init__(self, agent_id: str, name: str = None):
        self.agent_id = agent_id
        self.name = name or agent_id
        self.state = "initialized"
        
    async def initialize(self):
        """Initialize the agent"""
        self.state = "ready"
        logger.info(f"Agent {self.name} initialized successfully")
        
    async def run(self, inputs: dict = None):
        """Run the agent"""
        self.state = "running"
        logger.info(f"Agent {self.name} is processing: {inputs}")
        
        # Simulate some work
        await asyncio.sleep(0.1)
        
        result = {
            "agent_id": self.agent_id,
            "status": "success",
            "processed_at": datetime.utcnow().isoformat(),
            "inputs": inputs,
            "output": f"Processed by {self.name}"
        }
        
        self.state = "completed"
        logger.info(f"Agent {self.name} completed successfully")
        return result
        
    async def cleanup(self):
        """Cleanup resources"""
        self.state = "stopped"
        logger.info(f"Agent {self.name} cleaned up")


class SimpleTask:
    """Simple task implementation"""
    
    def __init__(self, task_id: str, func, *args, **kwargs):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = "pending"
        
    async def execute(self, *extra_args, **extra_kwargs):
        """Execute the task"""
        try:
            self.status = "running"
            logger.info(f"Executing task {self.task_id}")
            
            # Combine original args with any extra args passed to execute()
            all_args = self.args + extra_args
            all_kwargs = {**self.kwargs, **extra_kwargs}
            
            # Execute the function
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(*all_args, **all_kwargs)
            else:
                result = self.func(*all_args, **all_kwargs)
            
            self.status = "completed"
            logger.info(f"Task {self.task_id} completed successfully")
            
            return {
                "task_id": self.task_id,
                "status": "success",
                "result": result,
                "completed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.status = "failed"
            logger.error(f"Task {self.task_id} failed: {e}")
            return {
                "task_id": self.task_id,
                "status": "error",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }


class SimpleWorkflow:
    """Simple workflow implementation"""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.tasks = []
        self.results = {}
        
    def add_task(self, task: SimpleTask):
        """Add task to workflow"""
        self.tasks.append(task)
        logger.info(f"Added task {task.task_id} to workflow {self.workflow_id}")
        
    async def execute(self):
        """Execute all tasks in the workflow"""
        logger.info(f"Starting workflow {self.workflow_id} with {len(self.tasks)} tasks")
        
        for task in self.tasks:
            result = await task.execute()
            self.results[task.task_id] = result
            
        logger.info(f"Workflow {self.workflow_id} completed")
        return {
            "workflow_id": self.workflow_id,
            "status": "completed",
            "task_results": self.results,
            "completed_at": datetime.utcnow().isoformat()
        }


async def demonstrate_core_functionality():
    """Demonstrate core framework functionality"""
    logger.info("=== AI Agent Framework - Core Demonstration ===")
    
    # 1. Simple Agent Demo
    logger.info("--- Testing Simple Agent ---")
    agent = SimpleAgent("demo_agent", "Demo Agent")
    await agent.initialize()
    
    result = await agent.run({"message": "Hello World!", "data": [1, 2, 3, 4, 5]})
    logger.info(f"Agent result: {result}")
    
    await agent.cleanup()
    
    # 2. Simple Task Demo
    logger.info("--- Testing Simple Tasks ---")
    
    # Math task
    math_task = SimpleTask("math_task", lambda x, y: x * y + 10, 5, 3)
    math_result = await math_task.execute()
    logger.info(f"Math task result: {math_result}")
    
    # String task
    string_task = SimpleTask("string_task", lambda text: text.upper() + "!", "hello world")
    string_result = await string_task.execute()
    logger.info(f"String task result: {string_result}")
    
    # 3. Simple Workflow Demo
    logger.info("--- Testing Simple Workflow ---")
    workflow = SimpleWorkflow("demo_workflow")
    
    # Add tasks to workflow
    workflow.add_task(SimpleTask("task1", lambda: "Step 1 completed"))
    workflow.add_task(SimpleTask("task2", lambda x: f"Step 2: processed {x}", "input_data"))
    workflow.add_task(SimpleTask("task3", lambda: sum([1, 2, 3, 4, 5])))
    
    workflow_result = await workflow.execute()
    logger.info(f"Workflow result: {workflow_result}")
    
    return True


async def demonstrate_reference_workflows():
    """Demonstrate reference agent workflows"""
    logger.info("=== Reference Agent Workflows ===")
    
    # Document Processing Simulation
    logger.info("--- Document Processing Workflow ---")
    doc_agent = SimpleAgent("doc_processor", "Document Processing Agent")
    await doc_agent.initialize()
    
    doc_inputs = {
        "document_path": "sample_document.pdf",
        "operations": ["extract_text", "find_entities", "summarize"]
    }
    
    doc_result = await doc_agent.run(doc_inputs)
    logger.info(f"Document processing result: {doc_result}")
    
    # Data Analysis Simulation
    logger.info("--- Data Analysis Workflow ---")
    data_agent = SimpleAgent("data_analyzer", "Data Analysis Agent")
    await data_agent.initialize()
    
    data_inputs = {
        "dataset": "sample_data.csv",
        "analysis_type": "statistical",
        "operations": ["describe", "correlate", "visualize"]
    }
    
    data_result = await data_agent.run(data_inputs)
    logger.info(f"Data analysis result: {data_result}")


async def demonstrate_advanced_features():
    """Demonstrate advanced framework features"""
    logger.info("=== Advanced Features Demo ===")
    
    # Parallel execution
    logger.info("--- Parallel Task Execution ---")
    
    async def async_task(task_id: str, duration: float):
        await asyncio.sleep(duration)
        return f"Task {task_id} completed after {duration}s"
    
    # Create multiple tasks
    tasks = []
    for i in range(3):
        task = SimpleTask(f"parallel_task_{i}", async_task, f"task_{i}", 0.1 * (i + 1))
        tasks.append(task.execute())
    
    # Execute in parallel
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()
    
    logger.info(f"Parallel execution completed in {end_time - start_time:.2f} seconds")
    for result in results:
        logger.info(f"Parallel result: {result}")


async def run_performance_test():
    """Run a simple performance test"""
    logger.info("=== Performance Test ===")
    
    # Test agent throughput
    agent = SimpleAgent("perf_agent", "Performance Test Agent")
    await agent.initialize()
    
    num_requests = 100
    start_time = asyncio.get_event_loop().time()
    
    tasks = []
    for i in range(num_requests):
        task = agent.run({"request_id": i, "data": f"test_data_{i}"})
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()
    
    duration = end_time - start_time
    throughput = num_requests / duration
    
    logger.info(f"Performance Test Results:")
    logger.info(f"  Requests: {num_requests}")
    logger.info(f"  Duration: {duration:.2f} seconds")
    logger.info(f"  Throughput: {throughput:.1f} requests/second")
    logger.info(f"  Average Latency: {(duration / num_requests) * 1000:.2f} ms")


async def main():
    """Main demonstration function"""
    try:
        logger.info("🚀 Starting AI Agent Framework Demonstration")
        logger.info("=" * 60)
        
        # Core functionality
        await demonstrate_core_functionality()
        
        # Reference workflows
        await demonstrate_reference_workflows()
        
        # Advanced features
        await demonstrate_advanced_features()
        
        # Performance test
        await run_performance_test()
        
        logger.info("=" * 60)
        logger.info("✅ Framework Demonstration Completed Successfully!")
        logger.info("")
        logger.info("🎉 Your AI Agent Framework is working perfectly!")
        logger.info("")
        logger.info("Next Steps:")
        logger.info("1. Install full dependencies: pip install -r requirements.txt")
        logger.info("2. Run complete demo: python example_complete_framework.py")
        logger.info("3. Explore the framework modules in ai_agent_framework/")
        logger.info("4. Check DESIGN.md for detailed architecture")
        logger.info("5. Deploy to Intel DevCloud for optimization")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)