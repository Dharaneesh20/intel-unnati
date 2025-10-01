#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent Framework - Quick Setup and Test

This script provides a quick way to test the framework without external dependencies.
It runs core functionality tests to validate the framework is working properly.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuickTest:
    """Quick framework test without external dependencies"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        
    def test_import(self, module_name, description):
        """Test importing a module"""
        try:
            __import__(module_name)
            logger.info(f"[PASS] {description}")
            self.tests_passed += 1
            return True
        except ImportError as e:
            logger.warning(f"! {description} - {e}")
            self.tests_failed += 1
            return False
        except Exception as e:
            logger.error(f"[FAIL] {description} - Unexpected error: {e}")
            self.tests_failed += 1
            return False
    
    async def test_core_functionality(self):
        """Test core framework functionality"""
        logger.info("=== Testing Core Framework ===")
        
        # Test core imports
        core_modules = [
            ("ai_agent_framework.core.agent", "Core Agent System"),
            ("ai_agent_framework.core.task", "Core Task System"),
            ("ai_agent_framework.core.workflow", "Core Workflow System"),
            ("ai_agent_framework.core.memory", "Core Memory System"),
        ]
        
        for module, desc in core_modules:
            self.test_import(module, desc)
        
        # Test orchestration
        orchestration_modules = [
            ("ai_agent_framework.orchestration.dag_engine", "DAG Engine"),
            ("ai_agent_framework.orchestration.scheduler", "Scheduler"),
        ]
        
        for module, desc in orchestration_modules:
            self.test_import(module, desc)
        
        # Test monitoring
        monitoring_modules = [
            ("ai_agent_framework.monitoring.metrics", "Metrics System"),
            ("ai_agent_framework.monitoring.logging", "Logging System"),
        ]
        
        for module, desc in monitoring_modules:
            self.test_import(module, desc)
        
        # Test SDK
        self.test_import("ai_agent_framework.sdk.api", "SDK API")
        
    async def test_optional_components(self):
        """Test optional components"""
        logger.info("=== Testing Optional Components ===")
        
        # Intel optimizations
        intel_modules = [
            ("ai_agent_framework.intel_optimizations.openvino_optimizer", "OpenVINO Optimizer"),
            ("ai_agent_framework.intel_optimizations.pytorch_optimizer", "PyTorch Optimizer"),
            ("ai_agent_framework.intel_optimizations.neural_compressor", "Neural Compressor"),
            ("ai_agent_framework.intel_optimizations.devcloud_integration", "DevCloud Integration"),
        ]
        
        for module, desc in intel_modules:
            self.test_import(module, desc)
        
        # Apache integrations
        apache_modules = [
            ("ai_agent_framework.apache_integration.kafka_integration", "Kafka Integration"),
            ("ai_agent_framework.apache_integration.airflow_integration", "Airflow Integration"),
            ("ai_agent_framework.apache_integration.camel_integration", "Camel Integration"),
        ]
        
        for module, desc in apache_modules:
            self.test_import(module, desc)
        
        # Reference agents
        reference_modules = [
            ("ai_agent_framework.reference_agents.document_processor", "Document Processing Agent"),
            ("ai_agent_framework.reference_agents.data_analyzer", "Data Analysis Agent"),
        ]
        
        for module, desc in reference_modules:
            self.test_import(module, desc)
        
        # Benchmarks
        self.test_import("ai_agent_framework.benchmarks.benchmark_suite", "Benchmarking Suite")
        
    async def test_basic_functionality(self):
        """Test basic functionality if core modules loaded"""
        logger.info("=== Testing Basic Functionality ===")
        
        try:
            # Test agent creation
            from ai_agent_framework.core.agent import SimpleAgent, AgentConfig
            from ai_agent_framework.core.memory import InMemoryStorage
            
            # Create memory
            memory = InMemoryStorage()
            await memory.initialize()
            
            # Create agent
            agent = SimpleAgent("Test Agent")
            await agent.initialize()
            
            logger.info("[PASS] Basic agent creation successful")
            self.tests_passed += 1
            
            # Cleanup
            await agent.cleanup()
            await memory.close()
            
        except Exception as e:
            logger.error(f"[FAIL] Basic functionality test failed: {e}")
            self.tests_failed += 1
        
        try:
            # Test task execution
            from ai_agent_framework.core.task import FunctionTask
            
            task = FunctionTask("test_task", lambda x, context: x * 2)
            result = await task.run(21)
            
            if result.status.value == "completed" and result.result == 42:
                logger.info("[PASS] Basic task execution successful")
                self.tests_passed += 1
            else:
                logger.error(f"[FAIL] Task execution failed: {result}")
                self.tests_failed += 1
                
        except Exception as e:
            logger.error(f"[FAIL] Task execution test failed: {e}")
            self.tests_failed += 1
    
    def print_summary(self):
        """Print test summary"""
        total_tests = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        logger.info("=== Test Summary ===")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {self.tests_passed}")
        logger.info(f"Failed: {self.tests_failed}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if self.tests_failed == 0:
            logger.info("[PASS] All tests passed! Framework is ready to use.")
        elif success_rate >= 70:
            logger.info("[PASS] Framework core is functional. Some optional components need dependencies.")
        else:
            logger.warning("! Framework has significant issues. Check error messages above.")
        
        return success_rate >= 50


async def main():
    """Main test function"""
    logger.info("AI Agent Framework - Quick Setup and Test")
    logger.info("=" * 50)
    
    test = QuickTest()
    
    try:
        # Test core functionality
        await test.test_core_functionality()
        
        # Test optional components
        await test.test_optional_components()
        
        # Test basic functionality
        await test.test_basic_functionality()
        
        # Print summary
        success = test.print_summary()
        
        if success:
            logger.info("")
            logger.info("Framework Setup Complete!")
            logger.info("")
            logger.info("Next Steps:")
            logger.info("1. Install dependencies: pip install -r requirements.txt")
            logger.info("2. Run full demo: python example_complete_framework.py")
            logger.info("3. Run benchmarks: python example_complete_framework.py --run-benchmarks")
            logger.info("4. Check design documentation: DESIGN.md")
            return 0
        else:
            logger.error("Framework setup has issues. Please check error messages.")
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)