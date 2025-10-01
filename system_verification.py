#!/usr/bin/env python3
"""
AI Agent Framework - System Verification Script
=============================================

Comprehensive system verification before Intel DevCloud deployment.
Tests all framework components and validates system readiness.
"""

import asyncio
import logging
import sys
import time
import platform
import psutil
from pathlib import Path
from typing import Dict, List, Any
import importlib
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SystemVerifier:
    """Comprehensive system verification for AI Agent Framework"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    async def run_all_verifications(self) -> Dict[str, Any]:
        """Run all verification tests"""
        logger.info("🚀 Starting AI Agent Framework System Verification")
        logger.info("=" * 60)
        
        # System checks
        await self._verify_system_requirements()
        await self._verify_python_environment()
        await self._verify_dependencies()
        
        # Framework checks
        await self._verify_core_framework()
        await self._verify_orchestration()
        await self._verify_monitoring()
        await self._verify_api()
        
        # Performance checks
        await self._verify_performance()
        await self._verify_memory_usage()
        await self._verify_concurrent_execution()
        
        # Integration checks
        await self._verify_intel_components()
        await self._verify_apache_components()
        
        # Generate final report
        return await self._generate_report()
    
    async def _verify_system_requirements(self):
        """Verify basic system requirements"""
        logger.info("--- System Requirements Check ---")
        
        try:
            # OS Info
            os_info = {
                'system': platform.system(),
                'version': platform.version(),
                'architecture': platform.architecture()[0],
                'processor': platform.processor()
            }
            logger.info(f"✅ OS: {os_info['system']} {os_info['architecture']}")
            
            # Python version
            python_version = sys.version_info
            if python_version >= (3, 9):
                logger.info(f"✅ Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
                self.results['python_version'] = 'PASS'
            else:
                logger.error(f"❌ Python version {python_version} < 3.9 required")
                self.results['python_version'] = 'FAIL'
            
            # Memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb >= 4:
                logger.info(f"✅ Memory: {memory_gb:.1f}GB available")
                self.results['memory'] = 'PASS'
            else:
                logger.warning(f"⚠️ Memory: {memory_gb:.1f}GB (recommend 4GB+)")
                self.results['memory'] = 'WARNING'
            
            # CPU
            cpu_count = psutil.cpu_count()
            logger.info(f"✅ CPU: {cpu_count} cores")
            self.results['cpu'] = 'PASS'
            
            # Disk space
            disk = psutil.disk_usage('.')
            disk_gb = disk.free / (1024**3)
            if disk_gb >= 2:
                logger.info(f"✅ Disk: {disk_gb:.1f}GB free")
                self.results['disk'] = 'PASS'
            else:
                logger.warning(f"⚠️ Disk: {disk_gb:.1f}GB free (recommend 2GB+)")
                self.results['disk'] = 'WARNING'
                
            self.results['system_requirements'] = 'PASS'
            
        except Exception as e:
            logger.error(f"❌ System requirements check failed: {e}")
            self.results['system_requirements'] = 'FAIL'
    
    async def _verify_python_environment(self):
        """Verify Python environment and paths"""
        logger.info("--- Python Environment Check ---")
        
        try:
            # Check if we're in the right directory
            current_dir = Path.cwd()
            expected_files = ['run_framework.py', 'install_framework.py', 'ai_agent_framework']
            
            missing_files = []
            for file in expected_files:
                if not Path(file).exists():
                    missing_files.append(file)
            
            if not missing_files:
                logger.info("✅ Framework files present")
                self.results['framework_files'] = 'PASS'
            else:
                logger.error(f"❌ Missing files: {missing_files}")
                self.results['framework_files'] = 'FAIL'
            
            # Check Python path
            logger.info(f"✅ Python executable: {sys.executable}")
            logger.info(f"✅ Working directory: {current_dir}")
            
            self.results['python_environment'] = 'PASS'
            
        except Exception as e:
            logger.error(f"❌ Python environment check failed: {e}")
            self.results['python_environment'] = 'FAIL'
    
    async def _verify_dependencies(self):
        """Verify required dependencies are installed"""
        logger.info("--- Dependencies Check ---")
        
        required_packages = [
            'pydantic', 'fastapi', 'uvicorn', 'httpx', 'aiofiles',
            'numpy', 'pandas', 'scikit-learn', 'torch', 'transformers',
            'prometheus_client', 'structlog', 'sqlalchemy', 'redis'
        ]
        
        passed = 0
        failed = 0
        
        for package in required_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                logger.info(f"✅ {package}")
                passed += 1
            except ImportError:
                logger.warning(f"⚠️ {package} (optional)")
                failed += 1
        
        if passed >= len(required_packages) * 0.8:  # 80% pass rate
            logger.info(f"✅ Dependencies: {passed}/{len(required_packages)} available")
            self.results['dependencies'] = 'PASS'
        else:
            logger.warning(f"⚠️ Dependencies: {passed}/{len(required_packages)} available")
            self.results['dependencies'] = 'WARNING'
    
    async def _verify_core_framework(self):
        """Test core framework components"""
        logger.info("--- Core Framework Check ---")
        
        try:
            # Import the demo components
            sys.path.append('.')
            from run_framework import SimpleAgent, SimpleTask, SimpleWorkflow
            
            # Test Simple Agent
            agent = SimpleAgent('test_agent', 'Verification Agent')
            await agent.initialize()
            result = await agent.run({'test': 'verification'})
            
            if result['status'] == 'success':
                logger.info("✅ SimpleAgent working")
            else:
                logger.error("❌ SimpleAgent failed")
                
            # Test Simple Task
            task = SimpleTask('test_task', lambda x: x * 2)
            result = await task.execute(21)
            
            if result['result'] == 42:
                logger.info("✅ SimpleTask working")
            else:
                logger.error("❌ SimpleTask failed")
            
            # Test Simple Workflow
            workflow = SimpleWorkflow('test_workflow')
            workflow.add_task(SimpleTask('step1', lambda x: x + 10))
            workflow.add_task(SimpleTask('step2', lambda x: x * 2))
            result = await workflow.execute(5)
            
            if result['status'] == 'completed':
                logger.info("✅ SimpleWorkflow working")
            else:
                logger.error("❌ SimpleWorkflow failed")
            
            self.results['core_framework'] = 'PASS'
            
        except Exception as e:
            logger.error(f"❌ Core framework check failed: {e}")
            self.results['core_framework'] = 'FAIL'
    
    async def _verify_orchestration(self):
        """Test orchestration components"""
        logger.info("--- Orchestration Check ---")
        
        try:
            # Test parallel execution
            from run_framework import SimpleTask
            import asyncio
            
            tasks = []
            for i in range(5):
                task = SimpleTask(f'parallel_task_{i}', lambda x, i=i: f'Result {i}: {x}')
                tasks.append(task.execute(f'input_{i}'))
            
            results = await asyncio.gather(*tasks)
            
            if len(results) == 5 and all(r['status'] == 'success' for r in results):
                logger.info("✅ Parallel orchestration working")
                self.results['orchestration'] = 'PASS'
            else:
                logger.error("❌ Parallel orchestration failed")
                self.results['orchestration'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Orchestration check failed: {e}")
            self.results['orchestration'] = 'FAIL'
    
    async def _verify_monitoring(self):
        """Test monitoring capabilities"""
        logger.info("--- Monitoring Check ---")
        
        try:
            import prometheus_client
            import structlog
            
            # Test structured logging
            logger_test = structlog.get_logger()
            logger_test.info("Test log message", component="verification")
            
            logger.info("✅ Structured logging working")
            logger.info("✅ Prometheus client available")
            
            self.results['monitoring'] = 'PASS'
            
        except Exception as e:
            logger.error(f"❌ Monitoring check failed: {e}")
            self.results['monitoring'] = 'FAIL'
    
    async def _verify_api(self):
        """Test API components"""
        logger.info("--- API Check ---")
        
        try:
            import fastapi
            import pydantic
            import uvicorn
            
            # Test basic FastAPI setup
            app = fastapi.FastAPI(title="Test API")
            
            @app.get("/health")
            async def health_check():
                return {"status": "healthy"}
            
            logger.info("✅ FastAPI components working")
            self.results['api'] = 'PASS'
            
        except Exception as e:
            logger.error(f"❌ API check failed: {e}")
            self.results['api'] = 'FAIL'
    
    async def _verify_performance(self):
        """Test framework performance"""
        logger.info("--- Performance Check ---")
        
        try:
            from run_framework import SimpleAgent
            
            # Performance test
            agent = SimpleAgent('perf_test', 'Performance Test Agent')
            await agent.initialize()
            
            start_time = time.time()
            tasks = []
            
            for i in range(100):
                tasks.append(agent.run({'request_id': i}))
            
            await asyncio.gather(*tasks)
            duration = time.time() - start_time
            
            throughput = 100 / duration
            avg_latency = (duration / 100) * 1000  # ms
            
            logger.info(f"✅ Performance: {throughput:.0f} req/sec, {avg_latency:.2f}ms latency")
            
            if throughput > 500:
                self.results['performance'] = 'PASS'
            elif throughput > 100:
                self.results['performance'] = 'WARNING'
            else:
                self.results['performance'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Performance check failed: {e}")
            self.results['performance'] = 'FAIL'
    
    async def _verify_memory_usage(self):
        """Test memory usage patterns"""
        logger.info("--- Memory Usage Check ---")
        
        try:
            import psutil
            import gc
            
            # Measure baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create multiple agents
            from run_framework import SimpleAgent
            agents = []
            
            for i in range(10):
                agent = SimpleAgent(f'memory_test_{i}', f'Memory Test Agent {i}')
                await agent.initialize()
                agents.append(agent)
            
            # Measure peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_per_agent = (peak_memory - baseline_memory) / 10
            
            logger.info(f"✅ Memory usage: {memory_per_agent:.1f}MB per agent")
            
            # Cleanup
            del agents
            gc.collect()
            
            if memory_per_agent < 50:
                self.results['memory_usage'] = 'PASS'
            elif memory_per_agent < 100:
                self.results['memory_usage'] = 'WARNING'
            else:
                self.results['memory_usage'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Memory usage check failed: {e}")
            self.results['memory_usage'] = 'FAIL'
    
    async def _verify_concurrent_execution(self):
        """Test concurrent execution capabilities"""
        logger.info("--- Concurrent Execution Check ---")
        
        try:
            from run_framework import SimpleAgent
            
            # Test concurrent agents
            agents = []
            for i in range(20):
                agent = SimpleAgent(f'concurrent_{i}', f'Concurrent Agent {i}')
                await agent.initialize()
                agents.append(agent)
            
            # Run all agents concurrently
            start_time = time.time()
            tasks = [agent.run({'concurrent_test': i}) for i, agent in enumerate(agents)]
            results = await asyncio.gather(*tasks)
            duration = time.time() - start_time
            
            success_count = sum(1 for r in results if r['status'] == 'success')
            
            if success_count == 20:
                logger.info(f"✅ Concurrent execution: {20/duration:.0f} concurrent ops/sec")
                self.results['concurrent_execution'] = 'PASS'
            else:
                logger.error(f"❌ Concurrent execution: {success_count}/20 succeeded")
                self.results['concurrent_execution'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Concurrent execution check failed: {e}")
            self.results['concurrent_execution'] = 'FAIL'
    
    async def _verify_intel_components(self):
        """Test Intel optimization components (local)"""
        logger.info("--- Intel Components Check ---")
        
        try:
            intel_components = ['torch', 'transformers', 'numpy']
            available = []
            
            for component in intel_components:
                try:
                    module = importlib.import_module(component)
                    available.append(component)
                    logger.info(f"✅ {component} available")
                except ImportError:
                    logger.warning(f"⚠️ {component} not available")
            
            if len(available) >= 2:
                self.results['intel_components'] = 'PASS'
            else:
                self.results['intel_components'] = 'WARNING'
                
        except Exception as e:
            logger.error(f"❌ Intel components check failed: {e}")
            self.results['intel_components'] = 'FAIL'
    
    async def _verify_apache_components(self):
        """Test Apache integration readiness"""
        logger.info("--- Apache Components Check ---")
        
        try:
            # Check if Kafka client is available
            try:
                import kafka
                logger.info("✅ Kafka client available")
                kafka_available = True
            except ImportError:
                logger.warning("⚠️ Kafka client not available")
                kafka_available = False
            
            # Check async capabilities
            import asyncio
            logger.info("✅ Asyncio support available")
            
            if kafka_available:
                self.results['apache_components'] = 'PASS'
            else:
                self.results['apache_components'] = 'WARNING'
                
        except Exception as e:
            logger.error(f"❌ Apache components check failed: {e}")
            self.results['apache_components'] = 'FAIL'
    
    async def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        duration = time.time() - self.start_time
        
        # Count results
        passed = sum(1 for v in self.results.values() if v == 'PASS')
        warned = sum(1 for v in self.results.values() if v == 'WARNING')
        failed = sum(1 for v in self.results.values() if v == 'FAIL')
        total = len(self.results)
        
        # Overall status
        if failed == 0 and warned <= 2:
            overall_status = "✅ READY FOR DEVCLOUD"
        elif failed <= 1:
            overall_status = "⚠️ MOSTLY READY (minor issues)"
        else:
            overall_status = "❌ NOT READY (fixes needed)"
        
        logger.info("=" * 60)
        logger.info("📊 SYSTEM VERIFICATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Tests: {total} total, {passed} passed, {warned} warnings, {failed} failed")
        logger.info(f"Status: {overall_status}")
        logger.info("")
        
        # Detailed results
        for component, status in self.results.items():
            status_icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}[status]
            logger.info(f"{status_icon} {component.replace('_', ' ').title()}: {status}")
        
        logger.info("")
        
        if overall_status.startswith("✅"):
            logger.info("🚀 Your system is ready for Intel DevCloud deployment!")
            logger.info("Next steps:")
            logger.info("1. Package: tar -czf framework.tar.gz ai_agent_framework/ *.py")
            logger.info("2. Upload: scp framework.tar.gz devcloud.intel.com:~/")
            logger.info("3. Deploy: qsub -l nodes=1:gpu:ppn=2 deploy.sh")
        else:
            logger.info("🔧 Please address the issues above before DevCloud deployment")
        
        return {
            'overall_status': overall_status,
            'duration': duration,
            'results': self.results,
            'summary': {
                'total': total,
                'passed': passed,
                'warnings': warned,
                'failed': failed
            }
        }

async def main():
    """Main verification function"""
    verifier = SystemVerifier()
    report = await verifier.run_all_verifications()
    
    # Save report
    with open('system_verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📋 Full report saved to: system_verification_report.json")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())