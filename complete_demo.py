#!/usr/bin/env python3
"""
Complete Demo Runner
===================

Run all demo methods to verify the AI Agent Framework is working properly.
This script provides multiple verification levels before DevCloud deployment.
"""

import asyncio
import logging
import time
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteDemoRunner:
    """Run complete framework demonstration"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    async def run_complete_demo(self, demo_level: str = "quick") -> Dict[str, Any]:
        """
        Run complete demonstration based on level:
        - quick: 5 minutes, basic functionality
        - standard: 10 minutes, comprehensive testing  
        - full: 15 minutes, production readiness
        """
        logger.info("🎯 AI Agent Framework - Complete Demo")
        logger.info(f"Demo Level: {demo_level.upper()}")
        logger.info("=" * 60)
        
        if demo_level == "quick":
            await self._run_quick_demo()
        elif demo_level == "standard":
            await self._run_standard_demo()
        elif demo_level == "full":
            await self._run_full_demo()
        else:
            logger.error(f"Unknown demo level: {demo_level}")
            return {}
        
        return await self._generate_final_report()
    
    async def _run_quick_demo(self):
        """Quick 5-minute demo"""
        logger.info("🚀 QUICK DEMO (5 minutes)")
        logger.info("Testing core functionality...")
        
        # 1. Dependencies check
        await self._check_dependencies()
        
        # 2. Run framework demo (what user already ran successfully)
        await self._run_framework_demo()
        
        # 3. Quick health check
        await self._quick_health_check()
    
    async def _run_standard_demo(self):
        """Standard 10-minute demo"""
        logger.info("🔍 STANDARD DEMO (10 minutes)")
        logger.info("Comprehensive system testing...")
        
        # Include quick demo
        await self._run_quick_demo()
        
        # 4. System verification
        await self._run_system_verification()
        
        # 5. API testing (if server available)
        await self._test_api_availability()
    
    async def _run_full_demo(self):
        """Full 15-minute production readiness demo"""
        logger.info("🎯 FULL DEMO (15 minutes)")
        logger.info("Production readiness testing...")
        
        # Include standard demo
        await self._run_standard_demo()
        
        # 6. Intel validation
        await self._run_intel_validation()
        
        # 7. Performance benchmarks
        await self._run_performance_benchmarks()
        
        # 8. Generate deployment package
        await self._prepare_deployment_package()
    
    async def _check_dependencies(self):
        """Check core dependencies"""
        logger.info("--- Dependencies Check ---")
        
        try:
            required = ['numpy', 'torch', 'fastapi', 'pydantic']
            missing = []
            
            for pkg in required:
                try:
                    __import__(pkg.replace('-', '_'))
                    logger.info(f"✅ {pkg}")
                except ImportError:
                    missing.append(pkg)
                    logger.warning(f"⚠️ {pkg} missing")
            
            if not missing:
                self.results['dependencies'] = 'PASS'
                logger.info("✅ All core dependencies available")
            else:
                self.results['dependencies'] = 'WARNING'
                logger.info(f"⚠️ Missing: {missing}")
                logger.info("Run: python install_framework.py")
                
        except Exception as e:
            logger.error(f"❌ Dependencies check failed: {e}")
            self.results['dependencies'] = 'FAIL'
    
    async def _run_framework_demo(self):
        """Run the main framework demo"""
        logger.info("--- Framework Demo ---")
        
        try:
            # Run the framework demo script
            logger.info("Running framework demonstration...")
            
            # Import and run directly (since we know it works)
            sys.path.append('.')
            from run_framework import main as run_framework_main
            
            # Capture the result
            result = await run_framework_main()
            
            if result:
                logger.info("✅ Framework demo completed successfully")
                logger.info(f"✅ Performance: {result.get('throughput', 'N/A')} req/sec")
                self.results['framework_demo'] = 'PASS'
            else:
                logger.error("❌ Framework demo failed")
                self.results['framework_demo'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Framework demo error: {e}")
            self.results['framework_demo'] = 'FAIL'
    
    async def _quick_health_check(self):
        """Quick health check"""
        logger.info("--- Quick Health Check ---")
        
        try:
            # Simple agent test
            from run_framework import SimpleAgent
            
            agent = SimpleAgent('health_check', 'Health Check Agent')
            await agent.initialize()
            result = await agent.run({'health': 'check'})
            
            if result['status'] == 'success':
                logger.info("✅ Agent health check passed")
                self.results['health_check'] = 'PASS'
            else:
                logger.error("❌ Agent health check failed")
                self.results['health_check'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            self.results['health_check'] = 'FAIL'
    
    async def _run_system_verification(self):
        """Run comprehensive system verification"""
        logger.info("--- System Verification ---")
        
        try:
            # Import and run system verification
            from system_verification import SystemVerifier
            
            verifier = SystemVerifier()
            report = await verifier.run_all_verifications()
            
            overall_status = report.get('overall_status', '')
            if 'READY' in overall_status:
                logger.info("✅ System verification passed")
                self.results['system_verification'] = 'PASS'
            elif 'MOSTLY' in overall_status:
                logger.info("⚠️ System verification warnings")
                self.results['system_verification'] = 'WARNING'
            else:
                logger.error("❌ System verification failed")
                self.results['system_verification'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ System verification error: {e}")
            self.results['system_verification'] = 'FAIL'
    
    async def _test_api_availability(self):
        """Test API endpoints if server is available"""
        logger.info("--- API Availability ---")
        
        try:
            import httpx
            
            # Quick check if API server is running
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get("http://localhost:8000/health", timeout=2.0)
                    server_running = response.status_code == 200
                except Exception:
                    server_running = False
            
            if server_running:
                logger.info("✅ API server is running")
                
                # Run API tests
                from test_api_endpoints import test_api_with_server_check
                report = await test_api_with_server_check()
                
                if report.get('summary', {}).get('failed', 0) == 0:
                    self.results['api_tests'] = 'PASS'
                else:
                    self.results['api_tests'] = 'WARNING'
            else:
                logger.info("ℹ️ API server not running (optional)")
                logger.info("To test APIs: cd ai_agent_framework && python -m uvicorn api:app --reload")
                self.results['api_tests'] = 'SKIP'
                
        except Exception as e:
            logger.error(f"❌ API test error: {e}")
            self.results['api_tests'] = 'FAIL'
    
    async def _run_intel_validation(self):
        """Run Intel optimization validation"""
        logger.info("--- Intel Validation ---")
        
        try:
            from validate_intel_local import IntelValidator
            
            validator = IntelValidator()
            report = await validator.validate_all()
            
            intel_readiness = report.get('intel_readiness', '')
            if 'READY' in intel_readiness:
                logger.info("✅ Intel optimization ready")
                self.results['intel_validation'] = 'PASS'
            elif 'PARTIALLY' in intel_readiness:
                logger.info("⚠️ Partial Intel support")
                self.results['intel_validation'] = 'WARNING'  
            else:
                logger.info("⚠️ Limited Intel support")
                self.results['intel_validation'] = 'WARNING'
                
        except Exception as e:
            logger.error(f"❌ Intel validation error: {e}")
            self.results['intel_validation'] = 'FAIL'
    
    async def _run_performance_benchmarks(self):
        """Run performance benchmarks"""
        logger.info("--- Performance Benchmarks ---")
        
        try:
            # Simple performance test
            from run_framework import SimpleAgent
            import time
            
            agent = SimpleAgent('perf_test', 'Performance Test')
            await agent.initialize()
            
            # Measure throughput
            start_time = time.time()
            tasks = []
            
            for i in range(50):  # Smaller test for speed
                tasks.append(agent.run({'request': i}))
            
            await asyncio.gather(*tasks)
            duration = time.time() - start_time
            throughput = 50 / duration
            
            logger.info(f"✅ Performance: {throughput:.0f} req/sec")
            
            if throughput > 200:
                self.results['performance'] = 'PASS'
            elif throughput > 50:
                self.results['performance'] = 'WARNING'
            else:
                self.results['performance'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Performance benchmark error: {e}")
            self.results['performance'] = 'FAIL'
    
    async def _prepare_deployment_package(self):
        """Prepare DevCloud deployment package"""
        logger.info("--- Deployment Package Preparation ---")
        
        try:
            # Check required files exist
            required_files = [
                'ai_agent_framework/',
                'run_framework.py',
                'requirements.txt',
                'README.md'
            ]
            
            missing_files = []
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if not missing_files:
                logger.info("✅ All deployment files present")
                logger.info("Ready to create deployment package:")
                logger.info("  tar -czf ai_agent_framework.tar.gz ai_agent_framework/ *.py requirements.txt")
                self.results['deployment_package'] = 'PASS'
            else:
                logger.error(f"❌ Missing files: {missing_files}")
                self.results['deployment_package'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Deployment preparation error: {e}")
            self.results['deployment_package'] = 'FAIL'
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final comprehensive report"""
        duration = time.time() - self.start_time
        
        passed = sum(1 for v in self.results.values() if v == 'PASS')
        warned = sum(1 for v in self.results.values() if v == 'WARNING')
        failed = sum(1 for v in self.results.values() if v == 'FAIL')
        skipped = sum(1 for v in self.results.values() if v == 'SKIP')
        total = len(self.results)
        
        # Overall readiness
        if failed == 0 and warned <= 2:
            overall_status = "🎉 READY FOR DEVCLOUD DEPLOYMENT"
            readiness_score = "EXCELLENT"
        elif failed <= 1:
            overall_status = "🚀 MOSTLY READY (minor issues)"  
            readiness_score = "GOOD"
        else:
            overall_status = "🔧 NEEDS ATTENTION (fixes required)"
            readiness_score = "NEEDS_WORK"
        
        logger.info("=" * 60)
        logger.info("🎯 COMPLETE DEMO REPORT")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Tests: {total} total, {passed} passed, {warned} warnings, {failed} failed, {skipped} skipped")
        logger.info(f"Overall Status: {overall_status}")
        logger.info(f"DevCloud Readiness: {readiness_score}")
        logger.info("")
        
        # Detailed results
        for component, status in self.results.items():
            status_icons = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌", "SKIP": "ℹ️"}
            icon = status_icons.get(status, "❓")
            logger.info(f"{icon} {component.replace('_', ' ').title()}: {status}")
        
        logger.info("")
        
        # Next steps
        if readiness_score == "EXCELLENT":
            logger.info("🚀 DEVCLOUD DEPLOYMENT READY!")
            logger.info("Next steps:")
            logger.info("1. Create package: tar -czf framework.tar.gz ai_agent_framework/ *.py requirements.txt")
            logger.info("2. Upload to DevCloud: scp framework.tar.gz devcloud.intel.com:~/")
            logger.info("3. Submit job: qsub -l nodes=1:gpu:ppn=2 -N ai_framework deploy.sh")
            logger.info("4. Monitor: qstat -u $USER")
        elif readiness_score == "GOOD":
            logger.info("🔧 Minor issues to address, but ready for basic DevCloud deployment")
        else:
            logger.info("🔧 Please fix the failed components before DevCloud deployment")
        
        report = {
            'overall_status': overall_status,
            'readiness_score': readiness_score,
            'duration': duration,
            'results': self.results,
            'summary': {
                'total': total,
                'passed': passed,
                'warnings': warned,
                'failed': failed,
                'skipped': skipped
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save report
        with open('complete_demo_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("📋 Complete demo report saved to: complete_demo_report.json")
        
        return report

async def main():
    """Main demo runner"""
    
    # Parse command line arguments
    demo_level = "quick"
    if len(sys.argv) > 1:
        demo_level = sys.argv[1].lower()
        
    if demo_level not in ["quick", "standard", "full"]:
        print("Usage: python complete_demo.py [quick|standard|full]")
        print("  quick:    5 minutes, basic functionality (default)")
        print("  standard: 10 minutes, comprehensive testing")
        print("  full:     15 minutes, production readiness")
        return
    
    # Run demo
    runner = CompleteDemoRunner()
    report = await runner.run_complete_demo(demo_level)
    
    return report

if __name__ == "__main__":
    asyncio.run(main())