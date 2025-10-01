#!/usr/bin/env python3
"""
Local Intel Validation Script
============================

Validate Intel optimization capabilities locally before DevCloud deployment.
Tests what Intel components are available and working without requiring DevCloud.
"""

import asyncio
import logging
import time
import platform
import sys
import importlib
from typing import Dict, Any, List
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelValidator:
    """Validate Intel optimization capabilities locally"""
    
    def __init__(self):
        self.results = {}
        self.recommendations = []
        
    async def validate_all(self) -> Dict[str, Any]:
        """Run all Intel validation checks"""
        logger.info("🔬 Intel Optimization Validation (Local)")
        logger.info("=" * 60)
        
        await self._check_hardware_compatibility()
        await self._check_pytorch_intel()
        await self._check_openvino_compatibility()
        await self._check_neural_compressor()
        await self._check_mkl_availability()
        await self._validate_performance_baseline()
        await self._check_memory_optimization()
        
        return await self._generate_intel_report()
    
    async def _check_hardware_compatibility(self):
        """Check if hardware supports Intel optimizations"""
        logger.info("--- Hardware Compatibility ---")
        
        try:
            processor = platform.processor().lower()
            system = platform.system()
            
            # Check for Intel CPU
            is_intel = 'intel' in processor
            
            if is_intel:
                logger.info(f"✅ Intel processor detected: {processor}")
                self.results['intel_cpu'] = 'PASS'
            else:
                logger.warning(f"⚠️ Non-Intel processor: {processor}")
                logger.warning("Intel optimizations may have limited benefit")
                self.results['intel_cpu'] = 'WARNING'
                self.recommendations.append("Consider testing on Intel hardware for full optimization benefits")
            
            # Check architecture
            arch = platform.architecture()[0]
            if arch == '64bit':
                logger.info(f"✅ 64-bit architecture: {arch}")
                self.results['architecture'] = 'PASS'
            else:
                logger.warning(f"⚠️ Architecture: {arch}")
                self.results['architecture'] = 'WARNING'
            
            logger.info(f"✅ Operating System: {system}")
            
        except Exception as e:
            logger.error(f"❌ Hardware check failed: {e}")
            self.results['hardware'] = 'FAIL'
    
    async def _check_pytorch_intel(self):
        """Check PyTorch Intel Extension availability"""
        logger.info("--- PyTorch Intel Extension ---")
        
        try:
            # Check regular PyTorch first
            import torch
            logger.info(f"✅ PyTorch version: {torch.__version__}")
            
            # Try to import Intel Extension
            try:
                import intel_extension_for_pytorch as ipex
                logger.info(f"✅ Intel Extension for PyTorch: {ipex.__version__}")
                self.results['pytorch_intel'] = 'PASS'
                
                # Test basic functionality
                x = torch.randn(100, 100)
                y = torch.mm(x, x.t())
                logger.info("✅ Intel PyTorch operations working")
                
            except ImportError:
                logger.warning("⚠️ Intel Extension for PyTorch not installed")
                logger.info("Install with: pip install intel-extension-for-pytorch")
                self.results['pytorch_intel'] = 'WARNING'
                self.recommendations.append("Install Intel Extension for PyTorch for better performance")
            
            # Check MKL
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mkl'):
                mkl_available = torch.backends.mkl.is_available()
                if mkl_available:
                    logger.info("✅ MKL backend available")
                else:
                    logger.warning("⚠️ MKL backend not available")
            
        except ImportError:
            logger.error("❌ PyTorch not installed")
            self.results['pytorch_intel'] = 'FAIL'
        except Exception as e:
            logger.error(f"❌ PyTorch Intel check failed: {e}")
            self.results['pytorch_intel'] = 'FAIL'
    
    async def _check_openvino_compatibility(self):
        """Check OpenVINO compatibility"""
        logger.info("--- OpenVINO Compatibility ---")
        
        try:
            # Try to import OpenVINO
            try:
                import openvino as ov
                logger.info(f"✅ OpenVINO available: {ov.__version__}")
                
                # Test basic OpenVINO functionality
                core = ov.Core()
                available_devices = core.available_devices
                logger.info(f"✅ Available devices: {available_devices}")
                
                self.results['openvino'] = 'PASS'
                
            except ImportError:
                logger.warning("⚠️ OpenVINO not installed")
                logger.info("Install with: pip install openvino")
                self.results['openvino'] = 'WARNING'
                self.recommendations.append("Install OpenVINO for model optimization")
            
        except Exception as e:
            logger.error(f"❌ OpenVINO check failed: {e}")
            self.results['openvino'] = 'FAIL'
    
    async def _check_neural_compressor(self):
        """Check Neural Compressor availability"""
        logger.info("--- Neural Compressor ---")
        
        try:
            try:
                import neural_compressor
                logger.info(f"✅ Neural Compressor available: {neural_compressor.__version__}")
                self.results['neural_compressor'] = 'PASS'
                
            except ImportError:
                logger.warning("⚠️ Neural Compressor not installed")
                logger.info("Install with: pip install neural-compressor")
                self.results['neural_compressor'] = 'WARNING'
                self.recommendations.append("Install Neural Compressor for model quantization")
            
        except Exception as e:
            logger.error(f"❌ Neural Compressor check failed: {e}")
            self.results['neural_compressor'] = 'FAIL'
    
    async def _check_mkl_availability(self):
        """Check Intel MKL availability"""
        logger.info("--- Intel MKL ---")
        
        try:
            # Check numpy MKL
            import numpy as np
            
            # Check if numpy is compiled with MKL
            config = np.show_config()
            if 'mkl' in str(config).lower():
                logger.info("✅ NumPy compiled with MKL")
                self.results['mkl_numpy'] = 'PASS'
            else:
                logger.warning("⚠️ NumPy not compiled with MKL")
                self.results['mkl_numpy'] = 'WARNING'
            
            # Test basic linear algebra performance
            start_time = time.time()
            a = np.random.random((1000, 1000))
            b = np.random.random((1000, 1000))
            c = np.dot(a, b)
            duration = time.time() - start_time
            
            logger.info(f"✅ Matrix multiplication (1000x1000): {duration:.3f}s")
            
            if duration < 1.0:
                logger.info("✅ Good linear algebra performance")
            else:
                logger.warning("⚠️ Slow linear algebra performance")
            
        except Exception as e:
            logger.error(f"❌ MKL check failed: {e}")
            self.results['mkl'] = 'FAIL'
    
    async def _validate_performance_baseline(self):
        """Establish performance baseline for Intel optimizations"""
        logger.info("--- Performance Baseline ---")
        
        try:
            import numpy as np
            import time
            
            # CPU-intensive operations
            tests = [
                ("Matrix Multiplication", self._test_matrix_mult),
                ("FFT Transform", self._test_fft),
                ("Linear Algebra", self._test_linalg)
            ]
            
            performance_scores = {}
            
            for test_name, test_func in tests:
                try:
                    duration = await test_func()
                    performance_scores[test_name] = duration
                    logger.info(f"✅ {test_name}: {duration:.3f}s")
                except Exception as e:
                    logger.warning(f"⚠️ {test_name} failed: {e}")
            
            # Overall performance assessment
            avg_time = sum(performance_scores.values()) / len(performance_scores)
            if avg_time < 0.5:
                logger.info("✅ Excellent baseline performance")
                self.results['performance_baseline'] = 'PASS'
            elif avg_time < 2.0:
                logger.info("✅ Good baseline performance")
                self.results['performance_baseline'] = 'PASS'
            else:
                logger.warning("⚠️ Slow baseline performance")
                self.results['performance_baseline'] = 'WARNING'
            
        except Exception as e:
            logger.error(f"❌ Performance baseline failed: {e}")
            self.results['performance_baseline'] = 'FAIL'
    
    async def _test_matrix_mult(self) -> float:
        """Test matrix multiplication performance"""
        import numpy as np
        start_time = time.time()
        
        a = np.random.random((500, 500)).astype(np.float32)
        b = np.random.random((500, 500)).astype(np.float32)
        c = np.dot(a, b)
        
        return time.time() - start_time
    
    async def _test_fft(self) -> float:
        """Test FFT performance"""
        import numpy as np
        start_time = time.time()
        
        x = np.random.random(2**16).astype(np.complex64)
        y = np.fft.fft(x)
        z = np.fft.ifft(y)
        
        return time.time() - start_time
    
    async def _test_linalg(self) -> float:
        """Test linear algebra performance"""
        import numpy as np
        start_time = time.time()
        
        a = np.random.random((300, 300)).astype(np.float32)
        eigenvals = np.linalg.eigvals(a)
        
        return time.time() - start_time
    
    async def _check_memory_optimization(self):
        """Check memory optimization capabilities"""
        logger.info("--- Memory Optimization ---")
        
        try:
            import psutil
            import gc
            
            # Memory usage test
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create large arrays
            import numpy as np
            arrays = []
            for i in range(10):
                arr = np.random.random((1000, 1000)).astype(np.float32)
                arrays.append(arr)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = peak_memory - initial_memory
            
            # Cleanup
            del arrays
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = peak_memory - final_memory
            
            logger.info(f"✅ Memory usage: {memory_used:.1f}MB peak")
            logger.info(f"✅ Memory freed: {memory_freed:.1f}MB")
            
            if memory_used < 500:  # Less than 500MB for test
                self.results['memory_optimization'] = 'PASS'
            else:
                self.results['memory_optimization'] = 'WARNING'
            
        except Exception as e:
            logger.error(f"❌ Memory optimization check failed: {e}")
            self.results['memory_optimization'] = 'FAIL'
    
    async def _generate_intel_report(self) -> Dict[str, Any]:
        """Generate Intel validation report"""
        passed = sum(1 for v in self.results.values() if v == 'PASS')
        warned = sum(1 for v in self.results.values() if v == 'WARNING')
        failed = sum(1 for v in self.results.values() if v == 'FAIL')
        total = len(self.results)
        
        # Overall Intel readiness
        if failed == 0 and warned <= 2:
            intel_ready = "✅ READY FOR INTEL OPTIMIZATION"
        elif failed <= 1:
            intel_ready = "⚠️ PARTIALLY READY (some optimizations available)"
        else:
            intel_ready = "❌ LIMITED INTEL SUPPORT"
        
        logger.info("=" * 60)
        logger.info("📊 INTEL VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Tests: {total} total, {passed} passed, {warned} warnings, {failed} failed")
        logger.info(f"Intel Readiness: {intel_ready}")
        logger.info("")
        
        # Detailed results
        for component, status in self.results.items():
            status_icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}[status]
            logger.info(f"{status_icon} {component.replace('_', ' ').title()}: {status}")
        
        logger.info("")
        
        # Recommendations
        if self.recommendations:
            logger.info("🔧 RECOMMENDATIONS:")
            for rec in self.recommendations:
                logger.info(f"  • {rec}")
            logger.info("")
        
        # Next steps
        if intel_ready.startswith("✅"):
            logger.info("🚀 Your system supports Intel optimizations!")
            logger.info("DevCloud deployment will provide:")
            logger.info("  • Enhanced CPU performance with Intel MKL")
            logger.info("  • Model optimization with OpenVINO")
            logger.info("  • Quantization with Neural Compressor")
            logger.info("  • Multi-node scaling capabilities")
        else:
            logger.info("💡 To improve Intel optimization support:")
            logger.info("  • Install missing Intel packages")
            logger.info("  • Test on Intel hardware if available")
            logger.info("  • Deploy to DevCloud for full Intel optimization")
        
        return {
            'intel_readiness': intel_ready,
            'results': self.results,
            'recommendations': self.recommendations,
            'summary': {
                'total': total,
                'passed': passed,
                'warnings': warned,
                'failed': failed
            }
        }

async def main():
    """Main Intel validation function"""
    validator = IntelValidator()
    report = await validator.validate_all()
    
    # Save report
    with open('intel_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("📋 Intel validation report saved to: intel_validation_report.json")
    return report

if __name__ == "__main__":
    asyncio.run(main())