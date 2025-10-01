#!/usr/bin/env python3
"""
AI Agent Framework - Easy Install Script

This script helps you install the framework dependencies step by step,
handling problematic packages gracefully.
"""

import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pip_install(packages, description="packages"):
    """Install packages with pip, handling errors gracefully"""
    try:
        logger.info(f"Installing {description}...")
        if isinstance(packages, str):
            packages = [packages]
        
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully installed {description}")
            return True
        else:
            logger.warning(f"⚠️ Failed to install {description}: {result.stderr.strip()}")
            return False
    except Exception as e:
        logger.error(f"❌ Error installing {description}: {e}")
        return False

def install_core_dependencies():
    """Install core framework dependencies"""
    logger.info("=== Installing Core Dependencies ===")
    
    # Core packages that should work everywhere
    core_packages = [
        "pydantic>=2.0.0",
        "fastapi>=0.100.0", 
        "uvicorn>=0.23.0",
        "httpx>=0.24.0",
        "aiofiles>=23.0.0",
        "click>=8.1.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "jinja2>=3.1.0"
    ]
    
    success_count = 0
    for package in core_packages:
        if run_pip_install(package, f"core package {package.split('>=')[0]}"):
            success_count += 1
    
    logger.info(f"Core packages: {success_count}/{len(core_packages)} installed successfully")
    return success_count > len(core_packages) * 0.8  # 80% success rate

def install_ml_dependencies():
    """Install ML/AI dependencies"""
    logger.info("=== Installing ML/AI Dependencies ===")
    
    ml_packages = [
        "numpy>=1.24.0",
        "pandas>=2.0.0", 
        "scikit-learn>=1.3.0",
        "pillow>=10.0.0"
    ]
    
    success_count = 0
    for package in ml_packages:
        if run_pip_install(package, f"ML package {package.split('>=')[0]}"):
            success_count += 1
    
    # Try PyTorch (might be large)
    logger.info("Installing PyTorch (this may take a while)...")
    if run_pip_install("torch>=2.0.0", "PyTorch"):
        success_count += 1
        
        # Try transformers if torch succeeded
        if run_pip_install("transformers>=4.30.0", "Transformers"):
            success_count += 1
    
    # Try OpenCV
    if run_pip_install("opencv-python>=4.8.0", "OpenCV"):
        success_count += 1
    
    logger.info(f"ML packages: {success_count} installed successfully")
    return success_count > 4  # At least basic packages

def install_monitoring_dependencies():
    """Install monitoring dependencies"""
    logger.info("=== Installing Monitoring Dependencies ===")
    
    monitoring_packages = [
        "prometheus-client>=0.17.0",
        "psutil>=5.9.0"
    ]
    
    success_count = 0
    for package in monitoring_packages:
        if run_pip_install(package, f"monitoring package {package.split('>=')[0]}"):
            success_count += 1
    
    # Try structlog
    if run_pip_install("structlog>=23.1.0", "Structured logging"):
        success_count += 1
    
    logger.info(f"Monitoring packages: {success_count} installed successfully")
    return success_count > 0

def install_database_dependencies():
    """Install database dependencies"""
    logger.info("=== Installing Database Dependencies ===")
    
    db_packages = [
        "sqlalchemy>=2.0.0",
        "aiosqlite>=0.19.0"
    ]
    
    success_count = 0
    for package in db_packages:
        if run_pip_install(package, f"database package {package.split('>=')[0]}"):
            success_count += 1
    
    # Try Redis (optional)
    if run_pip_install("redis>=4.5.0", "Redis client"):
        success_count += 1
    
    if run_pip_install("aioredis>=2.0.0", "Async Redis client"):
        success_count += 1
    
    logger.info(f"Database packages: {success_count} installed successfully")
    return success_count > 1

def install_optional_dependencies():
    """Install optional dependencies"""
    logger.info("=== Installing Optional Dependencies ===")
    
    # Test framework
    run_pip_install("pytest>=7.4.0", "Testing framework")
    run_pip_install("pytest-asyncio>=0.21.0", "Async testing")
    
    # Security
    run_pip_install("cryptography>=41.0.0", "Cryptography")
    
    # Apache Kafka (might fail, that's OK)
    run_pip_install("kafka-python>=2.0.2", "Kafka client")

def print_intel_instructions():
    """Print instructions for Intel optimizations"""
    logger.info("=== Intel Optimization Instructions ===")
    logger.info("")
    logger.info("For Intel optimizations, install separately:")
    logger.info("")
    logger.info("1. Intel Extension for PyTorch:")
    logger.info("   pip install intel-extension-for-pytorch==1.13.0+cpu -f https://developer.intel.com/ipex-whl-stable-cpu")
    logger.info("")
    logger.info("2. OpenVINO:")
    logger.info("   pip install openvino==2023.1.0")
    logger.info("")
    logger.info("3. Neural Compressor:")
    logger.info("   pip install neural-compressor==2.3.0")
    logger.info("")

def main():
    """Main installation function"""
    logger.info("🚀 AI Agent Framework - Easy Installation")
    logger.info("=" * 50)
    
    # Install core dependencies
    core_success = install_core_dependencies()
    
    # Install ML dependencies  
    ml_success = install_ml_dependencies()
    
    # Install monitoring
    monitoring_success = install_monitoring_dependencies()
    
    # Install database
    db_success = install_database_dependencies()
    
    # Install optional
    install_optional_dependencies()
    
    # Print results
    logger.info("=" * 50)
    logger.info("📊 Installation Summary:")
    logger.info(f"✅ Core Framework: {'SUCCESS' if core_success else 'PARTIAL'}")
    logger.info(f"✅ ML/AI Libraries: {'SUCCESS' if ml_success else 'PARTIAL'}")
    logger.info(f"✅ Monitoring: {'SUCCESS' if monitoring_success else 'PARTIAL'}")
    logger.info(f"✅ Database: {'SUCCESS' if db_success else 'PARTIAL'}")
    
    if core_success:
        logger.info("")
        logger.info("🎉 Framework is ready to use!")
        logger.info("Run: python run_framework.py")
    else:
        logger.info("")
        logger.info("⚠️ Some core dependencies failed. Framework may have limited functionality.")
    
    # Print Intel instructions
    print_intel_instructions()
    
    logger.info("")
    logger.info("🎯 Next Steps:")
    logger.info("1. Test the framework: python run_framework.py")
    logger.info("2. Run quick validation: python quick_test.py")
    logger.info("3. Check HOW_TO_RUN.md for detailed usage")

if __name__ == "__main__":
    main()