#!/usr/bin/env python3
"""
Intel AI Agent Framework - Python Initialization Script
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.9+"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} available")


def create_directories():
    """Create necessary directories"""
    dirs = [
        "logs", "data", "models", "config", 
        "deployment/grafana", "deployment/prometheus"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories created")


def install_dependencies():
    """Install Python dependencies"""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        
        if Path("requirements.txt").exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
        
        print("✓ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)


def create_sample_config():
    """Create sample configuration file"""
    config = {
        "framework": {
            "name": "intel-ai-agent-framework",
            "version": "1.0.0",
            "environment": "development"
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "agent_framework"
        },
        "redis": {
            "host": "localhost", 
            "port": 6379
        },
        "observability": {
            "metrics_enabled": True,
            "log_level": "INFO"
        }
    }
    
    config_path = Path("config/framework.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Configuration created")


def validate_installation():
    """Validate the installation"""
    try:
        # Test core imports
        import fastapi
        import uvicorn
        import pydantic
        
        print("✓ Core dependencies validated")
        return True
    except ImportError as e:
        print(f"❌ Import validation failed: {e}")
        return False


def main():
    """Main initialization function"""
    print("🚀 Initializing Intel AI Agent Framework...")
    
    check_python_version()
    create_directories()
    install_dependencies()
    create_sample_config()
    
    if validate_installation():
        print("\n🎉 Initialization complete!")
        print("\nNext steps:")
        print("1. Start infrastructure: docker-compose up -d")
        print("2. Run API server: uvicorn src.api.main:app --reload")
        print("3. Try examples in examples/ directory")
    else:
        print("\n❌ Initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
