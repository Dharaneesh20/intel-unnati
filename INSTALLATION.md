# Installation Guide - Intel AI Agent Framework

## 🚀 Quick Start Installation

### **Basic Installation (Recommended)**

Install the core framework without heavy dependencies:

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Test installation
python examples/standalone_demo.py
```

## 📦 **Modular Installation Options**

### **Option 1: Core Framework Only**
```bash
pip install -r requirements.txt
```
**Includes**: FastAPI, Pydantic, Redis, Basic ML tools, PyTorch

### **Option 2: Core + Intel Optimizations**
```bash
pip install -r requirements.txt
pip install -r requirements-intel.txt
```
**Adds**: OpenVINO, Intel PyTorch Extensions

### **Option 3: Core + Apache Airflow**
```bash
pip install -r requirements.txt
pip install -r requirements-apache.txt
```
**Adds**: Apache Airflow, Kafka, Airflow Providers

### **Option 4: Full Installation**
```bash
pip install -r requirements.txt
pip install -r requirements-intel.txt
pip install -r requirements-apache.txt
pip install -r requirements-dev.txt
```

## ⚠️ **Important Notes**

### **Dependency Conflicts**

1. **SQLAlchemy Version**: 
   - Core framework uses SQLAlchemy 1.4.x for Apache Airflow compatibility
   - If you don't need Airflow, you can upgrade to SQLAlchemy 2.x separately

2. **OpenTelemetry Version**:
   - Apache Airflow requires specific OpenTelemetry versions
   - Install observability tools separately if needed

3. **Intel Dependencies**:
   - Intel optimizations require specific hardware
   - Skip if you don't have Intel CPUs/GPUs

### **Installation Order**

**Recommended order** to avoid conflicts:
```bash
# 1. Core first
pip install -r requirements.txt

# 2. Choose ONE of the following:
pip install -r requirements-intel.txt    # For Intel optimization
# OR
pip install -r requirements-apache.txt   # For Apache Airflow

# 3. Development tools (optional)
pip install -r requirements-dev.txt
```

## 🐳 **Docker Installation**

For a clean environment without dependency conflicts:

```bash
# Build Docker image
docker build -t intel-ai-framework .

# Run container
docker run -p 8000:8000 intel-ai-framework
```

## 🔧 **Development Setup**

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install development tools
pip install -r requirements-dev.txt

# 3. Install pre-commit hooks
pre-commit install

# 4. Run tests
pytest tests/
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Dependency Conflict Error**:
   ```bash
   # Clear pip cache and reinstall
   pip cache purge
   pip install --no-cache-dir -r requirements.txt
   ```

2. **Intel Dependencies Fail**:
   ```bash
   # Skip Intel optimizations
   pip install -r requirements.txt
   # Skip requirements-intel.txt
   ```

3. **Apache Airflow Issues**:
   ```bash
   # Install Airflow in separate environment
   python -m venv airflow-env
   source airflow-env/bin/activate
   pip install apache-airflow==2.7.3
   ```

### **Version Compatibility**

| Component | Compatible Versions | Notes |
|-----------|-------------------|-------|
| Python | 3.9, 3.10, 3.11 | Tested versions |
| SQLAlchemy | 1.4.36 - 1.4.x | Airflow compatible |
| FastAPI | 0.104.0+ | Core framework |
| PyTorch | 2.1.0+ | ML functionality |
| Apache Airflow | 2.7.0 - 2.7.x | Optional component |

## 📋 **Verification**

After installation, verify your setup:

```bash
# 1. Test core framework
python -c "from src.core.agents import Agent; print('✅ Core framework working')"

# 2. Test examples
python examples/standalone_demo.py

# 3. Run basic tests
pytest tests/unit/ -v

# 4. Check installed packages
pip list | grep -E "(fastapi|torch|pydantic)"
```

## 🔄 **Updating Dependencies**

To update to newer versions:

```bash
# 1. Backup current environment
pip freeze > current_requirements.txt

# 2. Update core dependencies
pip install --upgrade -r requirements.txt

# 3. Test compatibility
pytest tests/

# 4. If issues, rollback
pip uninstall -r requirements.txt
pip install -r current_requirements.txt
```

---

**Choose the installation option that best fits your use case!** 🎯
