# GitHub Workflows Configuration Guide

## 🎯 Overview

This guide explains how to configure and use GitHub workflows for the Intel AI Agent Framework project. The workflows provide comprehensive CI/CD, testing, security scanning, and deployment automation.

## 📋 Workflows Included

### 1. **CI/CD Pipeline** (`.github/workflows/ci-cd.yml`)
- **Triggered on**: Push to main/master/develop, Pull Requests
- **Features**:
  - Code quality checks (Black, isort, flake8, mypy)
  - Security scanning (bandit)
  - Multi-version testing (Python 3.9, 3.10, 3.11)
  - Integration tests with Redis and PostgreSQL
  - Docker build and push
  - Performance benchmarking
  - Deployment to staging/production

### 2. **Release Pipeline** (`.github/workflows/release.yml`)
- **Triggered on**: Release published, Manual dispatch
- **Features**:
  - Build release artifacts
  - Create GitHub releases
  - Publish to PyPI
  - Update documentation
  - Team notifications

### 3. **Security Scanning** (`.github/workflows/security.yml`)
- **Triggered on**: Schedule (weekly), Push to main, Pull Requests
- **Features**:
  - Dependency vulnerability scanning
  - Code security analysis
  - Docker image security scanning
  - License compliance checking
  - Secret detection

### 4. **Documentation** (`.github/workflows/documentation.yml`)
- **Triggered on**: Documentation changes, Push to main
- **Features**:
  - Build and deploy documentation
  - Link checking
  - Spell checking
  - Documentation quality metrics

### 5. **Performance Monitoring** (`.github/workflows/performance.yml`)
- **Triggered on**: Schedule (daily), Performance-related changes
- **Features**:
  - Performance benchmarking
  - Load testing
  - Performance comparison for PRs

## 🚀 Getting Started

### **Step 1: Repository Setup**

1. **Enable GitHub Actions**:
   - Go to your repository settings
   - Navigate to "Actions" → "General"
   - Select "Allow all actions and reusable workflows"

2. **Configure Branch Protection**:
   - Go to "Settings" → "Branches"
   - Add rule for `main` branch:
     - Require status checks to pass
     - Require up-to-date branches
     - Include administrators

### **Step 2: Secrets Configuration**

Add these secrets to your repository (`Settings` → `Secrets and variables` → `Actions`):

#### **Required Secrets**:
```bash
# Docker Hub (for image publishing)
DOCKER_USERNAME=your_docker_username
DOCKER_PASSWORD=your_docker_password

# PyPI (for package publishing)
PYPI_API_TOKEN=your_pypi_token

# GitHub Token (usually auto-provided)
GITHUB_TOKEN=auto_provided

# Notifications (optional)
SLACK_WEBHOOK_URL=your_slack_webhook
TEAMS_WEBHOOK_URL=your_teams_webhook
```

#### **Environment Variables**:
```bash
# Application Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agent_framework_test
POSTGRES_USER=test_user
POSTGRES_PASSWORD=test_password
```

### **Step 3: Workflow Configuration**

#### **Customize Branch Names**:
Edit workflow files to match your branch naming:
```yaml
on:
  push:
    branches: [ main, master, develop ]  # Update as needed
  pull_request:
    branches: [ main, master ]           # Update as needed
```

#### **Adjust Python Versions**:
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11']  # Add/remove versions
```

#### **Configure Deployment Environments**:
1. Create environments in GitHub:
   - Go to "Settings" → "Environments"
   - Create "staging" and "production" environments
   - Add protection rules and secrets

## 📊 Workflow Usage

### **Continuous Integration**

Every push and pull request triggers:
1. **Code Quality Checks**:
   ```bash
   # Code formatting
   black --check src/ tests/ examples/
   
   # Import sorting
   isort --check-only src/ tests/ examples/
   
   # Linting
   flake8 src/ tests/ examples/
   
   # Type checking
   mypy src/
   ```

2. **Testing**:
   ```bash
   # Unit tests with coverage
   pytest tests/ -v --cov=src/ --cov-report=xml
   
   # Integration tests
   pytest tests/integration/ -v
   ```

3. **Security Scanning**:
   ```bash
   # Security vulnerabilities
   bandit -r src/
   
   # Dependency scanning
   safety check
   ```

### **Release Process**

1. **Automated Release** (on Git tag):
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Manual Release**:
   - Go to "Actions" → "Release Pipeline"
   - Click "Run workflow"
   - Enter version number

### **Performance Monitoring**

- **Daily Benchmarks**: Automated performance testing
- **PR Comparisons**: Performance impact analysis
- **Load Testing**: Stress testing under load

## 🔧 Customization

### **Adding New Tests**

1. **Unit Tests**:
   ```python
   # tests/unit/test_new_feature.py
   import pytest
   
   def test_new_feature():
       assert True
   ```

2. **Integration Tests**:
   ```python
   # tests/integration/test_new_integration.py
   @pytest.mark.asyncio
   async def test_integration():
       # Integration test code
       pass
   ```

### **Custom Deployment Steps**

Add deployment commands to workflows:
```yaml
deploy-production:
  steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f deployment/k8s/
        kubectl rollout status deployment/intel-ai-framework
```

### **Additional Security Scans**

Add custom security tools:
```yaml
- name: Custom Security Scan
  run: |
    # Your custom security scanning commands
    custom-security-tool scan src/
```

## 📈 Monitoring & Metrics

### **GitHub Actions Insights**

Monitor workflow performance:
- Go to "Actions" tab
- View workflow runs and duration
- Analyze failure patterns

### **Code Coverage**

- Coverage reports uploaded to artifacts
- Integration with Codecov (optional)
- Coverage badges in README

### **Performance Tracking**

- Benchmark results stored as artifacts
- Performance regression alerts
- Load test reports

## 🚨 Troubleshooting

### **Common Issues**

1. **Test Failures**:
   ```bash
   # Check logs in Actions tab
   # Fix failing tests locally:
   pytest tests/unit/test_failing.py -v
   ```

2. **Docker Build Failures**:
   ```bash
   # Test locally:
   docker build -t test-image .
   ```

3. **Dependency Issues**:
   ```bash
   # Update requirements:
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

### **Debugging Workflows**

1. **Enable Debug Logging**:
   ```yaml
   env:
     ACTIONS_STEP_DEBUG: true
     ACTIONS_RUNNER_DEBUG: true
   ```

2. **Use tmate for SSH Access**:
   ```yaml
   - name: Setup tmate session
     uses: mxschmitt/action-tmate@v3
   ```

## 📚 Best Practices

### **Workflow Organization**

1. **Keep workflows focused**: One responsibility per workflow
2. **Use reusable actions**: DRY principle for common tasks
3. **Optimize for speed**: Cache dependencies, run in parallel
4. **Fail fast**: Run quick checks first

### **Security**

1. **Minimize permissions**: Use least privilege principle
2. **Scan dependencies**: Regular vulnerability checks
3. **Secure secrets**: Use GitHub secrets, not hardcoded values
4. **Review third-party actions**: Only use trusted actions

### **Maintenance**

1. **Regular updates**: Keep actions and dependencies current
2. **Monitor performance**: Track workflow execution times
3. **Clean up artifacts**: Remove old artifacts to save space
4. **Document changes**: Update this guide when modifying workflows

## 🔗 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Action Marketplace](https://github.com/marketplace?type=actions)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides)

---

**Next Steps**: Commit these workflow files to your repository and watch the automation magic happen! 🎉
