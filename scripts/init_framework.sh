#!/bin/bash

# Intel AI Agent Framework - Initialization Script

set -e  # Exit on any error

echo "🚀 Initializing Intel AI Agent Framework..."

# Check if running on Intel DevCloud
if [ -n "$PBS_NODEFILE" ]; then
    echo "✓ Running on Intel DevCloud"
    export INTEL_DEVCLOUD=true
else
    echo "ℹ Running on local environment"
    export INTEL_DEVCLOUD=false
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data models config deployment/grafana deployment/prometheus

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python 3.9+ is available
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "✓ Python 3.9+ available"
else
    echo "❌ Python 3.9+ required"
    exit 1
fi

# Set up virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Install development dependencies if requested
if [ "$1" = "--dev" ]; then
    echo "📦 Installing development dependencies..."
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    fi
fi

# Set up Intel OpenVINO if available
echo "🔧 Setting up Intel OpenVINO..."
if command -v setupvars.sh &> /dev/null; then
    source setupvars.sh
    echo "✓ Intel OpenVINO environment initialized"
else
    echo "ℹ Intel OpenVINO not found - will use CPU inference"
fi

# Create default configuration
echo "⚙️ Creating default configuration..."
cat > config/framework.yaml << EOF
framework:
  name: "intel-ai-agent-framework"
  version: "1.0.0"
  environment: "development"
  debug: true

database:
  host: "localhost"
  port: 5432
  database: "agent_framework"
  username: "agent_user"
  password: "secure_password"

redis:
  host: "localhost"
  port: 6379
  db: 0

kafka:
  bootstrap_servers: "localhost:9092"
  group_id: "agent_framework"

openvino:
  model_path: "./models"
  device: "CPU"
  num_threads: 4

observability:
  metrics_enabled: true
  tracing_enabled: true
  log_level: "INFO"
  prometheus_port: 9090

orchestrator:
  engine: "airflow"
  max_concurrent_workflows: 100
  default_timeout: 300

executor:
  default_executor: "thread_pool"
  max_workers: 10
  max_retries: 3
  backoff_factor: 2.0

memory:
  provider: "redis"
  ttl: 3600
  max_memory_mb: 1024
EOF

# Create environment file
echo "🔧 Creating environment file..."
cat > .env << EOF
# Intel AI Agent Framework Environment Variables

# Core Settings
AGENT_FRAMEWORK_ENV=development
AGENT_FRAMEWORK_LOG_LEVEL=INFO
AGENT_FRAMEWORK_DEBUG=true

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agent_framework
POSTGRES_USER=agent_user
POSTGRES_PASSWORD=secure_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_GROUP_ID=agent_framework

# Intel OpenVINO
OPENVINO_MODEL_PATH=./models
OPENVINO_DEVICE=CPU
OPENVINO_NUM_THREADS=4

# Observability
METRICS_ENABLED=true
TRACING_ENABLED=true
PROMETHEUS_PORT=9090
JAEGER_ENDPOINT=http://localhost:14268/api/traces
EOF

# Download sample models if on Intel DevCloud
if [ "$INTEL_DEVCLOUD" = "true" ]; then
    echo "📥 Setting up Intel DevCloud optimizations..."
    # Create model directory
    mkdir -p models/openvino
    
    # Note: In a real implementation, you would download actual Intel-optimized models
    echo "ℹ Model download placeholder - implement actual model downloading"
fi

# Initialize database schema (placeholder)
echo "🗄️ Initializing database schema..."
cat > deployment/init-db.sql << EOF
-- Intel AI Agent Framework Database Schema

-- Workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows(id),
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    inputs JSONB,
    outputs JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Executions table
CREATE TABLE IF NOT EXISTS executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id),
    workflow_id UUID REFERENCES workflows(id),
    status VARCHAR(50) DEFAULT 'running',
    inputs JSONB,
    outputs JSONB,
    error_message TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Conversation tracking
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id VARCHAR(255),
    channel VARCHAR(50),
    status VARCHAR(50) DEFAULT 'active',
    messages JSONB,
    metadata JSONB,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_tasks_workflow_id ON tasks(workflow_id);
CREATE INDEX IF NOT EXISTS idx_executions_agent_id ON executions(agent_id);
CREATE INDEX IF NOT EXISTS idx_conversations_customer_id ON conversations(customer_id);
EOF

# Create Prometheus configuration
echo "📊 Creating Prometheus configuration..."
cat > deployment/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'agent-framework'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
EOF

# Create Grafana provisioning
echo "📈 Creating Grafana configuration..."
mkdir -p deployment/grafana/dashboards deployment/grafana/datasources

cat > deployment/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Run health checks
echo "🏥 Running health checks..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "✓ Docker available"
    
    # Check if Docker Compose is available
    if command -v docker-compose &> /dev/null; then
        echo "✓ Docker Compose available"
        echo "ℹ You can start infrastructure with: docker-compose up -d"
    else
        echo "⚠️ Docker Compose not found - install for full infrastructure support"
    fi
else
    echo "⚠️ Docker not found - some features may not work"
fi

# Validate Python imports
echo "🔍 Validating Python imports..."
python3 -c "
import sys
try:
    import fastapi
    import uvicorn
    import pydantic
    import redis
    import sqlalchemy
    print('✓ Core dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 Intel AI Agent Framework initialization complete!"
echo ""
echo "Next steps:"
echo "1. Start infrastructure: docker-compose up -d"
echo "2. Run the API server: uvicorn src.api.main:app --reload"
echo "3. Access Grafana dashboard: http://localhost:3000 (admin/admin)"
echo "4. View API docs: http://localhost:8000/docs"
echo ""
echo "Example usage:"
echo "• Document Processing: cd examples/document_processing_agent && python run_agent.py --input-dir ./sample_docs"
echo "• Customer Support: cd examples/customer_support_agent && python run_agent.py --interactive"
echo ""

# Set executable permissions
chmod +x scripts/init_framework.py 2>/dev/null || true
