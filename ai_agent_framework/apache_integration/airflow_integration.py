"""
Apache Airflow integration for the AI Agent Framework

This module provides integration with Apache Airflow for workflow orchestration
and task scheduling.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import time

# Airflow imports
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.utils.dates import days_ago
    from airflow.models import Variable
    from airflow.hooks.base import BaseHook
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    logging.warning("Apache Airflow not available. Install apache-airflow to enable Airflow integration.")

# Framework imports
from ..core.workflow import Workflow, WorkflowResult
from ..core.task import Task, TaskResult, TaskStatus
from ..core.agent import Agent, AgentContext


@dataclass
class AirflowConfig:
    """Airflow integration configuration"""
    dag_dir: str = "/opt/airflow/dags"
    default_args: Optional[Dict[str, Any]] = None
    schedule_interval: Optional[str] = None
    catchup: bool = False
    max_active_runs: int = 1
    tags: Optional[List[str]] = None


class AirflowTaskWrapper:
    """Wrapper for framework tasks to be used in Airflow"""
    
    def __init__(self, task: Task, agent: Optional[Agent] = None):
        self.task = task
        self.agent = agent
        self.logger = logging.getLogger(__name__)
    
    async def execute_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute framework task in Airflow context"""
        try:
            # Create agent context if agent is provided
            if self.agent:
                agent_context = AgentContext(
                    agent_id=self.agent.config.agent_id,
                    inputs=context.get("inputs", {}),
                    metadata=context.get("metadata", {}),
                    correlation_id=context.get("task_instance_key_str", "")
                )
                
                # Run agent
                result = await self.agent.run(agent_context)
                return result
            else:
                # Execute task directly
                task_result = await self.task.execute(context)
                return {
                    "status": task_result.status.value,
                    "result": task_result.result,
                    "error": task_result.error,
                    "metadata": task_result.metadata
                }
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "result": None
            }
    
    def to_python_operator(
        self,
        task_id: str,
        dag: 'DAG',
        **kwargs
    ) -> 'PythonOperator':
        """Convert to Airflow PythonOperator"""
        if not AIRFLOW_AVAILABLE:
            raise RuntimeError("Airflow is not available")
        
        def sync_execute(**context):
            """Synchronous wrapper for async task execution"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.execute_task(context))
            finally:
                loop.close()
        
        return PythonOperator(
            task_id=task_id,
            python_callable=sync_execute,
            dag=dag,
            **kwargs
        )


class AirflowWorkflowConverter:
    """Converts framework workflows to Airflow DAGs"""
    
    def __init__(self, config: AirflowConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def workflow_to_dag(
        self,
        workflow: Workflow,
        dag_id: Optional[str] = None,
        description: Optional[str] = None,
        agents: Optional[Dict[str, Agent]] = None
    ) -> 'DAG':
        """
        Convert framework workflow to Airflow DAG
        
        Args:
            workflow: Framework workflow to convert
            dag_id: Airflow DAG ID (defaults to workflow.workflow_id)
            description: DAG description
            agents: Dictionary of agent_id -> Agent for task execution
            
        Returns:
            Airflow DAG instance
        """
        if not AIRFLOW_AVAILABLE:
            raise RuntimeError("Airflow is not available")
        
        # Set defaults
        dag_id = dag_id or workflow.workflow_id
        description = description or f"Generated from workflow: {workflow.name}"
        agents = agents or {}
        
        # Default arguments
        default_args = self.config.default_args or {
            'owner': 'ai-agent-framework',
            'depends_on_past': False,
            'start_date': days_ago(1),
            'email_on_failure': False,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }
        
        # Create DAG
        dag = DAG(
            dag_id=dag_id,
            default_args=default_args,
            description=description,
            schedule_interval=self.config.schedule_interval,
            catchup=self.config.catchup,
            max_active_runs=self.config.max_active_runs,
            tags=self.config.tags or ['ai-agent-framework']
        )
        
        # Convert tasks
        airflow_tasks = {}
        for task_id, task in workflow.tasks.items():
            # Find associated agent
            agent = agents.get(task_id)
            
            # Create wrapper
            wrapper = AirflowTaskWrapper(task, agent)
            
            # Create Airflow operator
            airflow_task = wrapper.to_python_operator(
                task_id=task_id,
                dag=dag
            )
            
            airflow_tasks[task_id] = airflow_task
        
        # Set up dependencies
        for task_id, dependencies in workflow.dependencies.items():
            if task_id in airflow_tasks:
                airflow_task = airflow_tasks[task_id]
                
                for dep_id in dependencies:
                    if dep_id in airflow_tasks:
                        airflow_tasks[dep_id] >> airflow_task
        
        self.logger.info(f"Created Airflow DAG: {dag_id} with {len(airflow_tasks)} tasks")
        return dag
    
    def save_dag_file(
        self,
        dag: 'DAG',
        filename: Optional[str] = None
    ) -> str:
        """
        Save DAG to Python file in DAGs directory
        
        Args:
            dag: Airflow DAG to save
            filename: Output filename (defaults to dag_id.py)
            
        Returns:
            Path to saved file
        """
        filename = filename or f"{dag.dag_id}.py"
        dag_path = Path(self.config.dag_dir) / filename
        
        # Ensure directory exists
        dag_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate DAG file content
        dag_content = self._generate_dag_file_content(dag)
        
        # Write file
        with open(dag_path, 'w') as f:
            f.write(dag_content)
        
        self.logger.info(f"Saved DAG file: {dag_path}")
        return str(dag_path)
    
    def _generate_dag_file_content(self, dag: 'DAG') -> str:
        """Generate Python code for DAG file"""
        # This is a simplified version - in practice, you'd need to
        # serialize the entire DAG structure properly
        
        return f'''
"""
Auto-generated Airflow DAG from AI Agent Framework
DAG ID: {dag.dag_id}
Generated at: {datetime.now().isoformat()}
"""

import asyncio
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# AI Agent Framework imports
from ai_agent_framework.apache_integration.airflow_integration import AirflowTaskWrapper
from ai_agent_framework.core.task import Task
from ai_agent_framework.core.agent import Agent

# Default arguments
default_args = {{
    'owner': 'ai-agent-framework',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}}

# Create DAG
dag = DAG(
    '{dag.dag_id}',
    default_args=default_args,
    description='{dag.description}',
    schedule_interval={repr(dag.schedule_interval)},
    catchup={dag.catchup},
    max_active_runs={dag.max_active_runs},
    tags={dag.tags}
)

# TODO: Add task definitions here
# This would need to be generated based on the actual workflow tasks

'''


class AirflowAgentScheduler:
    """Schedules agents using Airflow"""
    
    def __init__(self, config: AirflowConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scheduled_agents: Dict[str, 'DAG'] = {}
    
    async def schedule_agent(
        self,
        agent: Agent,
        schedule_interval: str,
        dag_id: Optional[str] = None,
        **dag_kwargs
    ) -> str:
        """
        Schedule agent execution using Airflow
        
        Args:
            agent: Agent to schedule
            schedule_interval: Cron expression or Airflow preset
            dag_id: DAG identifier
            **dag_kwargs: Additional DAG arguments
            
        Returns:
            DAG ID of scheduled agent
        """
        if not AIRFLOW_AVAILABLE:
            raise RuntimeError("Airflow is not available")
        
        dag_id = dag_id or f"agent_{agent.config.agent_id}"
        
        # Default arguments
        default_args = self.config.default_args or {
            'owner': 'ai-agent-framework',
            'depends_on_past': False,
            'start_date': days_ago(1),
            'email_on_failure': False,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }
        
        # Create DAG for agent
        dag = DAG(
            dag_id=dag_id,
            default_args=default_args,
            description=f"Scheduled execution of agent: {agent.config.name}",
            schedule_interval=schedule_interval,
            catchup=self.config.catchup,
            max_active_runs=self.config.max_active_runs,
            tags=['ai-agent-framework', 'scheduled-agent'],
            **dag_kwargs
        )
        
        # Create agent execution task
        def execute_agent(**context):
            """Execute agent in Airflow context"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Create agent context from Airflow context
                agent_context = AgentContext(
                    agent_id=agent.config.agent_id,
                    inputs=context.get('dag_run').conf or {},
                    metadata={
                        'dag_id': context['dag'].dag_id,
                        'run_id': context['dag_run'].run_id,
                        'execution_date': context['execution_date'].isoformat()
                    },
                    correlation_id=context.get('task_instance_key_str', '')
                )
                
                # Run agent
                return loop.run_until_complete(agent.run(agent_context))
            finally:
                loop.close()
        
        # Create PythonOperator
        agent_task = PythonOperator(
            task_id=f"execute_{agent.config.agent_id}",
            python_callable=execute_agent,
            dag=dag
        )
        
        # Store DAG
        self.scheduled_agents[dag_id] = dag
        
        self.logger.info(f"Scheduled agent {agent.config.agent_id} with DAG ID: {dag_id}")
        return dag_id
    
    def get_scheduled_agent(self, dag_id: str) -> Optional['DAG']:
        """Get scheduled agent DAG by ID"""
        return self.scheduled_agents.get(dag_id)
    
    def list_scheduled_agents(self) -> List[str]:
        """List all scheduled agent DAG IDs"""
        return list(self.scheduled_agents.keys())
    
    def unschedule_agent(self, dag_id: str) -> bool:
        """Remove agent from scheduler"""
        if dag_id in self.scheduled_agents:
            del self.scheduled_agents[dag_id]
            self.logger.info(f"Unscheduled agent DAG: {dag_id}")
            return True
        return False


class AirflowIntegration:
    """Main Airflow integration class"""
    
    def __init__(self, config: Optional[AirflowConfig] = None):
        self.config = config or AirflowConfig()
        self.converter = AirflowWorkflowConverter(self.config)
        self.scheduler = AirflowAgentScheduler(self.config)
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize Airflow integration"""
        if not AIRFLOW_AVAILABLE:
            self.logger.warning("Airflow is not available. Some features will be disabled.")
            return
        
        # Ensure DAG directory exists
        Path(self.config.dag_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Airflow integration initialized")
    
    def deploy_workflow(
        self,
        workflow: Workflow,
        agents: Optional[Dict[str, Agent]] = None,
        dag_id: Optional[str] = None,
        save_to_file: bool = True
    ) -> str:
        """
        Deploy workflow to Airflow
        
        Args:
            workflow: Framework workflow to deploy
            agents: Associated agents for tasks
            dag_id: Custom DAG ID
            save_to_file: Whether to save DAG file
            
        Returns:
            DAG ID of deployed workflow
        """
        # Convert workflow to DAG
        dag = self.converter.workflow_to_dag(
            workflow, dag_id, agents=agents
        )
        
        # Save to file if requested
        if save_to_file:
            self.converter.save_dag_file(dag)
        
        return dag.dag_id
    
    async def schedule_agent_execution(
        self,
        agent: Agent,
        schedule: str,
        **kwargs
    ) -> str:
        """Schedule regular agent execution"""
        return await self.scheduler.schedule_agent(
            agent, schedule, **kwargs
        )
    
    def create_sensor_dag(
        self,
        dag_id: str,
        sensor_callable: Callable,
        poke_interval: int = 60,
        timeout: int = 7200
    ) -> 'DAG':
        """
        Create a sensor DAG for external event monitoring
        
        Args:
            dag_id: DAG identifier
            sensor_callable: Function to check sensor condition
            poke_interval: How often to check (seconds)
            timeout: Maximum wait time (seconds)
            
        Returns:
            Airflow DAG with sensor
        """
        if not AIRFLOW_AVAILABLE:
            raise RuntimeError("Airflow is not available")
        
        from airflow.sensors.python import PythonSensor
        
        default_args = self.config.default_args or {
            'owner': 'ai-agent-framework',
            'depends_on_past': False,
            'start_date': days_ago(1),
        }
        
        dag = DAG(
            dag_id=dag_id,
            default_args=default_args,
            description=f"Sensor DAG: {dag_id}",
            schedule_interval=None,
            catchup=False,
            tags=['ai-agent-framework', 'sensor']
        )
        
        sensor = PythonSensor(
            task_id='sensor_check',
            python_callable=sensor_callable,
            poke_interval=poke_interval,
            timeout=timeout,
            dag=dag
        )
        
        return dag
    
    def create_branch_dag(
        self,
        dag_id: str,
        branch_callable: Callable,
        task_mapping: Dict[str, Callable]
    ) -> 'DAG':
        """
        Create a branching DAG for conditional execution
        
        Args:
            dag_id: DAG identifier
            branch_callable: Function that returns task ID to execute
            task_mapping: Mapping of task_id -> callable
            
        Returns:
            Airflow DAG with branching logic
        """
        if not AIRFLOW_AVAILABLE:
            raise RuntimeError("Airflow is not available")
        
        from airflow.operators.python import BranchPythonOperator
        
        default_args = self.config.default_args or {
            'owner': 'ai-agent-framework',
            'depends_on_past': False,
            'start_date': days_ago(1),
        }
        
        dag = DAG(
            dag_id=dag_id,
            default_args=default_args,
            description=f"Branch DAG: {dag_id}",
            schedule_interval=None,
            catchup=False,
            tags=['ai-agent-framework', 'branch']
        )
        
        # Create branch operator
        branch_task = BranchPythonOperator(
            task_id='branch_decision',
            python_callable=branch_callable,
            dag=dag
        )
        
        # Create task operators
        for task_id, callable_func in task_mapping.items():
            task = PythonOperator(
                task_id=task_id,
                python_callable=callable_func,
                dag=dag
            )
            branch_task >> task
        
        return dag


# Convenience functions
def create_airflow_integration(
    dag_dir: str = "/opt/airflow/dags",
    **config_kwargs
) -> AirflowIntegration:
    """Create Airflow integration with configuration"""
    config = AirflowConfig(dag_dir=dag_dir, **config_kwargs)
    return AirflowIntegration(config)


async def deploy_workflow_to_airflow(
    workflow: Workflow,
    dag_dir: str = "/opt/airflow/dags",
    agents: Optional[Dict[str, Agent]] = None
) -> str:
    """
    Convenience function to deploy workflow to Airflow
    
    Args:
        workflow: Workflow to deploy
        dag_dir: Airflow DAGs directory
        agents: Associated agents
        
    Returns:
        DAG ID
    """
    integration = create_airflow_integration(dag_dir)
    await integration.initialize()
    
    return integration.deploy_workflow(workflow, agents)


# Example usage
async def example_airflow_integration():
    """Example of using Airflow integration"""
    from ..core.workflow import Workflow
    from ..core.task import FunctionTask
    from ..core.agent import SimpleAgent, AgentConfig
    from ..core.memory import InMemoryStorage
    
    # Create sample workflow
    workflow = Workflow("sample_workflow")
    
    task1 = FunctionTask("task1", lambda x: x + 1, [1])
    task2 = FunctionTask("task2", lambda x: x * 2, [2])
    
    workflow.add_task(task1)
    workflow.add_task(task2, dependencies=["task1"])
    
    # Create sample agent
    memory = InMemoryStorage()
    agent_config = AgentConfig(
        agent_id="sample_agent",
        name="Sample Agent"
    )
    agent = SimpleAgent(agent_config, memory)
    
    # Deploy to Airflow
    integration = create_airflow_integration()
    await integration.initialize()
    
    # Deploy workflow
    dag_id = integration.deploy_workflow(
        workflow,
        agents={"task1": agent, "task2": agent}
    )
    
    print(f"Deployed workflow to Airflow with DAG ID: {dag_id}")
    
    # Schedule agent execution
    scheduled_dag_id = await integration.schedule_agent_execution(
        agent,
        schedule="0 */6 * * *"  # Every 6 hours
    )
    
    print(f"Scheduled agent with DAG ID: {scheduled_dag_id}")


if __name__ == "__main__":
    asyncio.run(example_airflow_integration())