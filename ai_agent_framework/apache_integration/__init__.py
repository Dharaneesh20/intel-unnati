"""
Apache Integration package initialization

This module provides integrations with Apache components including
Kafka, Airflow, and Camel for the AI Agent Framework.
"""

from .kafka_integration import (
    KafkaConfig,
    KafkaMessage,
    MessageType,
    KafkaProducerManager,
    KafkaConsumerManager,
    KafkaAgentMessaging,
    create_kafka_messaging,
    kafka_messaging_context
)

from .airflow_integration import (
    AirflowConfig,
    AirflowTaskWrapper,
    AirflowWorkflowConverter,
    AirflowAgentScheduler,
    AirflowIntegration,
    create_airflow_integration,
    deploy_workflow_to_airflow
)

from .camel_integration import (
    CamelConfig,
    CamelEndpoint,
    CamelRoute,
    IntegrationPattern,
    CamelRestClient,
    CamelRouteBuilder,
    RouteDefinition,
    ChoiceDefinition,
    CamelAgentIntegration,
    create_camel_integration,
    build_enterprise_patterns
)

__all__ = [
    # Kafka integration
    'KafkaConfig',
    'KafkaMessage',
    'MessageType',
    'KafkaProducerManager',
    'KafkaConsumerManager',
    'KafkaAgentMessaging',
    'create_kafka_messaging',
    'kafka_messaging_context',
    
    # Airflow integration
    'AirflowConfig',
    'AirflowTaskWrapper',
    'AirflowWorkflowConverter',
    'AirflowAgentScheduler',
    'AirflowIntegration',
    'create_airflow_integration',
    'deploy_workflow_to_airflow',
    
    # Camel integration
    'CamelConfig',
    'CamelEndpoint',
    'CamelRoute',
    'IntegrationPattern',
    'CamelRestClient',
    'CamelRouteBuilder',
    'RouteDefinition',
    'ChoiceDefinition',
    'CamelAgentIntegration',
    'create_camel_integration',
    'build_enterprise_patterns'
]