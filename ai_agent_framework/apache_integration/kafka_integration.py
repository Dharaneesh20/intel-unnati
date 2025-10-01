"""
Apache Kafka integration for the AI Agent Framework

This module provides integration with Apache Kafka for distributed messaging
and event-driven architectures.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import time
from contextlib import asynccontextmanager

# Kafka imports
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("Apache Kafka not available. Install kafka-python to enable Kafka integration.")


class MessageType(Enum):
    """Types of messages"""
    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"
    AGENT_STATUS = "agent_status"
    WORKFLOW_EVENT = "workflow_event"
    SYSTEM_EVENT = "system_event"


@dataclass
class KafkaMessage:
    """Kafka message structure"""
    message_id: str
    message_type: MessageType
    topic: str
    payload: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None
    reply_topic: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


@dataclass
class KafkaConfig:
    """Kafka configuration"""
    bootstrap_servers: List[str]
    client_id: str = "ai-agent-framework"
    group_id: str = "ai-agents"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 3000
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None


class KafkaProducerManager:
    """Manages Kafka producer instances"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer: Optional[KafkaProducer] = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize Kafka producer"""
        if not KAFKA_AVAILABLE:
            raise RuntimeError("Kafka is not available. Install kafka-python.")
        
        try:
            producer_config = {
                'bootstrap_servers': self.config.bootstrap_servers,
                'client_id': self.config.client_id,
                'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
                'key_serializer': lambda k: str(k).encode('utf-8') if k else None,
                'acks': 'all',
                'retries': 3,
                'batch_size': 16384,
                'linger_ms': 10,
                'buffer_memory': 33554432,
                'security_protocol': self.config.security_protocol
            }
            
            # Add authentication if configured
            if self.config.sasl_mechanism:
                producer_config.update({
                    'sasl_mechanism': self.config.sasl_mechanism,
                    'sasl_plain_username': self.config.sasl_username,
                    'sasl_plain_password': self.config.sasl_password
                })
            
            # Add SSL if configured
            if self.config.ssl_cafile:
                producer_config.update({
                    'ssl_cafile': self.config.ssl_cafile,
                    'ssl_certfile': self.config.ssl_certfile,
                    'ssl_keyfile': self.config.ssl_keyfile
                })
            
            self.producer = KafkaProducer(**producer_config)
            self.logger.info("Kafka producer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    async def send_message(
        self,
        topic: str,
        message: KafkaMessage,
        key: Optional[str] = None,
        partition: Optional[int] = None
    ) -> bool:
        """
        Send message to Kafka topic
        
        Args:
            topic: Kafka topic name
            message: Message to send
            key: Message key for partitioning
            partition: Specific partition to send to
            
        Returns:
            Success status
        """
        if not self.producer:
            await self.initialize()
        
        try:
            # Convert message to dict
            message_dict = asdict(message)
            
            # Add headers if present
            headers = []
            if message.headers:
                headers = [(k, v.encode('utf-8')) for k, v in message.headers.items()]
            
            # Send message
            future = self.producer.send(
                topic=topic,
                value=message_dict,
                key=key,
                partition=partition,
                headers=headers
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            self.logger.debug(
                f"Message sent successfully to {record_metadata.topic}:"
                f"{record_metadata.partition}:{record_metadata.offset}"
            )
            
            return True
            
        except KafkaError as e:
            self.logger.error(f"Failed to send Kafka message: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending message: {e}")
            return False
    
    async def send_task_request(
        self,
        agent_id: str,
        task_data: Dict[str, Any],
        correlation_id: Optional[str] = None,
        reply_topic: Optional[str] = None
    ) -> bool:
        """Send task request message"""
        message = KafkaMessage(
            message_id=f"task_req_{int(time.time() * 1000)}",
            message_type=MessageType.TASK_REQUEST,
            topic="agent-tasks",
            payload={
                "agent_id": agent_id,
                "task_data": task_data
            },
            timestamp=time.time(),
            correlation_id=correlation_id,
            reply_topic=reply_topic
        )
        
        return await self.send_message("agent-tasks", message, key=agent_id)
    
    async def send_task_result(
        self,
        task_id: str,
        result: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> bool:
        """Send task result message"""
        message = KafkaMessage(
            message_id=f"task_result_{int(time.time() * 1000)}",
            message_type=MessageType.TASK_RESULT,
            topic="task-results",
            payload={
                "task_id": task_id,
                "result": result
            },
            timestamp=time.time(),
            correlation_id=correlation_id
        )
        
        return await self.send_message("task-results", message, key=task_id)
    
    async def close(self):
        """Close producer connection"""
        if self.producer:
            self.producer.close()
            self.producer = None
            self.logger.info("Kafka producer closed")


class KafkaConsumerManager:
    """Manages Kafka consumer instances"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize Kafka consumer"""
        if not KAFKA_AVAILABLE:
            raise RuntimeError("Kafka is not available. Install kafka-python.")
        
        self.logger.info("Kafka consumer manager initialized")
    
    def register_handler(self, topic: str, handler: Callable[[KafkaMessage], None]):
        """Register message handler for topic"""
        self.message_handlers[topic] = handler
        self.logger.info(f"Registered handler for topic: {topic}")
    
    async def subscribe_to_topic(self, topic: str) -> bool:
        """Subscribe to Kafka topic"""
        try:
            consumer_config = {
                'bootstrap_servers': self.config.bootstrap_servers,
                'group_id': self.config.group_id,
                'client_id': f"{self.config.client_id}-consumer",
                'auto_offset_reset': self.config.auto_offset_reset,
                'enable_auto_commit': self.config.enable_auto_commit,
                'max_poll_records': self.config.max_poll_records,
                'session_timeout_ms': self.config.session_timeout_ms,
                'heartbeat_interval_ms': self.config.heartbeat_interval_ms,
                'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
                'key_deserializer': lambda k: k.decode('utf-8') if k else None,
                'security_protocol': self.config.security_protocol
            }
            
            # Add authentication if configured
            if self.config.sasl_mechanism:
                consumer_config.update({
                    'sasl_mechanism': self.config.sasl_mechanism,
                    'sasl_plain_username': self.config.sasl_username,
                    'sasl_plain_password': self.config.sasl_password
                })
            
            # Add SSL if configured
            if self.config.ssl_cafile:
                consumer_config.update({
                    'ssl_cafile': self.config.ssl_cafile,
                    'ssl_certfile': self.config.ssl_certfile,
                    'ssl_keyfile': self.config.ssl_keyfile
                })
            
            consumer = KafkaConsumer(**consumer_config)
            consumer.subscribe([topic])
            
            self.consumers[topic] = consumer
            self.logger.info(f"Subscribed to topic: {topic}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to topic {topic}: {e}")
            return False
    
    async def start_consuming(self):
        """Start consuming messages from all subscribed topics"""
        if not self.consumers:
            self.logger.warning("No topics subscribed")
            return
        
        self.running = True
        self.logger.info("Starting Kafka message consumption")
        
        # Create consumer tasks for each topic
        tasks = []
        for topic, consumer in self.consumers.items():
            task = asyncio.create_task(self._consume_topic(topic, consumer))
            tasks.append(task)
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in message consumption: {e}")
        finally:
            self.running = False
    
    async def _consume_topic(self, topic: str, consumer: KafkaConsumer):
        """Consume messages from a specific topic"""
        handler = self.message_handlers.get(topic)
        if not handler:
            self.logger.warning(f"No handler registered for topic: {topic}")
            return
        
        try:
            while self.running:
                # Poll for messages
                message_batch = consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Parse message
                            kafka_message = self._parse_message(message)
                            
                            # Call handler
                            await self._call_handler(handler, kafka_message)
                            
                        except Exception as e:
                            self.logger.error(f"Error processing message: {e}")
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
        except Exception as e:
            self.logger.error(f"Error consuming from topic {topic}: {e}")
    
    def _parse_message(self, raw_message) -> KafkaMessage:
        """Parse raw Kafka message to KafkaMessage"""
        try:
            data = raw_message.value
            
            return KafkaMessage(
                message_id=data.get('message_id', ''),
                message_type=MessageType(data.get('message_type', 'system_event')),
                topic=raw_message.topic,
                payload=data.get('payload', {}),
                timestamp=data.get('timestamp', time.time()),
                correlation_id=data.get('correlation_id'),
                reply_topic=data.get('reply_topic'),
                headers=self._parse_headers(raw_message.headers)
            )
        except Exception as e:
            self.logger.error(f"Failed to parse message: {e}")
            # Return a default message
            return KafkaMessage(
                message_id="parse_error",
                message_type=MessageType.SYSTEM_EVENT,
                topic=raw_message.topic,
                payload={"error": "Failed to parse message"},
                timestamp=time.time()
            )
    
    def _parse_headers(self, raw_headers) -> Optional[Dict[str, str]]:
        """Parse message headers"""
        if not raw_headers:
            return None
        
        headers = {}
        for key, value in raw_headers:
            headers[key] = value.decode('utf-8') if isinstance(value, bytes) else str(value)
        
        return headers
    
    async def _call_handler(self, handler: Callable, message: KafkaMessage):
        """Call message handler (async or sync)"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                handler(message)
        except Exception as e:
            self.logger.error(f"Handler error: {e}")
    
    async def stop_consuming(self):
        """Stop consuming messages"""
        self.running = False
        self.logger.info("Stopping Kafka message consumption")
    
    async def close(self):
        """Close all consumer connections"""
        await self.stop_consuming()
        
        for topic, consumer in self.consumers.items():
            consumer.close()
            self.logger.info(f"Closed consumer for topic: {topic}")
        
        self.consumers.clear()


class KafkaAgentMessaging:
    """High-level messaging interface for agents"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer = KafkaProducerManager(config)
        self.consumer = KafkaConsumerManager(config)
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize messaging system"""
        await self.producer.initialize()
        await self.consumer.initialize()
        
        # Subscribe to standard topics
        await self.consumer.subscribe_to_topic("agent-tasks")
        await self.consumer.subscribe_to_topic("task-results")
        await self.consumer.subscribe_to_topic("agent-status")
        await self.consumer.subscribe_to_topic("workflow-events")
        
        self.logger.info("Kafka agent messaging initialized")
    
    def register_task_handler(self, handler: Callable[[KafkaMessage], None]):
        """Register handler for incoming task requests"""
        self.consumer.register_handler("agent-tasks", handler)
    
    def register_result_handler(self, handler: Callable[[KafkaMessage], None]):
        """Register handler for task results"""
        self.consumer.register_handler("task-results", handler)
    
    def register_status_handler(self, handler: Callable[[KafkaMessage], None]):
        """Register handler for agent status updates"""
        self.consumer.register_handler("agent-status", handler)
    
    def register_workflow_handler(self, handler: Callable[[KafkaMessage], None]):
        """Register handler for workflow events"""
        self.consumer.register_handler("workflow-events", handler)
    
    async def send_task_request(
        self,
        agent_id: str,
        task_data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> bool:
        """Send task request to agent"""
        return await self.producer.send_task_request(
            agent_id, task_data, correlation_id
        )
    
    async def send_task_result(
        self,
        task_id: str,
        result: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> bool:
        """Send task execution result"""
        return await self.producer.send_task_result(
            task_id, result, correlation_id
        )
    
    async def send_agent_status(
        self,
        agent_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send agent status update"""
        message = KafkaMessage(
            message_id=f"status_{int(time.time() * 1000)}",
            message_type=MessageType.AGENT_STATUS,
            topic="agent-status",
            payload={
                "agent_id": agent_id,
                "status": status,
                "metadata": metadata or {}
            },
            timestamp=time.time()
        )
        
        return await self.producer.send_message("agent-status", message, key=agent_id)
    
    async def send_workflow_event(
        self,
        workflow_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """Send workflow event"""
        message = KafkaMessage(
            message_id=f"workflow_{int(time.time() * 1000)}",
            message_type=MessageType.WORKFLOW_EVENT,
            topic="workflow-events",
            payload={
                "workflow_id": workflow_id,
                "event_type": event_type,
                "event_data": event_data
            },
            timestamp=time.time()
        )
        
        return await self.producer.send_message("workflow-events", message, key=workflow_id)
    
    async def start_consuming(self):
        """Start consuming messages"""
        await self.consumer.start_consuming()
    
    async def stop_consuming(self):
        """Stop consuming messages"""
        await self.consumer.stop_consuming()
    
    async def close(self):
        """Close messaging system"""
        await self.producer.close()
        await self.consumer.close()


# Convenience functions
async def create_kafka_messaging(
    bootstrap_servers: List[str],
    group_id: str = "ai-agents",
    client_id: str = "ai-agent-framework"
) -> KafkaAgentMessaging:
    """
    Create and initialize Kafka messaging system
    
    Args:
        bootstrap_servers: List of Kafka broker addresses
        group_id: Consumer group ID
        client_id: Client identifier
        
    Returns:
        Initialized KafkaAgentMessaging instance
    """
    config = KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        client_id=client_id
    )
    
    messaging = KafkaAgentMessaging(config)
    await messaging.initialize()
    
    return messaging


@asynccontextmanager
async def kafka_messaging_context(
    bootstrap_servers: List[str],
    group_id: str = "ai-agents"
):
    """
    Context manager for Kafka messaging
    
    Usage:
        async with kafka_messaging_context(["localhost:9092"]) as messaging:
            await messaging.send_task_request("agent-1", {"data": "test"})
    """
    messaging = await create_kafka_messaging(bootstrap_servers, group_id)
    try:
        yield messaging
    finally:
        await messaging.close()


# Example usage
async def example_kafka_integration():
    """Example of using Kafka integration"""
    
    # Create messaging system
    messaging = await create_kafka_messaging(["localhost:9092"])
    
    # Register handlers
    async def handle_task_request(message: KafkaMessage):
        print(f"Received task request: {message.payload}")
        
        # Process task and send result
        result = {"status": "completed", "data": "processed"}
        task_id = message.payload.get("task_data", {}).get("task_id", "unknown")
        
        await messaging.send_task_result(
            task_id, result, message.correlation_id
        )
    
    messaging.register_task_handler(handle_task_request)
    
    # Send a task request
    await messaging.send_task_request(
        "document-processor",
        {"task_id": "task-123", "document_path": "/path/to/doc.pdf"}
    )
    
    # Start consuming (this would run in the background)
    # await messaging.start_consuming()
    
    # Clean up
    await messaging.close()


if __name__ == "__main__":
    asyncio.run(example_kafka_integration())