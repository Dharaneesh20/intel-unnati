"""
Customer Support Agent - Reference Implementation

This agent demonstrates a multi-channel customer support workflow:
1. Intent classification and understanding
2. Knowledge base search and retrieval
3. Sentiment analysis and priority assessment
4. Automated response generation
5. Escalation and human handoff workflows
6. Conversation tracking and analytics
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.state_machine import StateMachine, State, Transition, StateType, TransitionType
from src.core.agents import Agent
from src.tools.support_tools import (
    IntentClassificationTool,
    KnowledgeBaseTool,
    SentimentAnalysisTool,
    ResponseGeneratorTool,
    EscalationTool,
    ConversationTrackerTool
)


class CustomerSupportAgent:
    """Customer support agent with state machine workflow."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create the customer support state machine and agent."""
        
        # Create state machine
        state_machine = StateMachine(name="customer_support_workflow")
        
        # Define states
        initial_state = State(
            name="initial",
            state_type=StateType.INITIAL,
            action=self._initialize_conversation
        )
        
        intent_analysis_state = State(
            name="intent_analysis",
            action=self._analyze_intent
        )
        
        knowledge_search_state = State(
            name="knowledge_search", 
            action=self._search_knowledge_base
        )
        
        sentiment_analysis_state = State(
            name="sentiment_analysis",
            action=self._analyze_sentiment
        )
        
        response_generation_state = State(
            name="response_generation",
            action=self._generate_response
        )
        
        escalation_check_state = State(
            name="escalation_check",
            action=self._check_escalation_need
        )
        
        human_handoff_state = State(
            name="human_handoff",
            action=self._initiate_human_handoff
        )
        
        conversation_update_state = State(
            name="conversation_update",
            action=self._update_conversation
        )
        
        resolution_state = State(
            name="resolution",
            state_type=StateType.FINAL,
            action=self._finalize_resolution
        )
        
        error_state = State(
            name="error_handling",
            state_type=StateType.ERROR,
            action=self._handle_error
        )
        
        # Add states to state machine
        state_machine.add_state(initial_state)
        state_machine.add_state(intent_analysis_state)
        state_machine.add_state(knowledge_search_state)
        state_machine.add_state(sentiment_analysis_state)
        state_machine.add_state(response_generation_state)
        state_machine.add_state(escalation_check_state)
        state_machine.add_state(human_handoff_state)
        state_machine.add_state(conversation_update_state)
        state_machine.add_state(resolution_state)
        state_machine.add_state(error_state)
        
        # Define transitions
        # Initial -> Intent Analysis
        state_machine.add_transition(Transition(
            from_state=initial_state,
            to_state=intent_analysis_state,
            condition=lambda ctx: ctx.get("customer_input") is not None
        ))
        
        # Intent Analysis -> Knowledge Search
        state_machine.add_transition(Transition(
            from_state=intent_analysis_state,
            to_state=knowledge_search_state,
            condition=lambda ctx: ctx.get("intent_confidence", 0) > 0.7
        ))
        
        # Intent Analysis -> Sentiment Analysis (low confidence)
        state_machine.add_transition(Transition(
            from_state=intent_analysis_state,
            to_state=sentiment_analysis_state,
            condition=lambda ctx: ctx.get("intent_confidence", 0) <= 0.7
        ))
        
        # Knowledge Search -> Response Generation
        state_machine.add_transition(Transition(
            from_state=knowledge_search_state,
            to_state=response_generation_state,
            condition=lambda ctx: len(ctx.get("knowledge_results", [])) > 0
        ))
        
        # Knowledge Search -> Escalation Check (no results)
        state_machine.add_transition(Transition(
            from_state=knowledge_search_state,
            to_state=escalation_check_state,
            condition=lambda ctx: len(ctx.get("knowledge_results", [])) == 0
        ))
        
        # Sentiment Analysis -> Escalation Check (negative sentiment)
        state_machine.add_transition(Transition(
            from_state=sentiment_analysis_state,
            to_state=escalation_check_state,
            condition=lambda ctx: ctx.get("sentiment") == "negative"
        ))
        
        # Sentiment Analysis -> Knowledge Search (neutral/positive)
        state_machine.add_transition(Transition(
            from_state=sentiment_analysis_state,
            to_state=knowledge_search_state,
            condition=lambda ctx: ctx.get("sentiment") in ["neutral", "positive"]
        ))
        
        # Response Generation -> Conversation Update
        state_machine.add_transition(Transition(
            from_state=response_generation_state,
            to_state=conversation_update_state
        ))
        
        # Escalation Check -> Human Handoff (needs escalation)
        state_machine.add_transition(Transition(
            from_state=escalation_check_state,
            to_state=human_handoff_state,
            condition=lambda ctx: ctx.get("needs_escalation", False)
        ))
        
        # Escalation Check -> Response Generation (no escalation)
        state_machine.add_transition(Transition(
            from_state=escalation_check_state,
            to_state=response_generation_state,
            condition=lambda ctx: not ctx.get("needs_escalation", False)
        ))
        
        # Human Handoff -> Resolution
        state_machine.add_transition(Transition(
            from_state=human_handoff_state,
            to_state=resolution_state
        ))
        
        # Conversation Update -> Resolution
        state_machine.add_transition(Transition(
            from_state=conversation_update_state,
            to_state=resolution_state,
            condition=lambda ctx: ctx.get("conversation_complete", False)
        ))
        
        # Conversation Update -> Intent Analysis (continue conversation)
        state_machine.add_transition(Transition(
            from_state=conversation_update_state,
            to_state=intent_analysis_state,
            condition=lambda ctx: not ctx.get("conversation_complete", False) and ctx.get("follow_up_input")
        ))
        
        # Error transitions (any state can go to error)
        for state in [intent_analysis_state, knowledge_search_state, sentiment_analysis_state, 
                     response_generation_state, escalation_check_state]:
            state_machine.add_transition(Transition(
                from_state=state,
                to_state=error_state,
                condition=lambda ctx: ctx.get("error_occurred", False),
                priority=10  # High priority
            ))
        
        # Create agent
        agent = Agent(
            name="CustomerSupportAgent",
            state_machine=state_machine,
            description="AI-powered customer support agent with escalation capabilities",
            max_retries=2,
            timeout=600  # 10 minutes
        )
        
        return agent
    
    async def _initialize_conversation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize conversation tracking."""
        tracker = ConversationTrackerTool()
        result = await tracker.execute({
            "action": "initialize",
            "customer_id": context.get("customer_id"),
            "channel": context.get("channel", "chat"),
            "initial_message": context.get("customer_input")
        })
        
        context.update(result)
        return {"status": "initialized", "conversation_id": result.get("conversation_id")}
    
    async def _analyze_intent(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze customer intent."""
        intent_tool = IntentClassificationTool()
        result = await intent_tool.execute({
            "text": context.get("customer_input", ""),
            "conversation_history": context.get("conversation_history", [])
        })
        
        context.update(result)
        return result
    
    async def _search_knowledge_base(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Search knowledge base for relevant information."""
        kb_tool = KnowledgeBaseTool()
        result = await kb_tool.execute({
            "query": context.get("customer_input"),
            "intent": context.get("intent"),
            "entities": context.get("entities", []),
            "top_k": 5
        })
        
        context.update(result)
        return result
    
    async def _analyze_sentiment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze customer sentiment."""
        sentiment_tool = SentimentAnalysisTool()
        result = await sentiment_tool.execute({
            "text": context.get("customer_input"),
            "conversation_history": context.get("conversation_history", [])
        })
        
        context.update(result)
        return result
    
    async def _generate_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate response."""
        response_tool = ResponseGeneratorTool()
        result = await response_tool.execute({
            "intent": context.get("intent"),
            "sentiment": context.get("sentiment"),
            "knowledge_results": context.get("knowledge_results", []),
            "customer_input": context.get("customer_input"),
            "conversation_context": context.get("conversation_history", [])
        })
        
        context.update(result)
        return result
    
    async def _check_escalation_need(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if escalation to human is needed."""
        escalation_tool = EscalationTool()
        result = await escalation_tool.execute({
            "intent": context.get("intent"),
            "sentiment": context.get("sentiment"), 
            "confidence": context.get("intent_confidence", 0),
            "knowledge_found": len(context.get("knowledge_results", [])) > 0,
            "conversation_length": len(context.get("conversation_history", [])),
            "customer_priority": context.get("customer_priority", "normal")
        })
        
        context.update(result)
        return result
    
    async def _initiate_human_handoff(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate handoff to human agent."""
        return {
            "handoff_initiated": True,
            "handoff_reason": context.get("escalation_reason"),
            "queue": context.get("escalation_queue", "general"),
            "priority": context.get("escalation_priority", "normal"),
            "handoff_message": "Connecting you with a human agent who can better assist you."
        }
    
    async def _update_conversation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update conversation with latest interaction."""
        tracker = ConversationTrackerTool()
        result = await tracker.execute({
            "action": "update",
            "conversation_id": context.get("conversation_id"),
            "agent_response": context.get("generated_response"),
            "intent": context.get("intent"),
            "sentiment": context.get("sentiment"),
            "resolution_status": context.get("resolution_status", "in_progress")
        })
        
        context.update(result)
        
        # Check if conversation should continue
        conversation_complete = (
            context.get("intent") == "goodbye" or 
            context.get("resolution_status") == "resolved" or
            not context.get("follow_up_input")
        )
        
        return {"conversation_complete": conversation_complete}
    
    async def _finalize_resolution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize conversation resolution."""
        tracker = ConversationTrackerTool()
        result = await tracker.execute({
            "action": "finalize",
            "conversation_id": context.get("conversation_id"),
            "resolution_status": "resolved",
            "satisfaction_score": context.get("satisfaction_score"),
            "resolution_method": "automated" if not context.get("handoff_initiated") else "human_handoff"
        })
        
        return {
            "final_response": context.get("generated_response", "Thank you for contacting support!"),
            "resolution_summary": result.get("summary"),
            "conversation_complete": True
        }
    
    async def _handle_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors in the workflow."""
        error_message = context.get("error_message", "An unexpected error occurred.")
        
        return {
            "error_handled": True,
            "fallback_response": "I apologize, but I'm experiencing technical difficulties. Let me connect you with a human agent.",
            "needs_escalation": True,
            "escalation_reason": "system_error"
        }
    
    async def handle_customer_interaction(
        self,
        customer_input: str,
        customer_id: str = None,
        channel: str = "chat",
        conversation_id: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle a single customer interaction."""
        
        # Prepare execution context
        execution_context = {
            "customer_input": customer_input,
            "customer_id": customer_id or "anonymous",
            "channel": channel,
            "conversation_id": conversation_id
        }
        
        if context:
            execution_context.update(context)
        
        # Execute agent
        print(f"Processing customer input: {customer_input[:50]}...")
        result = await self.agent.execute(execution_context)
        
        if result.status == "completed":
            sm_result = result.state_machine_result
            final_response = sm_result.get("final_response", "Thank you for contacting support!")
            
            return {
                "status": "success",
                "response": final_response,
                "conversation_id": sm_result.get("conversation_id"),
                "intent": sm_result.get("intent"),
                "sentiment": sm_result.get("sentiment"),
                "needs_human": sm_result.get("handoff_initiated", False),
                "resolution_status": sm_result.get("resolution_status", "resolved"),
                "execution_time": result.execution_time,
                "execution_path": sm_result.get("execution_path", [])
            }
        else:
            return {
                "status": "error",
                "error": result.error,
                "fallback_response": "I apologize, but I'm unable to process your request right now. Please try again later or contact human support.",
                "execution_time": result.execution_time
            }
    
    async def handle_conversation(
        self,
        messages: List[Dict[str, str]],
        customer_id: str = None,
        channel: str = "chat"
    ) -> List[Dict[str, Any]]:
        """Handle a full conversation with multiple messages."""
        
        conversation_results = []
        conversation_id = None
        conversation_context = {}
        
        for i, message in enumerate(messages):
            customer_input = message.get("content", "")
            
            # Build conversation context
            if i > 0:
                conversation_context["conversation_history"] = [
                    r.get("response", "") for r in conversation_results
                ]
                conversation_context["follow_up_input"] = customer_input
            
            result = await self.handle_customer_interaction(
                customer_input=customer_input,
                customer_id=customer_id,
                channel=channel,
                conversation_id=conversation_id,
                context=conversation_context
            )
            
            conversation_results.append(result)
            
            # Update conversation ID for subsequent messages
            if not conversation_id:
                conversation_id = result.get("conversation_id")
            
            # Break if human handoff is needed
            if result.get("needs_human"):
                break
        
        return conversation_results


async def main():
    """Main function for running the customer support agent."""
    parser = argparse.ArgumentParser(description="Customer Support Agent")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--test-input",
        help="Single test input message"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create agent
    agent = CustomerSupportAgent(config)
    
    if args.interactive:
        # Interactive mode
        print("Customer Support Agent - Interactive Mode")
        print("Type 'quit' to exit")
        print("-" * 40)
        
        customer_id = input("Enter customer ID (optional): ").strip() or None
        conversation_id = None
        
        while True:
            user_input = input("\nCustomer: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            if not user_input:
                continue
            
            try:
                result = await agent.handle_customer_interaction(
                    customer_input=user_input,
                    customer_id=customer_id,
                    conversation_id=conversation_id
                )
                
                print(f"Agent: {result['response']}")
                
                if result.get('needs_human'):
                    print("[System: Escalating to human agent]")
                    break
                
                conversation_id = result.get('conversation_id')
                
            except Exception as e:
                print(f"Error: {str(e)}")
    
    elif args.test_input:
        # Single test input
        try:
            result = await agent.handle_customer_interaction(
                customer_input=args.test_input,
                customer_id="test_user"
            )
            
            print("\n" + "="*50)
            print("SUPPORT INTERACTION RESULT")
            print("="*50)
            print(f"Status: {result['status']}")
            print(f"Response: {result['response']}")
            print(f"Intent: {result.get('intent', 'unknown')}")
            print(f"Sentiment: {result.get('sentiment', 'unknown')}")
            print(f"Needs Human: {result.get('needs_human', False)}")
            print(f"Execution Time: {result['execution_time']:.2f}s")
            
        except Exception as e:
            print(f"Error processing test input: {str(e)}")
            return 1
    
    else:
        # Example conversation
        example_messages = [
            {"content": "Hi, I'm having trouble with my account login"},
            {"content": "I keep getting an error message saying invalid credentials"},
            {"content": "I've tried resetting my password but it's still not working"}
        ]
        
        print("Running example conversation...")
        results = await agent.handle_conversation(
            messages=example_messages,
            customer_id="example_user"
        )
        
        print("\n" + "="*50)
        print("CONVERSATION RESULTS")
        print("="*50)
        
        for i, result in enumerate(results):
            print(f"\nInteraction {i+1}:")
            print(f"  Status: {result['status']}")
            print(f"  Response: {result['response']}")
            print(f"  Intent: {result.get('intent', 'unknown')}")
            print(f"  Sentiment: {result.get('sentiment', 'unknown')}")
            
            if result.get('needs_human'):
                print("  [ESCALATED TO HUMAN]")
                break
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
