"""
Customer support tools for the AI Agent Framework.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import random


class IntentClassificationTool:
    """Tool for classifying customer intent."""
    
    def __init__(self, model: str = "intent_classifier"):
        self.model = model
        self.intent_categories = [
            "account_issue", "billing_inquiry", "technical_support", 
            "product_question", "complaint", "compliment", "goodbye", "general_inquiry"
        ]
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute intent classification."""
        text = inputs.get("text", "")
        conversation_history = inputs.get("conversation_history", [])
        
        # Simple rule-based intent classification (placeholder)
        intent = self._classify_intent(text)
        confidence = self._calculate_confidence(text, intent)
        entities = self._extract_entities(text)
        
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "entities": entities,
            "original_text": text
        }
    
    def _classify_intent(self, text: str) -> str:
        """Classify intent based on keywords."""
        text_lower = text.lower()
        
        # Account issues
        if any(word in text_lower for word in ["login", "password", "account", "locked", "access"]):
            return "account_issue"
        
        # Billing
        elif any(word in text_lower for word in ["bill", "charge", "payment", "refund", "invoice"]):
            return "billing_inquiry"
        
        # Technical support
        elif any(word in text_lower for word in ["error", "bug", "not working", "broken", "technical"]):
            return "technical_support"
        
        # Product questions
        elif any(word in text_lower for word in ["how to", "feature", "product", "functionality"]):
            return "product_question"
        
        # Complaints
        elif any(word in text_lower for word in ["complaint", "problem", "issue", "frustrated", "angry"]):
            return "complaint"
        
        # Compliments
        elif any(word in text_lower for word in ["thank", "great", "excellent", "good job", "appreciate"]):
            return "compliment"
        
        # Goodbye
        elif any(word in text_lower for word in ["bye", "goodbye", "thanks", "done", "resolved"]):
            return "goodbye"
        
        else:
            return "general_inquiry"
    
    def _calculate_confidence(self, text: str, intent: str) -> float:
        """Calculate confidence score."""
        # Simple confidence calculation based on keyword matches
        text_lower = text.lower()
        
        intent_keywords = {
            "account_issue": ["login", "password", "account", "locked", "access"],
            "billing_inquiry": ["bill", "charge", "payment", "refund", "invoice"],
            "technical_support": ["error", "bug", "not working", "broken", "technical"],
            "product_question": ["how to", "feature", "product", "functionality"],
            "complaint": ["complaint", "problem", "issue", "frustrated", "angry"],
            "compliment": ["thank", "great", "excellent", "good job", "appreciate"],
            "goodbye": ["bye", "goodbye", "thanks", "done", "resolved"],
            "general_inquiry": []
        }
        
        keywords = intent_keywords.get(intent, [])
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        if not keywords:  # general_inquiry
            return 0.5
        
        confidence = min(0.9, 0.6 + (matches / len(keywords)) * 0.3)
        return confidence
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        import re
        
        entities = []
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        for email in emails:
            entities.append({"type": "email", "value": email})
        
        # Extract phone numbers (simple pattern)
        phones = re.findall(r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b', text)
        for phone in phones:
            entities.append({"type": "phone", "value": phone})
        
        # Extract order numbers (pattern: letters followed by numbers)
        orders = re.findall(r'\b[A-Z]{2,}\d{4,}\b', text)
        for order in orders:
            entities.append({"type": "order_number", "value": order})
        
        return entities


class KnowledgeBaseTool:
    """Tool for searching knowledge base."""
    
    def __init__(self):
        # Mock knowledge base
        self.knowledge_base = [
            {
                "id": "kb001",
                "title": "Login Issues - Password Reset",
                "content": "To reset your password: 1. Go to login page 2. Click 'Forgot Password' 3. Enter your email 4. Check your email for reset link",
                "category": "account_issue",
                "keywords": ["login", "password", "reset", "forgot"]
            },
            {
                "id": "kb002", 
                "title": "Billing - Understanding Your Invoice",
                "content": "Your monthly invoice includes: base subscription fee, usage charges, taxes, and any additional services. You can view detailed breakdown in your account portal.",
                "category": "billing_inquiry",
                "keywords": ["bill", "invoice", "charges", "payment"]
            },
            {
                "id": "kb003",
                "title": "Technical Support - Common Error Messages",
                "content": "Common errors and solutions: 'Connection timeout' - check internet connection, 'Invalid session' - please log out and log back in, 'Server error' - try again in a few minutes.",
                "category": "technical_support",
                "keywords": ["error", "technical", "connection", "session"]
            },
            {
                "id": "kb004",
                "title": "Product Features - Getting Started Guide",
                "content": "Welcome to our platform! Key features include: dashboard for overview, analytics for insights, settings for customization, and support for help.",
                "category": "product_question",
                "keywords": ["features", "getting started", "how to", "guide"]
            }
        ]
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute knowledge base search."""
        query = inputs.get("query", "")
        intent = inputs.get("intent", "")
        entities = inputs.get("entities", [])
        top_k = inputs.get("top_k", 3)
        
        # Search knowledge base
        results = self._search_knowledge_base(query, intent, top_k)
        
        return {
            "knowledge_results": results,
            "result_count": len(results),
            "search_query": query
        }
    
    def _search_knowledge_base(self, query: str, intent: str, top_k: int) -> List[Dict[str, Any]]:
        """Search knowledge base with scoring."""
        query_lower = query.lower()
        scored_results = []
        
        for article in self.knowledge_base:
            score = 0
            
            # Category match bonus
            if article["category"] == intent:
                score += 0.5
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in article["keywords"] if keyword in query_lower)
            score += keyword_matches * 0.2
            
            # Title matching
            if any(word in article["title"].lower() for word in query_lower.split()):
                score += 0.3
            
            # Content matching
            content_matches = sum(1 for word in query_lower.split() if word in article["content"].lower())
            score += content_matches * 0.1
            
            if score > 0:
                scored_results.append({
                    "article": article,
                    "relevance_score": score
                })
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return [result["article"] for result in scored_results[:top_k]]


class SentimentAnalysisTool:
    """Tool for analyzing customer sentiment."""
    
    def __init__(self):
        self.positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic", 
            "pleased", "happy", "satisfied", "love", "perfect", "awesome"
        ]
        self.negative_words = [
            "bad", "terrible", "awful", "horrible", "hate", "angry", "frustrated",
            "disappointed", "upset", "annoyed", "problem", "issue", "broken"
        ]
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sentiment analysis."""
        text = inputs.get("text", "")
        conversation_history = inputs.get("conversation_history", [])
        
        # Analyze current message sentiment
        current_sentiment = self._analyze_sentiment(text)
        
        # Analyze overall conversation sentiment
        overall_text = " ".join(conversation_history + [text])
        overall_sentiment = self._analyze_sentiment(overall_text)
        
        # Calculate sentiment scores
        sentiment_score = self._calculate_sentiment_score(text)
        
        return {
            "sentiment": current_sentiment,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": sentiment_score,
            "confidence": abs(sentiment_score)
        }
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text."""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative" 
        else:
            return "neutral"
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score (-1 to 1)."""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        total_words = len(text_lower.split())
        
        if total_words == 0:
            return 0.0
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        # Score between -1 and 1
        score = positive_ratio - negative_ratio
        return max(-1.0, min(1.0, score * 5))  # Amplify the score


class ResponseGeneratorTool:
    """Tool for generating appropriate responses."""
    
    def __init__(self):
        self.response_templates = {
            "account_issue": [
                "I understand you're having trouble with your account. Let me help you with that. {solution}",
                "I can help you resolve this account issue. {solution}",
                "Account problems can be frustrating. Here's how we can fix this: {solution}"
            ],
            "billing_inquiry": [
                "I'd be happy to help you with your billing question. {solution}",
                "Let me explain your billing details. {solution}",
                "I can clarify that billing information for you. {solution}"
            ],
            "technical_support": [
                "I see you're experiencing a technical issue. Let's get this resolved. {solution}",
                "Technical problems can be frustrating. Here's what we can do: {solution}",
                "I'll help you troubleshoot this technical issue. {solution}"
            ],
            "product_question": [
                "I'm happy to explain that feature for you. {solution}",
                "Great question about our product! {solution}",
                "Let me walk you through that functionality. {solution}"
            ],
            "complaint": [
                "I sincerely apologize for this experience. Let me make this right. {solution}",
                "I understand your frustration and I'm here to help resolve this. {solution}",
                "I'm sorry this happened. Here's how we'll fix it: {solution}"
            ],
            "compliment": [
                "Thank you so much for your kind words! We really appreciate your feedback.",
                "That means a lot to us! Thank you for taking the time to share your positive experience.",
                "We're so glad you're happy with our service! Your feedback motivates our team."
            ],
            "goodbye": [
                "Thank you for contacting us! Have a great day!",
                "You're welcome! Feel free to reach out if you need any further assistance.",
                "Glad I could help! Take care and have a wonderful day!"
            ],
            "general_inquiry": [
                "I'm here to help! {solution}",
                "I'd be happy to assist you with that. {solution}",
                "Let me help you with your question. {solution}"
            ]
        }
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute response generation."""
        intent = inputs.get("intent", "general_inquiry")
        sentiment = inputs.get("sentiment", "neutral")
        knowledge_results = inputs.get("knowledge_results", [])
        customer_input = inputs.get("customer_input", "")
        conversation_context = inputs.get("conversation_context", [])
        
        # Generate response
        response = self._generate_response(intent, sentiment, knowledge_results)
        
        # Add sentiment-appropriate tone
        response = self._adjust_tone_for_sentiment(response, sentiment)
        
        return {
            "generated_response": response,
            "response_intent": intent,
            "includes_solution": len(knowledge_results) > 0
        }
    
    def _generate_response(self, intent: str, sentiment: str, knowledge_results: List[Dict]) -> str:
        """Generate response based on intent and knowledge."""
        templates = self.response_templates.get(intent, self.response_templates["general_inquiry"])
        template = random.choice(templates)
        
        # If we have knowledge base results, include the solution
        if knowledge_results and "{solution}" in template:
            primary_solution = knowledge_results[0]["content"]
            response = template.format(solution=primary_solution)
        elif "{solution}" in template:
            response = template.format(solution="I'll look into this for you and get back with more information.")
        else:
            response = template
        
        return response
    
    def _adjust_tone_for_sentiment(self, response: str, sentiment: str) -> str:
        """Adjust response tone based on customer sentiment."""
        if sentiment == "negative":
            # Add empathy for negative sentiment
            empathy_phrases = [
                "I completely understand your frustration. ",
                "I can see why this would be concerning. ",
                "I apologize for any inconvenience this has caused. "
            ]
            empathy = random.choice(empathy_phrases)
            return empathy + response
        
        elif sentiment == "positive":
            # Add enthusiasm for positive sentiment
            enthusiasm_phrases = [
                "I'm so glad you reached out! ",
                "It's my pleasure to help! ",
                "I'm happy to assist you with this! "
            ]
            enthusiasm = random.choice(enthusiasm_phrases)
            return enthusiasm + response
        
        return response


class EscalationTool:
    """Tool for determining if escalation to human is needed."""
    
    def __init__(self):
        self.escalation_rules = {
            "high_priority_intents": ["complaint", "billing_inquiry"],
            "negative_sentiment_threshold": -0.5,
            "low_confidence_threshold": 0.3,
            "max_conversation_length": 5
        }
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute escalation decision logic."""
        intent = inputs.get("intent", "")
        sentiment = inputs.get("sentiment", "neutral")
        confidence = inputs.get("confidence", 1.0)
        knowledge_found = inputs.get("knowledge_found", True)
        conversation_length = inputs.get("conversation_length", 0)
        customer_priority = inputs.get("customer_priority", "normal")
        
        # Determine if escalation is needed
        needs_escalation, reason, priority, queue = self._evaluate_escalation(
            intent, sentiment, confidence, knowledge_found, 
            conversation_length, customer_priority
        )
        
        return {
            "needs_escalation": needs_escalation,
            "escalation_reason": reason,
            "escalation_priority": priority,
            "escalation_queue": queue
        }
    
    def _evaluate_escalation(
        self, intent: str, sentiment: str, confidence: float, 
        knowledge_found: bool, conversation_length: int, customer_priority: str
    ) -> tuple:
        """Evaluate if escalation is needed."""
        
        reasons = []
        priority = "normal"
        queue = "general"
        
        # High priority intents
        if intent in self.escalation_rules["high_priority_intents"]:
            reasons.append(f"high_priority_intent_{intent}")
            if intent == "complaint":
                priority = "high"
                queue = "complaints"
            elif intent == "billing_inquiry":
                queue = "billing"
        
        # Negative sentiment
        if sentiment == "negative":
            reasons.append("negative_sentiment")
            if priority == "normal":
                priority = "medium"
        
        # Low confidence in intent classification
        if confidence < self.escalation_rules["low_confidence_threshold"]:
            reasons.append("low_confidence")
        
        # No knowledge base results found
        if not knowledge_found:
            reasons.append("no_solution_found")
        
        # Long conversation without resolution
        if conversation_length >= self.escalation_rules["max_conversation_length"]:
            reasons.append("conversation_too_long")
            priority = "high"
        
        # Customer priority
        if customer_priority == "vip":
            reasons.append("vip_customer")
            priority = "high"
            queue = "vip"
        
        # Determine if escalation is needed
        needs_escalation = len(reasons) > 0 and (
            intent in self.escalation_rules["high_priority_intents"] or
            sentiment == "negative" or
            not knowledge_found or
            confidence < self.escalation_rules["low_confidence_threshold"] or
            conversation_length >= self.escalation_rules["max_conversation_length"] or
            customer_priority == "vip"
        )
        
        escalation_reason = ", ".join(reasons) if reasons else None
        
        return needs_escalation, escalation_reason, priority, queue


class ConversationTrackerTool:
    """Tool for tracking conversation state and history."""
    
    def __init__(self):
        self.conversations = {}  # In-memory storage (would be database in production)
        
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conversation tracking action."""
        action = inputs.get("action", "")
        
        if action == "initialize":
            return await self._initialize_conversation(inputs)
        elif action == "update":
            return await self._update_conversation(inputs)
        elif action == "finalize":
            return await self._finalize_conversation(inputs)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _initialize_conversation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize a new conversation."""
        conversation_id = str(uuid.uuid4())
        
        conversation = {
            "id": conversation_id,
            "customer_id": inputs.get("customer_id"),
            "channel": inputs.get("channel", "chat"),
            "started_at": datetime.now().isoformat(),
            "messages": [],
            "status": "active",
            "initial_message": inputs.get("initial_message", "")
        }
        
        # Add initial message if provided
        if conversation["initial_message"]:
            conversation["messages"].append({
                "timestamp": datetime.now().isoformat(),
                "sender": "customer",
                "content": conversation["initial_message"],
                "type": "text"
            })
        
        self.conversations[conversation_id] = conversation
        
        return {
            "conversation_id": conversation_id,
            "status": "initialized"
        }
    
    async def _update_conversation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Update conversation with new interaction."""
        conversation_id = inputs.get("conversation_id")
        
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        
        # Add agent response
        agent_response = inputs.get("agent_response")
        if agent_response:
            conversation["messages"].append({
                "timestamp": datetime.now().isoformat(),
                "sender": "agent",
                "content": agent_response,
                "type": "text",
                "intent": inputs.get("intent"),
                "sentiment": inputs.get("sentiment")
            })
        
        # Update conversation status
        conversation["status"] = inputs.get("resolution_status", "active")
        conversation["last_updated"] = datetime.now().isoformat()
        
        return {
            "conversation_updated": True,
            "message_count": len(conversation["messages"]),
            "conversation_history": [msg["content"] for msg in conversation["messages"]]
        }
    
    async def _finalize_conversation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize conversation."""
        conversation_id = inputs.get("conversation_id")
        
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        
        # Update final status
        conversation["status"] = inputs.get("resolution_status", "resolved")
        conversation["ended_at"] = datetime.now().isoformat()
        conversation["satisfaction_score"] = inputs.get("satisfaction_score")
        conversation["resolution_method"] = inputs.get("resolution_method", "automated")
        
        # Calculate conversation metrics
        duration = (
            datetime.fromisoformat(conversation["ended_at"]) - 
            datetime.fromisoformat(conversation["started_at"])
        ).total_seconds()
        
        message_count = len(conversation["messages"])
        
        summary = {
            "conversation_id": conversation_id,
            "duration_seconds": duration,
            "message_count": message_count,
            "resolution_status": conversation["status"],
            "resolution_method": conversation["resolution_method"],
            "satisfaction_score": conversation.get("satisfaction_score")
        }
        
        return {
            "conversation_finalized": True,
            "summary": summary
        }
