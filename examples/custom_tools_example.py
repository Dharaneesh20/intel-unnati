"""
Example: Creating custom tools for your specific use case.
"""

import asyncio
from typing import Dict, Any


class EmailAnalyzerTool:
    """Tool for analyzing email content."""
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        email_content = inputs.get("email_content", "")
        
        # Mock analysis logic (replace with real ML models)
        analysis = {
            "sentiment": "positive" if "good" in email_content.lower() else "neutral",
            "urgency": "high" if any(word in email_content.lower() 
                                  for word in ["urgent", "asap", "emergency"]) else "normal",
            "category": "support_request",  # Could use ML classification here
            "confidence": 0.85
        }
        
        return {
            "email_analysis": analysis,
            "processed": True
        }


class DatabaseLookupTool:
    """Tool for looking up information in a database."""
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_id = inputs.get("user_id", "")
        
        # Mock database lookup (replace with real database queries)
        user_data = {
            "user_id": user_id,
            "account_type": "premium",
            "last_contact": "2024-01-15",
            "issues_count": 2,
            "satisfaction_score": 4.2
        }
        
        return {
            "user_profile": user_data,
            "lookup_successful": True
        }


class ResponseGeneratorTool:
    """Tool for generating responses based on analysis."""
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        analysis = inputs.get("email_analysis", {})
        user_profile = inputs.get("user_profile", {})
        
        # Generate response based on context
        if analysis.get("urgency") == "high":
            response = "Thank you for contacting us. We understand this is urgent and will prioritize your request."
        elif user_profile.get("account_type") == "premium":
            response = "Thank you for being a valued premium customer. We'll handle your request promptly."
        else:
            response = "Thank you for contacting us. We'll get back to you soon."
        
        return {
            "generated_response": response,
            "recommended_action": "send_immediate_reply" if analysis.get("urgency") == "high" else "queue_for_agent"
        }


# Example usage
async def email_processing_workflow():
    """Example of a complete email processing workflow."""
    
    # Initialize tools
    email_analyzer = EmailAnalyzerTool()
    db_lookup = DatabaseLookupTool() 
    response_generator = ResponseGeneratorTool()
    
    # Input data
    inputs = {
        "email_content": "This is urgent! I need help with my account ASAP.",
        "user_id": "customer_123",
        "sender_email": "customer@example.com"
    }
    
    print("📧 Processing email workflow...")
    
    # Step 1: Analyze email
    analysis_result = await email_analyzer.execute(inputs)
    inputs.update(analysis_result)
    
    # Step 2: Lookup user profile
    lookup_result = await db_lookup.execute(inputs)
    inputs.update(lookup_result)
    
    # Step 3: Generate response
    response_result = await response_generator.execute(inputs)
    inputs.update(response_result)
    
    print("✅ Email processing completed!")
    print(f"Analysis: {analysis_result['email_analysis']}")
    print(f"User Profile: {lookup_result['user_profile']}")
    print(f"Generated Response: {response_result['generated_response']}")
    print(f"Recommended Action: {response_result['recommended_action']}")
    
    return inputs


if __name__ == "__main__":
    asyncio.run(email_processing_workflow())
