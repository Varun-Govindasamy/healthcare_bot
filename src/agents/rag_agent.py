"""
RAG Agent - Retrieval Augmented Generation for healthcare knowledge.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from crewai import Agent, Task
import json

from ..services.pinecone_service import pinecone_service
from ..config.settings import settings

logger = logging.getLogger(__name__)


def healthcare_knowledge_retriever(query: str, user_id: Optional[str] = None, top_k: int = 5) -> str:
    """Retrieves relevant healthcare information from medical knowledge base and user documents."""
    try:
        # This would be implemented with actual Pinecone search
        results = f"Retrieved {top_k} relevant documents for query: {query}"
        if user_id:
            results += f" for user: {user_id}"
        return results
    except Exception as e:
        return f"Error retrieving information: {str(e)}"


def user_document_retriever(query: str, user_id: str, top_k: int = 3) -> str:
    """Retrieves information from user's uploaded medical documents and reports."""
    try:
        # This would search user's document namespace in Pinecone
        return f"Retrieved user documents for {user_id} matching: {query}"
    except Exception as e:
        return f"Error retrieving user documents: {str(e)}"


class RAGAgent:
    """Retrieval Augmented Generation agent for healthcare knowledge."""
    
    def __init__(self):
        self.tools = []
        
        self.agent = Agent(
            role="Healthcare Knowledge Specialist",
            goal="Provide accurate medical information by retrieving relevant knowledge from healthcare databases and user documents",
            backstory="""You are a medical knowledge retrieval specialist with access to comprehensive 
            healthcare databases including WHO guidelines, medical literature, and user-specific medical documents. 
            Your role is to find the most relevant and accurate medical information to support healthcare 
            recommendations and answer medical queries.""",
            tools=self.tools,
            verbose=True,
            allow_delegation=False
        )
    
    def create_knowledge_retrieval_task(self, query: str, user_id: Optional[str] = None) -> Task:
        """Create task for retrieving general healthcare knowledge."""
        return Task(
            description=f"""
            Retrieve relevant healthcare information for the following medical query:
            Query: "{query}"
            User ID: {user_id or 'General'}
            
            Search the healthcare knowledge base for:
            1. Medical conditions and symptoms related to the query
            2. Treatment options and recommendations
            3. Medication information and interactions
            4. Prevention and lifestyle advice
            5. When to seek medical attention
            
            Focus on evidence-based medical information from reliable sources.
            If user ID is provided, also consider their medical profile context.
            
            Return the most relevant information with source references.
            """,
            agent=self.agent,
            expected_output="Relevant healthcare information with source citations for the medical query"
        )
    
    def create_user_document_task(self, query: str, user_id: str) -> Task:
        """Create task for retrieving user-specific document information."""
        return Task(
            description=f"""
            Search user's uploaded medical documents for information related to:
            Query: "{query}"
            User ID: {user_id}
            
            Look for:
            1. Previous test results and lab values
            2. Prescribed medications and dosages
            3. Medical history and diagnoses
            4. Doctor's recommendations and follow-ups
            5. Allergies and contraindications
            
            Cross-reference with the current query to provide personalized insights.
            Highlight any relevant patterns or changes in the user's medical history.
            """,
            agent=self.agent,
            expected_output="Relevant information from user's medical documents with specific details and context"
        )
    
    async def get_healthcare_knowledge(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get relevant healthcare knowledge for a query."""
        try:
            # Search general healthcare knowledge
            general_results = await pinecone_service.search_healthcare_knowledge(
                query=query,
                top_k=5
            )
            
            user_results = []
            if user_id:
                # Search user-specific documents
                user_results = await pinecone_service.search_user_documents(
                    query=query,
                    user_id=user_id,
                    top_k=3
                )
            
            return {
                "general_knowledge": general_results,
                "user_specific": user_results,
                "query": query,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error retrieving healthcare knowledge: {e}")
            return {
                "general_knowledge": [],
                "user_specific": [],
                "query": query,
                "user_id": user_id,
                "error": str(e)
            }
    
    async def get_similar_cases(self, symptoms: List[str], user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get similar medical cases based on symptoms."""
        try:
            # Create query from symptoms
            symptoms_query = " ".join(symptoms)
            
            # Search for similar cases
            results = await pinecone_service.search_healthcare_knowledge(
                query=f"symptoms: {symptoms_query}",
                top_k=10
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")
            return []
    
    async def get_medication_interactions(self, medications: List[str]) -> Dict[str, Any]:
        """Check for medication interactions and contraindications."""
        try:
            # Create query for medication interactions
            med_query = " ".join(medications)
            
            results = await pinecone_service.search_healthcare_knowledge(
                query=f"medication interactions contraindications {med_query}",
                top_k=5
            )
            
            return {
                "medications": medications,
                "interactions": results,
                "safety_warnings": []
            }
            
        except Exception as e:
            logger.error(f"Error checking medication interactions: {e}")
            return {
                "medications": medications,
                "interactions": [],
                "safety_warnings": [],
                "error": str(e)
            }
    
    async def get_condition_guidance(self, condition: str, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get comprehensive guidance for a medical condition."""
        try:
            # Build query with user context
            query = f"medical condition {condition}"
            if user_profile:
                age = user_profile.get('age')
                gender = user_profile.get('gender')
                existing_conditions = user_profile.get('existing_conditions', [])
                
                if age:
                    query += f" age {age}"
                if gender:
                    query += f" {gender}"
                if existing_conditions:
                    query += f" comorbidities {' '.join(existing_conditions)}"
            
            results = await pinecone_service.search_healthcare_knowledge(
                query=query,
                top_k=8
            )
            
            return {
                "condition": condition,
                "guidance": results,
                "personalized": bool(user_profile)
            }
            
        except Exception as e:
            logger.error(f"Error getting condition guidance: {e}")
            return {
                "condition": condition,
                "guidance": [],
                "personalized": False,
                "error": str(e)
            }


# Global RAG agent instance
rag_agent = RAGAgent()