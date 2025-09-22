"""
Pinecone vector database service for healthcare knowledge and user documents.
"""
import logging
from typing import List, Dict, Any, Optional
import pinecone
from openai import AsyncOpenAI
import hashlib
import json
from datetime import datetime

from ..config.settings import settings

logger = logging.getLogger(__name__)


class PineconeService:
    """Service for managing Pinecone vector database operations."""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.index = None
        self.embedding_model = "text-embedding-3-large"
        self.dimension = 3072  # Dimension for text-embedding-3-large
        
        # Namespace configurations
        self.healthcare_namespace = "healthcare_knowledge"
        self.user_documents_namespace = "user_documents"
    
    async def initialize(self):
        """Initialize Pinecone connection and index."""
        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
            
            # Check if index exists, create if not
            if settings.pinecone_index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=settings.pinecone_index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    metadata_config={
                        "indexed": ["document_type", "user_id", "source", "date"]
                    }
                )
                logger.info(f"Created Pinecone index: {settings.pinecone_index_name}")
            
            # Connect to index
            self.index = pinecone.Index(settings.pinecone_index_name)
            logger.info("Successfully connected to Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []
    
    async def upsert_healthcare_knowledge(self, documents: List[Dict[str, Any]]) -> bool:
        """Upsert healthcare knowledge documents."""
        try:
            vectors = []
            
            for doc in documents:
                # Generate embedding
                text_content = doc.get("content", "")
                if not text_content:
                    continue
                
                embedding = await self.generate_embedding(text_content)
                if not embedding:
                    continue
                
                # Create vector ID
                vector_id = hashlib.md5(text_content.encode()).hexdigest()
                
                # Prepare metadata
                metadata = {
                    "document_type": doc.get("type", "general"),
                    "source": doc.get("source", "unknown"),
                    "title": doc.get("title", "")[:512],  # Pinecone metadata limit
                    "date": doc.get("date", datetime.utcnow().isoformat()),
                    "content": text_content[:8192]  # Store first 8k chars in metadata
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=self.healthcare_namespace
                )
            
            logger.info(f"Upserted {len(vectors)} healthcare documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert healthcare knowledge: {e}")
            return False
    
    async def upsert_user_document(self, user_id: str, document_content: str, 
                                  document_type: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Upsert user-specific document."""
        try:
            # Generate embedding
            embedding = await self.generate_embedding(document_content)
            if not embedding:
                return False
            
            # Create vector ID
            vector_id = f"{user_id}_{hashlib.md5(document_content.encode()).hexdigest()}"
            
            # Prepare metadata
            doc_metadata = {
                "user_id": user_id,
                "document_type": document_type,
                "date": datetime.utcnow().isoformat(),
                "content": document_content[:8192]
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            # Upsert vector
            self.index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": doc_metadata
                }],
                namespace=f"{self.user_documents_namespace}_{user_id}"
            )
            
            logger.info(f"Upserted user document for {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert user document: {e}")
            return False
    
    async def search_healthcare_knowledge(self, query: str, top_k: int = 5, 
                                        filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search healthcare knowledge base."""
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            if not query_embedding:
                return []
            
            # Perform search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.healthcare_namespace,
                filter=filter_metadata,
                include_metadata=True
            )
            
            # Format results
            results = []
            for match in search_results.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "content": match.metadata.get("content", ""),
                    "source": match.metadata.get("source", ""),
                    "title": match.metadata.get("title", ""),
                    "document_type": match.metadata.get("document_type", ""),
                    "date": match.metadata.get("date", "")
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} healthcare knowledge results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search healthcare knowledge: {e}")
            return []
    
    async def search_user_documents(self, query: str, user_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search user-specific documents."""
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            if not query_embedding:
                return []
            
            # Perform search in user namespace
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=f"{self.user_documents_namespace}_{user_id}",
                include_metadata=True
            )
            
            # Format results
            results = []
            for match in search_results.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "content": match.metadata.get("content", ""),
                    "document_type": match.metadata.get("document_type", ""),
                    "date": match.metadata.get("date", "")
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} user document results for {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search user documents: {e}")
            return []
    
    async def delete_user_documents(self, user_id: str) -> bool:
        """Delete all documents for a user."""
        try:
            # Delete entire user namespace
            self.index.delete(delete_all=True, namespace=f"{self.user_documents_namespace}_{user_id}")
            logger.info(f"Deleted all documents for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user documents: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "namespaces": stats.namespaces,
                "dimension": stats.dimension
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    async def initialize_default_healthcare_knowledge(self):
        """Initialize with default healthcare knowledge base."""
        try:
            # Default healthcare documents
            default_docs = [
                {
                    "content": """
                    Fever Management:
                    - Normal body temperature: 98.6°F (37°C)
                    - Fever: Temperature above 100.4°F (38°C)
                    - Treatment: Paracetamol 500mg every 6-8 hours, plenty of fluids, rest
                    - See doctor if: Fever above 103°F, persistent for >3 days, with severe symptoms
                    - For children: Use pediatric doses, consult pediatrician
                    """,
                    "type": "treatment_guideline",
                    "source": "Medical Guidelines",
                    "title": "Fever Management Protocol"
                },
                {
                    "content": """
                    Dengue Prevention and Management:
                    - Symptoms: High fever, severe headache, eye pain, muscle aches, rash
                    - Prevention: Remove stagnant water, use mosquito nets, wear full sleeves
                    - Treatment: Paracetamol for fever, NO aspirin, increase fluid intake
                    - Warning signs: Severe abdominal pain, persistent vomiting, bleeding
                    - Seek immediate help if: Platelet count drops, severe dehydration
                    """,
                    "type": "disease_guideline",
                    "source": "WHO Guidelines",
                    "title": "Dengue Prevention and Treatment"
                },
                {
                    "content": """
                    Diabetes Management:
                    - Normal blood sugar: 80-130 mg/dL (fasting), <180 mg/dL (after meals)
                    - Diet: Low carb, high fiber, regular meal times
                    - Exercise: 30 minutes daily walking, yoga
                    - Medication: Take as prescribed, monitor blood sugar
                    - Complications: Check feet daily, eye exams, kidney function tests
                    - Emergency: If blood sugar <70 or >300, seek immediate help
                    """,
                    "type": "chronic_condition",
                    "source": "Diabetes Association",
                    "title": "Diabetes Daily Management"
                },
                {
                    "content": """
                    Skin Rash and Allergies:
                    - Common causes: Contact dermatitis, food allergies, insect bites
                    - Treatment: Cool compress, calamine lotion, antihistamines
                    - Avoid: Scratching, harsh soaps, known allergens
                    - See doctor if: Widespread rash, difficulty breathing, swelling
                    - Allergy prevention: Identify triggers, carry antihistamines
                    """,
                    "type": "dermatology",
                    "source": "Dermatology Guidelines",
                    "title": "Common Skin Conditions"
                },
                {
                    "content": """
                    Hypertension (High Blood Pressure):
                    - Normal: <120/80 mmHg, High: >130/80 mmHg
                    - Lifestyle: Low salt diet, exercise, weight management
                    - Medication: ACE inhibitors, diuretics as prescribed
                    - Monitoring: Check BP regularly, maintain log
                    - Complications: Heart disease, stroke, kidney damage
                    - Emergency: BP >180/120 with symptoms - seek immediate help
                    """,
                    "type": "chronic_condition",
                    "source": "Cardiology Guidelines",
                    "title": "Hypertension Management"
                }
            ]
            
            await self.upsert_healthcare_knowledge(default_docs)
            logger.info("Initialized default healthcare knowledge base")
            
        except Exception as e:
            logger.error(f"Failed to initialize default knowledge: {e}")


# Global Pinecone service instance
pinecone_service = PineconeService()