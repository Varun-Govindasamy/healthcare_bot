"""
Search Agent - Real-time outbreak and disease information using Serper API.
"""
import logging
from typing import List, Dict, Any, Optional
from crewai import Agent, Task
from crewai.tools import BaseTool
import httpx
import json
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)


class OutbreakSearchTool(BaseTool):
    """Tool for searching disease outbreak information."""
    
    name: str = "outbreak_searcher"
    description: str = "Searches for current disease outbreaks and health alerts in specific locations"
    
    def _run(self, disease: str, location: str, timeframe: str = "recent") -> str:
        """Search for disease outbreaks in specific location."""
        try:
            query = f"{disease} outbreak {location} {timeframe}"
            # This would use Serper API in practice
            return f"Searched for: {query}"
        except Exception as e:
            return f"Error searching for outbreaks: {str(e)}"


class HealthNewsSearchTool(BaseTool):
    """Tool for searching health news and updates."""
    
    name: str = "health_news_searcher"
    description: str = "Searches for latest health news and medical updates"
    
    def _run(self, query: str, location: Optional[str] = None) -> str:
        """Search for health news and updates."""
        try:
            search_query = f"health news {query}"
            if location:
                search_query += f" {location}"
            return f"Searched health news for: {search_query}"
        except Exception as e:
            return f"Error searching health news: {str(e)}"


class SearchAgent:
    """Agent for real-time health information and outbreak searches."""
    
    def __init__(self):
        self.tools = [
            OutbreakSearchTool(),
            HealthNewsSearchTool()
        ]
        
        self.agent = Agent(
            role="Health Information Researcher",
            goal="Provide real-time health information, disease outbreaks, and medical news from reliable sources",
            backstory="""You are a health information researcher specialized in finding current, 
            location-specific health data including disease outbreaks, health alerts, and medical news. 
            You have access to real-time search capabilities and focus on providing timely, 
            accurate information from government health departments and reliable medical sources.""",
            tools=self.tools,
            verbose=True,
            allow_delegation=False
        )
        
        self.serper_api_key = settings.serper_api_key
        self.base_url = "https://google.serper.dev/search"
    
    async def search_serper(self, query: str, location: Optional[str] = None) -> Dict[str, Any]:
        """Search using Serper API."""
        try:
            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json"
            }
            
            search_query = query
            if location:
                search_query += f" {location}"
            
            payload = {
                "q": search_query,
                "num": 10,
                "location": location or "India"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Serper API error: {response.status_code}")
                    return {"error": f"API error: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            return {"error": str(e)}
    
    def create_outbreak_search_task(self, disease: str, district: str, state: str) -> Task:
        """Create task for searching disease outbreaks."""
        return Task(
            description=f"""
            Search for current information about {disease} outbreaks in {district}, {state}.
            
            Find information about:
            1. Recent cases reported in the area
            2. Government health department alerts
            3. Prevention measures being recommended
            4. Vaccination drives or health campaigns
            5. Hospital preparedness and capacity
            6. Travel advisories or restrictions
            
            Focus on official sources like:
            - Ministry of Health and Family Welfare (MoHFW)
            - State health departments
            - WHO updates
            - Credible news sources
            
            Location: {district}, {state}
            Disease: {disease}
            Timeframe: Last 30 days
            """,
            agent=self.agent,
            expected_output="Current outbreak information with case numbers, prevention measures, and official recommendations"
        )
    
    def create_health_news_task(self, topic: str, location: Optional[str] = None) -> Task:
        """Create task for searching health news."""
        return Task(
            description=f"""
            Search for latest health news and updates about {topic}.
            Location focus: {location or 'India'}
            
            Find recent information about:
            1. Medical research and breakthroughs
            2. New treatment guidelines
            3. Health policy changes
            4. Vaccination updates
            5. Public health initiatives
            6. Disease prevention campaigns
            
            Focus on:
            - Government health announcements
            - Medical journal publications
            - WHO/CDC updates
            - Credible medical news sources
            
            Timeframe: Last 7 days
            """,
            agent=self.agent,
            expected_output="Latest health news and medical updates with source links and publication dates"
        )
    
    async def search_disease_outbreak(self, disease: str, district: str, state: str) -> Dict[str, Any]:
        """Search for disease outbreak information."""
        try:
            # Primary search query
            primary_query = f"{disease} outbreak cases {district} {state} 2024"
            primary_results = await self.search_serper(primary_query, f"{district}, {state}")
            
            # Secondary search for state-level info
            state_query = f"{disease} cases {state} health department alert"
            state_results = await self.search_serper(state_query, state)
            
            # Prevention measures search
            prevention_query = f"{disease} prevention measures guidelines {state}"
            prevention_results = await self.search_serper(prevention_query, state)
            
            return {
                "disease": disease,
                "location": f"{district}, {state}",
                "local_cases": primary_results,
                "state_info": state_results,
                "prevention": prevention_results,
                "search_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error searching disease outbreak: {e}")
            return {
                "disease": disease,
                "location": f"{district}, {state}",
                "error": str(e)
            }
    
    async def search_health_topic(self, topic: str, location: Optional[str] = None) -> Dict[str, Any]:
        """Search for general health topic information."""
        try:
            # General health search
            health_query = f"{topic} health news India recent"
            health_results = await self.search_serper(health_query, location)
            
            # Guidelines search
            guidelines_query = f"{topic} guidelines treatment India MoHFW"
            guidelines_results = await self.search_serper(guidelines_query, location)
            
            return {
                "topic": topic,
                "location": location or "India",
                "health_news": health_results,
                "guidelines": guidelines_results,
                "search_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error searching health topic: {e}")
            return {
                "topic": topic,
                "location": location,
                "error": str(e)
            }
    
    async def get_vaccination_info(self, vaccine: str, location: str) -> Dict[str, Any]:
        """Get vaccination drive and availability information."""
        try:
            # Vaccination availability search
            availability_query = f"{vaccine} vaccination drive {location} government hospital"
            availability_results = await self.search_serper(availability_query, location)
            
            # Side effects and guidelines
            safety_query = f"{vaccine} side effects precautions guidelines India"
            safety_results = await self.search_serper(safety_query)
            
            return {
                "vaccine": vaccine,
                "location": location,
                "availability": availability_results,
                "safety_info": safety_results,
                "search_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error searching vaccination info: {e}")
            return {
                "vaccine": vaccine,
                "location": location,
                "error": str(e)
            }
    
    async def get_seasonal_health_alerts(self, season: str, location: str) -> Dict[str, Any]:
        """Get seasonal health alerts and precautions."""
        try:
            # Seasonal health issues
            seasonal_query = f"{season} health issues diseases {location} prevention"
            seasonal_results = await self.search_serper(seasonal_query, location)
            
            # Government advisories
            advisory_query = f"{season} health advisory {location} government MoHFW"
            advisory_results = await self.search_serper(advisory_query, location)
            
            return {
                "season": season,
                "location": location,
                "health_issues": seasonal_results,
                "advisories": advisory_results,
                "search_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error searching seasonal health alerts: {e}")
            return {
                "season": season,
                "location": location,
                "error": str(e)
            }


# Global search agent instance
search_agent = SearchAgent()