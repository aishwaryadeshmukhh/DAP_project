from typing import TypedDict, List, Dict, Any, Annotated
import operator

class AgentState(TypedDict):
    """
    The state of the travel assistant agent.
    
    This TypedDict defines the shared data structure that flows between nodes 
    in the LangGraph orchestration.
    """
    # Inputs & Extraction
    user_query: str
    city_name: str
    is_in_vector_store: bool
    
    # Fetched Data
    weather_forecast: List[Dict[str, Any]]
    image_urls: List[str]
    city_summary: str
    key_attractions: List[str]
    coordinates: Dict[str, float]  # {'lat': ..., 'lng': ...}
    
    # Message History (Annotated with operator.add for merging)
    # This allows LangGraph to append new messages to the history automatically
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # Metadata for Tab 4 & 5 (Performance & Debug)
    data_quality: Dict[str, Any]  # {'is_cached': bool, 'weather_source': str, ...}
    latency_breakdown: Dict[str, float]  # {'node_name': time_in_seconds, ...}
    tokens_used: int
    
    # Final consolidated output for Streamlit
    final_output: Dict[str, Any]
