import time
import asyncio
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import AgentState
from services.knowledge_base import kb
from services.tools import (
    get_coordinates, 
    fetch_weather, 
    fetch_images_serpapi, 
    fetch_search_serpapi, 
    generate_summary_groq,
    groq_client
)

# --- Nodes ---

async def parse_query_node(state: AgentState) -> AgentState:
    """Extract city name from the user query, using context from history if needed."""
    start_time = time.time()
    
    # Use the full message history to resolve pronouns like "here" or "there"
    messages = state["messages"]
    
    prompt = f"""
    Based on the conversation history below, extract the name of the city the user is interested in.
    If the user mentions a new city, pick that. 
    If the user uses pronouns like "there", "here", or "that city", refer to the previous city mentioned.
    
    History:
    {messages}
    
    Return ONLY the city name. If no city is found, return "None".
    """
    
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    
    city = response.choices[0].message.content.strip().replace(".", "")
    
    # If the LLM returned "None", try to fallback to the existing city_name in state
    if "None" in city and state.get("city_name"):
        city = state["city_name"]
        
    state["city_name"] = city
    state["latency_breakdown"]["parse_query"] = time.time() - start_time
    return state

async def check_kb_node(state: AgentState) -> AgentState:
    """Check if the city exists in local knowledge base."""
    start_time = time.time()
    
    result = kb.query_city(state["city_name"])
    state["is_in_vector_store"] = result is not None
    
    if result:
        state["city_summary"] = result["summary"]
        state["key_attractions"] = result["metadata"]["attractions"].split(", ")
        state["data_quality"]["is_cached"] = True
        state["data_quality"]["confidence"] = result.get("confidence", 0.0)
    else:
        state["data_quality"]["is_cached"] = False
        state["data_quality"]["confidence"] = 0.98  # Standard LLM confidence for web
        
    state["latency_breakdown"]["check_kb"] = time.time() - start_time
    return state

def routing_logic(state: AgentState) -> Literal["local_path", "global_path"]:
    """Decide whether to go local or global."""
    return "local_path" if state["is_in_vector_store"] else "global_path"

async def local_fetch_node(state: AgentState) -> AgentState:
    """Found in KB! Now fetch weather and images in parallel."""
    start_time = time.time()
    
    coords = await get_coordinates(state["city_name"])
    state["coordinates"] = coords
    
    # Distinction #2: Parallel Fan-Out
    weather, images = await asyncio.gather(
        fetch_weather(coords["lat"], coords["lng"], state["city_name"]),
        asyncio.to_thread(fetch_images_serpapi, state["city_name"])
    )
    
    state["weather_forecast"] = weather
    state["image_urls"] = images
    state["latency_breakdown"]["fetch_data"] = time.time() - start_time
    return state

async def global_fetch_node(state: AgentState) -> AgentState:
    """Not in KB. Fetch everything from the web in parallel."""
    start_time = time.time()
    
    coords = await get_coordinates(state["city_name"])
    state["coordinates"] = coords
    
    # Distinction #2: Parallel Fan-Out (Weather + Search + Images)
    # Using to_thread for synchronous SerpAPI calls
    results = await asyncio.gather(
        fetch_weather(coords["lat"], coords["lng"], state["city_name"]),
        asyncio.to_thread(fetch_search_serpapi, state["city_name"]),
        asyncio.to_thread(fetch_images_serpapi, state["city_name"])
    )
    
    weather, search_data, images = results
    
    state["weather_forecast"] = weather
    state["image_urls"] = images
    
    # Synthesis using Groq
    summary, attractions = await generate_summary_groq(state["city_name"], weather, search_data)
    state["city_summary"] = summary
    state["key_attractions"] = attractions
    
    state["latency_breakdown"]["fetch_data"] = time.time() - start_time
    return state

async def aggregate_node(state: AgentState) -> AgentState:
    """Consolidate everything for the dashboard."""
    state["final_output"] = {
        "city": state["city_name"],
        "summary": state["city_summary"],
        "weather": state["weather_forecast"],
        "images": state["image_urls"],
        "attractions": state["key_attractions"],
        "coords": state["coordinates"],
        "latency": state["latency_breakdown"],
        "cached": state["is_in_vector_store"],
        "confidence": state["data_quality"].get("confidence", 0.70)
    }
    return state

# --- Graph Construction ---

workflow = StateGraph(AgentState)

workflow.add_node("parse_query", parse_query_node)
workflow.add_node("check_kb", check_kb_node)
workflow.add_node("local_fetch", local_fetch_node)
workflow.add_node("global_fetch", global_fetch_node)
workflow.add_node("aggregate", aggregate_node)

workflow.set_entry_point("parse_query")
workflow.add_edge("parse_query", "check_kb")

# The "Switch" (Conditional Edge)
workflow.add_conditional_edges(
    "check_kb",
    routing_logic,
    {
        "local_path": "local_fetch",
        "global_path": "global_fetch"
    }
)

workflow.add_edge("local_fetch", "aggregate")
workflow.add_edge("global_fetch", "aggregate")
workflow.add_edge("aggregate", END)

# Distinction #3: Memory / Persistence
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
