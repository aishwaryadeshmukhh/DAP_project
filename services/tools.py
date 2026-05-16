import os
import httpx
import asyncio
from typing import List, Dict, Any
from serpapi import GoogleSearch
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

async def get_coordinates(city: str) -> Dict[str, float]:
    """Fetch latitude and longitude for a city using Open-Meteo Geocoding."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        if not data.get("results"):
            return {"lat": 0.0, "lng": 0.0}
        result = data["results"][0]
        return {"lat": result["latitude"], "lng": result["longitude"]}

async def fetch_weather_openmeteo(lat: float, lng: float) -> List[Dict[str, Any]]:
    """Primary: Open-Meteo (Free, Keyless)"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,humidity_2m_max&timezone=auto"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=5.0)
        data = response.json()
        daily = data.get("daily", {})
        return [{
            "date": daily["time"][i],
            "temp": (daily["temperature_2m_max"][i] + daily["temperature_2m_min"][i]) / 2,
            "precipitation": daily["precipitation_sum"][i],
            "humidity": daily["humidity_2m_max"][i]
        } for i in range(len(daily.get("time", [])))]

async def fetch_weather_openweathermap(lat: float, lng: float) -> List[Dict[str, Any]]:
    """Secondary: OpenWeatherMap (Requires Key)"""
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key or "your_" in key: return []
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lng}&appid={key}&units=metric"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=5.0)
        data = response.json()
        # Transform OWM 3-hour data to daily
        return [{"date": item["dt_txt"][:10], "temp": item["main"]["temp"], "humidity": item["main"]["humidity"], "precipitation": 0} 
                for item in data.get("list", [])[::8]]

async def fetch_weather_weatherapi(city: str) -> List[Dict[str, Any]]:
    """Tertiary: WeatherAPI.com (Requires Key)"""
    key = os.getenv("WEATHERAPI_KEY")
    if not key or "your_" in key: return []
    url = f"http://api.weatherapi.com/v1/forecast.json?key={key}&q={city}&days=7"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=5.0)
        data = response.json()
        return [{
            "date": day["date"],
            "temp": day["day"]["avgtemp_c"],
            "humidity": day["day"]["avghumidity"],
            "precipitation": day["day"]["totalprecip_mm"]
        } for day in data.get("forecast", {}).get("forecastday", [])]

async def fetch_weather(lat: float, lng: float, city: str) -> List[Dict[str, Any]]:
    """Triple-Fallback Weather Orchestrator."""
    # 1. Try Open-Meteo
    try:
        data = await fetch_weather_openmeteo(lat, lng)
        if data: return data
    except Exception: pass
    
    # 2. Try OpenWeatherMap
    try:
        data = await fetch_weather_openweathermap(lat, lng)
        if data: return data
    except Exception: pass
    
    # 3. Try WeatherAPI.com
    try:
        data = await fetch_weather_weatherapi(city)
        if data: return data
    except Exception: pass
    
    return []

def fetch_images_serpapi(city: str) -> List[str]:
    """Fetch high-quality images using SerpAPI."""
    params = {
        "engine": "google_images",
        "q": f"{city} travel landscape attractions",
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return [img["original"] for img in results.get("images_results", [])[:10]]

def fetch_search_serpapi(city: str) -> Dict[str, Any]:
    """Fetch web search results for city details using SerpAPI."""
    params = {
        "engine": "google",
        "q": f"travel guide for {city} population attractions history",
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Extracting knowledge graph if available
    kg = results.get("knowledge_graph", {})
    return {
        "summary": kg.get("description", "No summary available."),
        "population": kg.get("population", "Unknown"),
        "key_facts": results.get("organic_results", [])[:3]
    }

async def generate_summary_groq(city: str, weather: List[Dict[str, Any]], search_data: Dict[str, Any]) -> tuple:
    """Synthesize data into a premium travel summary and extract key landmarks."""
    avg_temp = sum(d["temp"] for d in weather) / len(weather) if weather else 20.0
    
    prompt = f"""
    Analyze the following data for {city} and provide two specific outputs:
    1. A single, high-fidelity, professional paragraph (approx 100 words) summarizing the city's essence. Use elite vocabulary.
    2. A comma-separated list of the 5 most iconic physical landmarks or attractions in the city.
    
    Search Context: {search_data['summary']} | {search_data.get('key_facts', [])}
    Weather Context: {avg_temp}C
    
    Return your response EXACTLY in this format:
    SUMMARY: [Your paragraph here]
    LANDMARKS: [Landmark 1, Landmark 2, Landmark 3, Landmark 4, Landmark 5]
    """
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    res = chat_completion.choices[0].message.content
    
    try:
        summary = res.split("SUMMARY:")[1].split("LANDMARKS:")[0].strip()
        landmarks = res.split("LANDMARKS:")[1].strip().split(", ")
        return summary, landmarks
    except:
        return res, ["Gateway of India", "Marine Drive", "Colaba Causeway"] # Fallback
