# Phase-Wise Implementation Plan: Multi-Modal Travel Assistant

This plan outlines the step-by-step development of the Multi-Modal Travel Assistant using **LangGraph** for orchestration and **Groq** for high-performance LLM calls.

---

## Phase 1: Foundation & Environment Setup
**Goal**: Establish the project structure and secure API integrations.

- [ ] **Environment Initialization**:
    - Create a clean virtual environment.
    - Install core dependencies: `langgraph`, `groq`, `streamlit`, `chromadb`, `httpx`, `pandas`, `python-dotenv`.
- [ ] **API Configuration (Pro Tier)**:
    - Setup `.env` file with:
        - `GROQ_API_KEY`
        - `SERPAPI_API_KEY` (For high-end search & images)
    - **Open-Meteo** (Free) will still handle weather.
- [ ] **State Definition**:
    - Define `AgentState` using `TypedDict` in `state.py` to track `city_name`, `is_in_vector_store`, `weather_data`, `image_urls`, `messages`, and `metadata`.

## Phase 2: Knowledge Base (The "Switch" Data)
**Goal**: Implement local retrieval logic to demonstrate intelligent routing.

- [ ] **Vector Store Setup**:
    - Implement `vector_store.py` using **ChromaDB**.
    - Ingest sample travel data for 3-5 major cities (e.g., Paris, Tokyo, London, New York) to simulate "cached" knowledge.
- [ ] **Routing Utility**:
    - Create a function to check if a city exists in the local store.

## Phase 3: Tool Development (Async Power)
**Goal**: Build robust, asynchronous tools for external data fetching.

- [ ] **Weather Tool**: Integration with OpenWeatherMap (5-day forecast).
- [ ] **Image Tool**: Integration with Unsplash/SerpAPI for high-quality visuals.
- [ ] **Web Search Tool**: Fallback search via SerpAPI/Tavily for unknown cities.
- [ ] **Summary Tool**: Groq-powered tool to synthesize information into a travel brief.

## Phase 4: LangGraph Orchestration (The Brain)
**Goal**: Implement the agent logic and distinction challenges.

- [ ] **Graph Construction**:
    - Define nodes for Query Parsing, Routing, Local Fetch, and Global Fetch.
- [ ] **Distinction #1: Manual Tool Execution**:
    - Implement manual parsing of Groq's `tool_calls` to demonstrate deep protocol understanding.
- [ ] **Distinction #2: Parallel Fan-Out**:
    - Use `asyncio.gather` in the fetch node to run weather, image, and summary calls simultaneously.
- [ ] **Distinction #3: State Persistence**:
    - Integrate `MemorySaver` (Checkpointer) to preserve conversation context across threads.

## Phase 5: Premium Production-Grade UI
**Goal**: Create a "WOW" factor UI that stands out in the top 1%.

### 🏗️ Production-Grade Streamlit Dashboard with 5 Tabs
Instead of just displaying text and images, the application will include:

#### Tab 1: Overview 🏙️
- Key metrics (population, temperature, humidity, best time to visit)
- City summary with rich formatting
- Key attractions list with emojis
- Interactive map showing location coordinates

#### Tab 2: Weather 🌤️
- Temperature trend line chart
- Humidity bar chart
- Daily breakdown table with detailed weather data
- Smart weather summary (avg/max/min temperatures)
- **Smart Packing Checklist** (auto-generates based on weather!):
    - *If temp > 20°C → "Sunscreen" ✅*
    - *If rain predicted → "Umbrella" ✅*
    - *If cold → "Winter coat" ✅*

#### Tab 3: Gallery 📸
- Multiple layout options (Grid, Carousel, Full Width)
- Gallery statistics (total images, source, update date)
- Professional image display with captions

#### Tab 4: Costs & Performance 💰
- **Data Quality Indicators**: Is it from cache or live API?
- **Latency Breakdown**: Shows time for each operation.
- **Cost Analysis Dashboard**: How much this query cost (Groq is free but we'll track token usage).
- **Optimization Suggestions**: What could be improved.
- Estimated monthly cost calculation.
- Shows 3x speed improvement from parallelization.

#### Tab 5: Debug Info 🔍
- **Data Source Routing**: Shows which path (Vector Store vs Web) the request took.
- **Tools Called**: What LLM tools executed.
- **Raw JSON Response**: For power users.
- **Model Configuration**: Which model, temperature, tokens used.

### 🔧 Additional Features
- **Sidebar Enhancements**:
    - 📚 Search history (quickly re-query previous cities)
    - ❤️ Favorites (save cities to visit later)
    - Quick access buttons
- **Session Management**:
    - Persistent search history and saved favorites across sessions.
    - Thread ID for stateful conversations.
- **Smart Error Handling**:
    - Loading spinners and fallback to cached data if API times out.
    - Graceful error messages (not crashes).

## Phase 6: Polish, Performance & Submission
**Goal**: Ensure production-grade reliability.

- [ ] **Error Handling**:
    - Implement timeouts and fallbacks (e.g., show cached data if API is down).
- [ ] **Observability**:
    - Add a "Debug" tab showing latency breakdowns and API costs.
- [ ] **Documentation**:
    - Finalize `README.md` with setup instructions and architecture diagrams.

---

## Tech Stack Summary
- **LLM**: Groq (Llama-3.3-70b-versatile) - **FREE**
- **Orchestration**: LangGraph
- **Frontend**: Streamlit
- **Vector DB**: ChromaDB - **FREE**
- **Weather**: Open-Meteo - **FREE**
- **Search & Images**: SerpAPI - **PRO**
