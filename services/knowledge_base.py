import chromadb
from chromadb.utils import embedding_functions
import os
from typing import Dict, Any, Optional, List

# Using a simple default embedding function (sentence-transformers)
# This keeps it free and local
embedding_func = embedding_functions.DefaultEmbeddingFunction()

class TravelKnowledgeBase:
    def __init__(self, path: str = "./data/chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="city_knowledge_v6",
            embedding_function=embedding_func
        )
        # Auto-seed if empty
        if self.collection.count() == 0:
            self.seed_initial_data()

    def seed_initial_data(self):
        """Seed the database with pure, single-paragraph intelligence reports."""
        cities = [
            {
                "id": "paris",
                "name": "Paris",
                "summary": "Paris serves as the vanguard of global aesthetics, a city where the weight of 19th-century Haussmann architecture meets the kinetic energy of a modern cultural titan. From the soaring iron lattice of the Eiffel Tower to the Gothic majesty of Notre-Dame, the city’s architectural vernacular is a testament to its enduring role as the City of Light. The Seine remains the pulse of the metropolis, carving through twenty arrondissements that balance storied tradition with avant-garde culinary and fashion innovation. Paris maintains an uncontested dominance in the global tourism index, continuing to offer a high-density sensory experience that rewards both first-time travelers and seasoned cultural connoisseurs with its unparalleled depth of history and sophistication.",
                "attractions": "Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, Montmartre District, Champs-Élysées"
            },
            {
                "id": "tokyo",
                "name": "Tokyo",
                "summary": "Tokyo represents the ultimate synthesis of futuristic hyper-efficiency and profound ancestral tradition. The metropolis is a masterclass in urban density, where the neon-drenched corridors of Shinjuku coexist seamlessly with the meditative silence of the Meiji Jingu shrine. As a global gastronomic epicenter, Tokyo’s culinary landscape is defined by a relentless pursuit of perfection, boasting a concentration of Michelin-starred excellence that remains unrivaled. Beyond its role as a critical hub for international commerce, Tokyo stands as a resilient model of the 21st-century megacity—a place where cutting-edge technological infrastructure supports a social tapestry deeply rooted in precision, respect, and aesthetic harmony.",
                "attractions": "Shibuya Crossing, Senso-ji Temple, Tokyo Tower, Akihabara Electric Town, Tsukiji Outer Market"
            },
            {
                "id": "london",
                "name": "London",
                "summary": "London is a sprawling, multi-layered palimpsest of Roman heritage and contemporary global influence. The city’s silhouette—defined by the neo-Gothic grandeur of the Palace of Westminster and the futuristic edge of the Shard—perfectly mirrors its capacity for constant reinvention. From the financial arteries of the Square Mile to the cultural richness of the West End, London functions as a premier global crossroad where tradition is not just preserved, but dynamically integrated into the modern zeitgeist. As a premier global capital, London remains a cornerstone of international culture and finance, offering a prestigious environment defined by world-class institutions, historic royal pageantry, and an inexhaustible diversity of global perspectives.",
                "attractions": "Tower of London, British Museum, London Eye, Buckingham Palace, Westminster Abbey"
            }
        ]
        
        for city in cities:
            self.collection.upsert(
                ids=[city["id"]],
                documents=[city["summary"].strip()],
                metadatas=[{
                    "name": city["name"],
                    "attractions": city["attractions"]
                }]
            )
        print(f"✅ Knowledge base seeded with {len(cities)} pure narrative reports.")

    def query_city(self, city_name: str) -> Optional[Dict[str, Any]]:
        """Check if city info exists in the vector store with literal match priority."""
        # Step 1: Check for literal match first (High Confidence Override)
        ids = self.collection.get(ids=[city_name.lower()])
        if ids['ids']:
            return {
                "summary": ids['documents'][0],
                "metadata": ids['metadatas'][0],
                "confidence": 1.0
            }

        # Step 2: Fallback to semantic similarity
        results = self.collection.query(
            query_texts=[city_name],
            n_results=1
        )
        
        if results['ids'] and results['distances'][0][0] < 0.6: 
            return {
                "summary": results['documents'][0][0],
                "metadata": results['metadatas'][0][0],
                "confidence": 1.0 - results['distances'][0][0]
            }
        return None

    def upsert_city(self, city_name: str, summary: str, attractions: List[str]):
        """Dynamically add or update a city in the local knowledge base."""
        self.collection.upsert(
            ids=[city_name.lower()],
            documents=[summary],
            metadatas=[{
                "name": city_name,
                "attractions": ", ".join(attractions)
            }]
        )
        print(f"✅ {city_name} has been synced to local memory.")

# Initialize the KB
kb = TravelKnowledgeBase()
