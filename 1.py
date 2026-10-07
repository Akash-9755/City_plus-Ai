from dotenv import load_dotenv
load_dotenv()

import os
import requests
from rich import print as rprint  # EDIT 14: renamed so the built-in print is not shadowed

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langgraph.checkpoint.memory import InMemorySaver  # EDIT 6: conversation memory
from tavily import TavilyClient
# EDIT 12: removed unused imports (HumanMessage, ChatGoogleGenerativeAI, RunnableLambda)

REQUEST_TIMEOUT = 10  # EDIT 1/2: every HTTP call now has a timeout
REQUIRE_APPROVAL = True  # EDIT 9: approval can now be switched off from one place
HEADERS = {"User-Agent": "CityPulseAI/1.0"}

# EDIT 5: Tavily client created once, not on every call
_tavily_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=_tavily_key) if _tavily_key else None


# =========================
# 📍 Shared geocoder
# =========================
# EDIT 8: one geocoder used by ALL tools, so a city name always resolves to the same place

def geocode(city: str):
    """Return (name, latitude, longitude) for a city, or None if not found."""
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("results")
    except (requests.RequestException, ValueError):
        return None

    if not results:
        return None

    loc = results[0]
    name = loc["name"]
    if loc.get("country"):
        name = f"{name}, {loc['country']}"
    return name, loc["latitude"], loc["longitude"]


# =========================
# 🌦️ Weather Tool
# =========================

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""

    # EDIT 1: check the API key first
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather service is not configured (missing OPENWEATHER_API_KEY)."

    place = geocode(city)  # EDIT 8: shared geocoder
    if not place:
        return f"Could not find the city: {city}"
    name, lat, lon = place

    # EDIT 1: try/except + raise_for_status + timeout
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return f"Could not fetch weather for {city}. Please try again."

    return f"Weather in {name}: {desc}, {temp}°C"


# =========================
# 🌫️ Air Quality Tool
# =========================  (EDIT 13: header fixed, it was labelled "News")

@tool
def get_air_quality(city: str) -> str:
    """Get the current air quality and AQI for a city."""

    place = geocode(city)  # EDIT 8: shared geocoder
    if not place:
        return f"Could not find the city: {city}"
    name, lat, lon = place

    # EDIT 2: try/except + raise_for_status + timeout + safe access to "current"
    try:
        response = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "european_aqi,us_aqi,pm2_5,pm10",
                "timezone": "auto",
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        current = response.json().get("current")
    except (requests.RequestException, ValueError):
        return f"Could not fetch air quality for {city}. Please try again."

    if not current:
        return f"No air quality data available for {city}."

    return (
        f"Air quality in {name}:\n"
        f"European AQI: {current.get('european_aqi')}\n"
        f"US AQI: {current.get('us_aqi')}\n"
        f"PM2.5: {current.get('pm2_5')} µg/m³\n"
        f"PM10: {current.get('pm10')} µg/m³"
    )


# =========================
# 📍 Search Places (OpenStreetMap)
# =========================  (EDIT 13: header fixed, it was labelled "News")

# EDIT 10: dictionary replaces the repeated if/elif chain.
# To add a new place type, add one entry here.
PLACE_TAGS = {
    "restaurant": [("amenity", "restaurant"), ("amenity", "fast_food")],
    "hospital": [("amenity", "hospital")],
    "attraction": [("tourism", "attraction"), ("tourism", "museum"), ("tourism", "viewpoint")],
    "petrol pump": [("amenity", "fuel")],
}

PLACE_ALIASES = {
    "restaurants": "restaurant",
    "hospitals": "hospital",
    "attractions": "attraction",
    "petrolpump": "petrol pump",
    "petrol pumps": "petrol pump",
    "fuel": "petrol pump",
}


@tool
def search_places(city: str, place_type: str, radius_m: int = 2000) -> str:
    """
    Search for restaurants, hospitals, petrol pumps, or attractions
    near the centre of a city using OpenStreetMap.
    radius_m is the search radius in metres (default 2000).
    """

    place = geocode(city)  # EDIT 8: shared geocoder (replaces Nominatim)
    if not place:
        return f"Could not find the city: {city}"
    city_name, lat, lon = place

    place_type = place_type.lower().strip()
    place_type = PLACE_ALIASES.get(place_type, place_type)

    if place_type not in PLACE_TAGS:
        return "Invalid place type. Use restaurant, hospital, attraction, or petrol pump."

    radius_m = max(500, min(int(radius_m), 10000))  # EDIT 4: adjustable, but capped

    # Build the Overpass query from the dictionary
    filters = "\n".join(
        f'nwr["{key}"="{value}"](around:{radius_m},{lat},{lon});'
        for key, value in PLACE_TAGS[place_type]
    )
    query = f"[out:json][timeout:15];\n(\n{filters}\n);\nout tags;"

    try:
        overpass_response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers=HEADERS,
            timeout=30,
        )
        if overpass_response.status_code != 200:
            return "OpenStreetMap service is temporarily unavailable. Please try again."
        elements = overpass_response.json().get("elements", [])
    except requests.RequestException:
        return "Could not connect to OpenStreetMap. Please try again."
    except ValueError:
        return "OpenStreetMap returned an invalid response."

    # EDIT 3: filter for named places FIRST, then take the top 10
    named = [p for p in elements if p.get("tags", {}).get("name")]

    if not named:
        return f"No named {place_type} places found within {radius_m} m of the centre of {city_name}."

    results = []
    for p in named[:10]:
        tags = p["tags"]
        address = ", ".join(
            part
            for part in (tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city"))
            if part
        ) or "Address not available"
        results.append(f"Name: {tags['name']}\nAddress: {address}")

    # EDIT 4: the output now says the search area, so the model can tell the user
    return (
        f"{place_type.title()} places within {radius_m} m of the centre of {city_name}:\n\n"
        + "\n\n".join(results)
    )


# =========================
# 📰 News Tool (Tavily)
# =========================

@tool
def get_news(topic: str) -> str:
    """Search the web for the latest news about a given topic."""

    # EDIT 5: missing-key check + try/except, client reused
    if tavily is None:
        return "News service is not configured (missing TAVILY_API_KEY)."

    try:
        response = tavily.search(
            query=f"latest news about {topic}",
            topic="news",
            max_results=5,
        )
    except Exception:
        return f"Could not fetch news about {topic}. Please try again."

    results = response.get("results", [])
    if not results:
        return f"No latest news found about {topic}."

    news = [
        f"Title: {r.get('title', 'No title')}\n"
        f"Summary: {r.get('content', 'No description')}\n"
        f"Source: {r.get('url', '')}"
        for r in results
    ]
    return "\n\n".join(news)


# =========================
# 🧠 LLM + Agent Setup
# =========================

# llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
llm = ChatGroq(model="openai/gpt-oss-120b")


@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before tool calls (terminal only)."""
    if not REQUIRE_APPROVAL:  # EDIT 9
        return handler(request)

    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.strip().lower() not in ("yes", "y"):  # EDIT 9: accepts "y" and stray spaces
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"],
        )

    return handler(request)


# EDIT 15: a real system prompt
SYSTEM_PROMPT = (
    "You are a helpful city assistant. You have four tools: "
    "get_weather (current weather), get_air_quality (AQI and pollutants) and also tell me full details about that, "
    "get_news (latest news on a topic), and search_places "
    "(restaurants, hospitals, petrol pumps, attractions near a city centre). "
    "Use a tool whenever the question needs live data, and never guess values. "
    "If the user does not mention a city, ask which city they mean. "
    "If a tool returns an error or no results, say so plainly instead of making things up. "
    "Keep answers proper and understandable ways."
)

agent = create_agent(
    llm,
    tools=[get_weather, get_news, get_air_quality, search_places],
    system_prompt=SYSTEM_PROMPT,
    middleware=[human_approval],
    checkpointer=InMemorySaver(),  # EDIT 6: remembers earlier messages in the session
)

# EDIT 6: all messages in one session share this thread id
CONFIG = {"configurable": {"thread_id": "city-agent-session"}}

rprint("City Agent | type exit to quit")

while True:
    user_input = input("You : ").strip()
    if not user_input:
        continue
    if user_input.lower() == "exit":
        break

    # EDIT 7: one failure no longer kills the chat loop
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=CONFIG,
        )
        # EDIT 11: clean_response/RunnableLambda removed, plain .strip() is enough
        final_response = str(result["messages"][-1].content).strip()
        rprint("bot : ", final_response)
    except Exception as e:
        rprint(f"bot : Something went wrong ({type(e).__name__}). Please try again.")