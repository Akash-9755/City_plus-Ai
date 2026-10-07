from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent 
from langchain.agents.middleware import wrap_tool_call
import requests
import os
from langchain_core.runnables import RunnableLambda

# =========================
# 🌦️ Weather Tool
# =========================


@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params)

    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"

# =========================
# 📰 Air Quality 
# =========================

@tool
def get_air_quality(city: str) -> str:
    """Get the current air quality and AQI for a city."""
    
    # Step 1: Find latitude and longitude of the city
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
 
    geo_params = {
        "name" : city,
        "count" : 1,
        "language": "en",
        "format": "json"
        
    }
    
    geo_response = requests.get(geo_url, params=geo_params)
    geo_data = geo_response.json()
    
    if "results" not in geo_data:
        return f"Could not find the city: {city}"

    location = geo_data["results"][0]

    latitude = location["latitude"]
    longitude = location["longitude"]
    
    # Step 2: Get air quality
    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    air_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "european_aqi,us_aqi,pm2_5,pm10",
        "timezone": "auto"
    }
    air_response = requests.get(air_url, params=air_params)
    air_data = air_response.json()

    current = air_data["current"]

    return (
        f"Air quality in {location['name']}:\n"
        f"European AQI: {current.get('european_aqi')}\n"
        f"US AQI: {current.get('us_aqi')}\n"
        f"PM2.5: {current.get('pm2_5')} µg/m³\n"
        f"PM10: {current.get('pm10')} µg/m³"
    )
    

# =========================
# 📰 Search Places (OpenStreetMap)
# =========================

@tool
def search_places(city: str, place_type: str) -> str:
    """
    Search for restaurants, hospitals, petrol pumps, or attractions
    in a city using OpenStreetMap.
    """

    # ------------------------------------------------
    # STEP 1: Convert city name into coordinates
    # ------------------------------------------------

    geocode_url = "https://nominatim.openstreetmap.org/search"

    geocode_params = {
        "q": city,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "CityPulseAI/1.0"
    }

    try:
        response = requests.get(
            geocode_url,
            params=geocode_params,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException:
        return f"Could not find location information for {city}."

    if not data:
        return f"Could not find the city: {city}"

    latitude = float(data[0]["lat"])
    longitude = float(data[0]["lon"])


    # ------------------------------------------------
    # STEP 2: Normalize place type
    # ------------------------------------------------

    place_type = place_type.lower().strip()


    # ------------------------------------------------
    # STEP 3: Create Overpass query
    # ------------------------------------------------

    if place_type in ["restaurant", "restaurants"]:

        query = f"""
        [out:json][timeout:15];

        (
          nwr["amenity"="restaurant"]
             (around:2000,{latitude},{longitude});

          nwr["amenity"="fast_food"]
             (around:2000,{latitude},{longitude});
        );

        out tags;
        """


    elif place_type in ["hospital", "hospitals"]:

        query = f"""
        [out:json][timeout:15];

        nwr["amenity"="hospital"]
           (around:2000,{latitude},{longitude});

        out tags;
        """


    elif place_type in ["attraction", "attractions"]:

        query = f"""
        [out:json][timeout:15];

        (
          nwr["tourism"="attraction"]
             (around:2000,{latitude},{longitude});

          nwr["tourism"="museum"]
             (around:2000,{latitude},{longitude});

          nwr["tourism"="viewpoint"]
             (around:2000,{latitude},{longitude});
        );

        out tags;
        """


    elif place_type in [
        "petrolpump",
        "petrol pump",
        "petrol pumps",
        "fuel"
    ]:

        query = f"""
        [out:json][timeout:15];

        nwr["amenity"="fuel"]
           (around:2000,{latitude},{longitude});

        out tags;
        """


    else:

        return (
            "Invalid place type. "
            "Use restaurant, hospital, attraction, or petrol pump."
        )


    # ------------------------------------------------
    # STEP 4: Call Overpass API
    # ------------------------------------------------

    overpass_url = "https://overpass-api.de/api/interpreter"

    try:

        overpass_response = requests.post(
            overpass_url,
            data={"data": query},
            headers=headers,
            timeout=30
        )

        if overpass_response.status_code != 200:

            return (
                "OpenStreetMap service is temporarily unavailable. "
                "Please try again."
            )

        places_data = overpass_response.json()

    except requests.RequestException:

        return (
            "Could not connect to OpenStreetMap. "
            "Please try again."
        )

    except ValueError:

        return "OpenStreetMap returned an invalid response."


    # ------------------------------------------------
    # STEP 5: Get places
    # ------------------------------------------------

    elements = places_data.get("elements", [])

    if not elements:

        return f"No {place_type} places found near {city}."


    # ------------------------------------------------
    # STEP 6: Format results
    # ------------------------------------------------

    results = []

    for place in elements[:10]:

        tags = place.get("tags", {})

        name = tags.get("name")

        if not name:
            continue

        street = tags.get("addr:street")
        house_number = tags.get("addr:housenumber")
        city_name = tags.get("addr:city")

        address_parts = []

        if house_number:
            address_parts.append(house_number)

        if street:
            address_parts.append(street)

        if city_name:
            address_parts.append(city_name)

        address = ", ".join(address_parts)

        if not address:
            address = "Address not available"

        results.append(
            f"Name: {name}\n"
            f"Address: {address}"
        )


    # ------------------------------------------------
    # STEP 7: Final response
    # ------------------------------------------------

    if not results:

        return f"No named {place_type} places found near {city}."


    return (
        f"Places found in/near {city}:\n\n"
        + "\n\n".join(results)
    )
# =========================
# 📰 News Tool (Tavily)
# =========================

@tool
def get_news(topic: str) -> str:
    """Search the web for the latest news about a given topic."""

    # 1. Get Tavily API key
    api_key = os.getenv("TAVILY_API_KEY")

    # 2. Create Tavily client
    tavily = TavilyClient(api_key=api_key)

    # 3. Search the web
    response = tavily.search(
        query=f"latest news about {topic}",
        topic="news",
        max_results=5
    )

    # 4. Get search results
    results = response.get("results", [])

    if not results:
        return f"No latest news found about {topic}."

    # 5. Format the results
    news = []

    for result in results:
        title = result.get("title", "No title")
        content = result.get("content", "No description")
        url = result.get("url", "")

        news.append(
            f"Title: {title}\n"
            f"Summary: {content}\n"
            f"Source: {url}"
        )

    # 6. Return the news
    return "\n\n".join(news)


def clean_response(response):
    return response.strip()


response_runnable = RunnableLambda(clean_response)


# =========================
# 🧠 LLM Setup
# =========================

#llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
llm = ChatGroq(model = "openai/gpt-oss-120b")


@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)  

agent = create_agent(
    llm,
    tools = [get_weather,get_news,get_air_quality,search_places],
    system_prompt= """
You are City Pulse, a friendly and knowledgeable city assistant.

You have four tools:
- get_weather: current weather of a city.
- get_air_quality: current AQI and pollutant levels (PM2.5, PM10) of a city.
- get_news: latest news about a topic or city.
- search_places: restaurants, hospitals, petrol pumps, or attractions near a city centre.

HOW TO USE TOOLS
- Use a tool whenever the question needs live or location-specific data. Never guess or invent weather, AQI, news, or places.
- If the user asks about more than one thing (for example weather and air quality), call all the tools needed before answering.
- If the user does not mention a city, use the city from earlier in the conversation. If there is none, ask which city they mean.
- If a tool returns an error or no results, say so plainly and suggest trying again or rephrasing. Do not make up data.

HOW TO ANSWER
- Give a complete, helpful answer. Use every useful detail the tool returned: temperature, conditions, AQI values, every place name and address, every news headline and summary.
- Explain what the numbers mean in plain language. For example, say whether an AQI value is good, moderate, or unhealthy and what that means for going outside, exercise, or sensitive people.
- When the user asks a combined question like "is it a good day to go out?", give a clear recommendation based on the data.
- Show lists (places, news) with one item per line. For news, include the source link.
- For places, mention that results are near the city centre and that some entries may lack an address.
- Keep a warm, natural tone. Do not be overly short, and do not pad with filler.

BOUNDARIES
- Stay on topic: weather, air quality, news, and places in cities. For anything unrelated, politely say what you can help with.
""",
    middleware= [human_approval]
)

print("City Agent | type exit to quit")

# i want to create a memory for the agent so that it can remember the previous conversations and use them in the future. I will create a list to store the messages and pass it to the agent when invoking it.
messages = []


while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break 
    messages.append({"role": "user", "content": user_input})
    result = agent.invoke({
        "messages": messages
    })
    
    messages = result["messages"]

    final_response = result["messages"][-1].content

    cleaned_response = response_runnable.invoke(final_response)

    print("bot : ", cleaned_response)
    
    