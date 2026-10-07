import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

load_dotenv()
st.set_page_config(page_title="City Pulse", page_icon="🏙️", layout="wide")

# Streamlit Cloud: copy st.secrets into env vars so the rest of the code can use os.getenv
for _k in ("GROQ_API_KEY", "OPENWEATHER_API_KEY", "TAVILY_API_KEY"):
    try:
        if _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
    except Exception:
        pass


# =====================================================================
# TOOLS
# =====================================================================
HEADERS = {"User-Agent": "CityPulseAI/1.0"}

# canonical place type -> OSM tag filters
PLACE_FILTERS = {
    "restaurant": [("amenity", "restaurant"), ("amenity", "fast_food")],
    "hospital": [("amenity", "hospital")],
    "attraction": [("tourism", "attraction"), ("tourism", "museum"), ("tourism", "viewpoint")],
    "petrol pump": [("amenity", "fuel")],
}
PLACE_ALIASES = {
    "restaurants": "restaurant", "hospitals": "hospital", "attractions": "attraction",
    "petrolpump": "petrol pump", "petrol pumps": "petrol pump", "fuel": "petrol pump",
}


# ---------------------------------------------------------------- data helpers
def fetch_weather(city: str) -> dict:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=10,
        )
        data = r.json()
    except (requests.RequestException, ValueError):
        return {"error": "Could not reach the weather service."}
    if str(data.get("cod")) != "200":
        return {"error": data.get("message", "Could not fetch weather")}
    return {
        "city": data.get("name", city),
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "wind": data.get("wind", {}).get("speed"),
        "desc": data["weather"][0]["description"],
    }


def fetch_air_quality(city: str) -> dict:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        ).json()
        if "results" not in geo:
            return {"error": f"Could not find the city: {city}"}
        loc = geo["results"][0]
        air = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "european_aqi,us_aqi,pm2_5,pm10",
                "timezone": "auto",
            },
            timeout=10,
        ).json()
        cur = air["current"]
    except (requests.RequestException, ValueError, KeyError):
        return {"error": "Could not fetch air quality."}
    return {
        "city": loc["name"],
        "eu_aqi": cur.get("european_aqi"),
        "us_aqi": cur.get("us_aqi"),
        "pm25": cur.get("pm2_5"),
        "pm10": cur.get("pm10"),
    }


def fetch_news(topic: str, max_results: int = 5) -> dict:
    try:
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        resp = tavily.search(query=f"latest news about {topic}", topic="news", max_results=max_results)
    except Exception:
        return {"error": "Could not fetch news."}
    articles = [
        {
            "title": r.get("title", "No title"),
            "summary": r.get("content", "No description"),
            "url": r.get("url", ""),
        }
        for r in resp.get("results", [])
    ]
    return {"articles": articles}


def fetch_places(city: str, place_type: str, radius: int = 2000, limit: int = 10) -> dict:
    place_type = place_type.lower().strip()
    place_type = PLACE_ALIASES.get(place_type, place_type)
    if place_type not in PLACE_FILTERS:
        return {"error": "Invalid place type. Use restaurant, hospital, attraction, or petrol pump."}

    try:
        geo = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers=HEADERS,
            timeout=10,
        )
        geo.raise_for_status()
        geo = geo.json()
    except (requests.RequestException, ValueError):
        return {"error": f"Could not find location information for {city}."}
    if not geo:
        return {"error": f"Could not find the city: {city}"}
    lat, lon = float(geo[0]["lat"]), float(geo[0]["lon"])

    clauses = "\n".join(
        f'nwr["{k}"="{v}"](around:{radius},{lat},{lon});' for k, v in PLACE_FILTERS[place_type]
    )
    query = f"[out:json][timeout:25];({clauses});out center tags 60;"

    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query},
                          headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return {"error": "OpenStreetMap service is temporarily unavailable. Please try again."}
        elements = r.json().get("elements", [])
    except requests.RequestException:
        return {"error": "Could not connect to OpenStreetMap. Please try again."}
    except ValueError:
        return {"error": "OpenStreetMap returned an invalid response."}

    places = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        center = el.get("center", {})
        p_lat = el.get("lat", center.get("lat"))
        p_lon = el.get("lon", center.get("lon"))
        addr = ", ".join(
            x for x in (tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city")) if x
        ) or "Address not available"
        places.append({"name": name, "address": addr, "lat": p_lat, "lon": p_lon})
        if len(places) >= limit:
            break

    if not places:
        return {"error": f"No named {place_type} places found near {city}."}
    return {"city": city, "type": place_type, "center": (lat, lon), "places": places}


# ---------------------------------------------------------------- agent tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""
    d = fetch_weather(city)
    if "error" in d:
        return f"Error: {d['error']}"
    return (f"Weather in {d['city']}: {d['desc']}, {d['temp']}°C "
            f"(feels like {d['feels_like']}°C), humidity {d['humidity']}%, wind {d['wind']} m/s")


@tool
def get_air_quality(city: str) -> str:
    """Get the current air quality and AQI for a city."""
    d = fetch_air_quality(city)
    if "error" in d:
        return d["error"]
    return (f"Air quality in {d['city']}:\nEuropean AQI: {d['eu_aqi']}\nUS AQI: {d['us_aqi']}\n"
            f"PM2.5: {d['pm25']} µg/m³\nPM10: {d['pm10']} µg/m³")


@tool
def search_places(city: str, place_type: str) -> str:
    """Search for restaurants, hospitals, petrol pumps, or attractions in a city using OpenStreetMap."""
    d = fetch_places(city, place_type)
    if "error" in d:
        return d["error"]
    body = "\n\n".join(f"Name: {p['name']}\nAddress: {p['address']}" for p in d["places"])
    return f"Places found in/near {city}:\n\n{body}"


@tool
def get_news(topic: str) -> str:
    """Search the web for the latest news about a given topic."""
    d = fetch_news(topic)
    if "error" in d:
        return d["error"]
    if not d["articles"]:
        return f"No latest news found about {topic}."
    return "\n\n".join(
        f"Title: {a['title']}\nSummary: {a['summary']}\nSource: {a['url']}" for a in d["articles"]
    )


ALL_TOOLS = [get_weather, get_news, get_air_quality, search_places]


# =====================================================================
# AGENT
# =====================================================================
SYSTEM_PROMPT = """
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
"""


def build_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b")
    return create_agent(llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


# =====================================================================
# UI
# =====================================================================
TOOL_LABELS = {
    "get_weather": "🌦 Checking the weather",
    "get_air_quality": "😷 Checking air quality",
    "get_news": "📰 Searching the news",
    "search_places": "📍 Searching places",
}
PLACE_TYPES = ["restaurant", "hospital", "attraction", "petrol pump"]


# ------------------------------------------------------------------ caching
@st.cache_resource
def get_agent():
    return build_agent()


@st.cache_data(ttl=600, show_spinner=False)
def cached_weather(city): return fetch_weather(city)

@st.cache_data(ttl=600, show_spinner=False)
def cached_air(city): return fetch_air_quality(city)

@st.cache_data(ttl=600, show_spinner=False)
def cached_news(city): return fetch_news(city)

@st.cache_data(ttl=600, show_spinner=False)
def cached_places(city, ptype): return fetch_places(city, ptype)


# ------------------------------------------------------------------ helpers
def aqi_band(aqi):
    if aqi is None:
        return "Unknown", "#888888"
    for limit, label, color in [
        (50, "Good", "#2e9e4f"), (100, "Moderate", "#d4b106"), (150, "Unhealthy for sensitive groups", "#e8832a"),
        (200, "Unhealthy", "#d93f3f"), (300, "Very unhealthy", "#8e44ad"),
    ]:
        if aqi <= limit:
            return label, color
    return "Hazardous", "#7b1e2b"


def run_agent(history):
    """Stream the agent, show tool activity live, return (final_text, tools_used)."""
    final, used = "", []
    with st.status("Thinking…", expanded=True) as status:
        try:
            for chunk in get_agent().stream({"messages": history}, stream_mode="updates"):
                for _node, data in chunk.items():
                    for msg in (data or {}).get("messages", []):
                        if isinstance(msg, AIMessage):
                            for call in msg.tool_calls:
                                used.append(call["name"])
                                st.write(f"{TOOL_LABELS.get(call['name'], call['name'])} · `{call['args']}`")
                            if not msg.tool_calls and msg.content:
                                c = msg.content
                                final = c if isinstance(c, str) else "".join(
                                    p.get("text", "") for p in c if isinstance(p, dict))
                        elif isinstance(msg, ToolMessage):
                            st.write(f"✅ Got data from `{getattr(msg, 'name', 'tool')}`")
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            return f"⚠️ I hit an error while answering: `{e}`. Please try again.", used
    return final.strip() or "I couldn't produce an answer. Please try rephrasing.", used


# ------------------------------------------------------------------ state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("city", "Indore")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🏙️ City Pulse")
    st.caption("Live weather, air quality, news and places for any city.")
    st.text_input("City", key="city")
    city = st.session_state.city.strip() or "Indore"

    st.markdown("**Quick actions**")
    place_type = st.selectbox("Place type", PLACE_TYPES)
    if st.button("🌦 Weather", use_container_width=True):
        st.session_state.pending = f"What's the weather in {city}?"
    if st.button("😷 Air quality", use_container_width=True):
        st.session_state.pending = f"How is the air quality in {city}? Is it safe to exercise outside?"
    if st.button("📰 Latest news", use_container_width=True):
        st.session_state.pending = f"Give me the latest news about {city}."
    if st.button("📍 Find places", use_container_width=True):
        st.session_state.pending = f"Show me {place_type} places in {city}."
    if st.button("✅ Good day to go out?", use_container_width=True):
        st.session_state.pending = f"Is it a good day to go out in {city}? Check weather and air quality."

    st.divider()
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ------------------------------------------------------------------ main
st.title("🏙️ City Pulse")
tab_chat, tab_dash = st.tabs(["💬 Chat", "📊 Dashboard"])

with tab_chat:
    if not st.session_state.messages:
        st.info(f"Hi! Ask me anything about **{city}** — weather, air quality, news, restaurants, hospitals, "
                "petrol pumps or attractions. Try one of these:")
        examples = [
            f"What's the weather in {city}?",
            f"Find hospitals in {city}",
            f"Is the air safe in {city} today?",
            f"Latest news in {city}",
        ]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex, use_container_width=True):
                st.session_state.pending = ex

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("tools"):
                st.caption("Tools used: " + ", ".join(dict.fromkeys(m["tools"])))

prompt = st.chat_input("Ask about a city…")
if st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

if prompt:
    with tab_chat:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        with st.chat_message("assistant"):
            answer, used = run_agent(history)
            st.markdown(answer)
            if used:
                st.caption("Tools used: " + ", ".join(dict.fromkeys(used)))
        st.session_state.messages.append({"role": "assistant", "content": answer, "tools": used})

with tab_dash:
    st.subheader(f"Live snapshot · {city}")
    w, a = cached_weather(city), cached_air(city)

    c1, c2, c3, c4 = st.columns(4)
    if "error" in w:
        c1.error(w["error"])
    else:
        c1.metric("Temperature", f"{w['temp']:.1f} °C", help=w["desc"].title())
        c2.metric("Feels like", f"{w['feels_like']:.1f} °C")
        c3.metric("Humidity", f"{w['humidity']} %")
        st.caption(f"Conditions: **{w['desc'].title()}**" + (f" · Wind {w['wind']} m/s" if w["wind"] is not None else ""))
    with c4:
        if "error" in a:
            st.error(a["error"])
        else:
            label, color = aqi_band(a["us_aqi"])
            st.metric("US AQI", a["us_aqi"])
            st.markdown(
                f"<span style='background:{color};color:white;padding:3px 10px;border-radius:12px;"
                f"font-size:0.85rem'>{label}</span>", unsafe_allow_html=True)
            st.caption(f"PM2.5 {a['pm25']} · PM10 {a['pm10']} µg/m³")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### 📰 News")
        news = cached_news(city)
        if "error" in news:
            st.error(news["error"])
        elif not news["articles"]:
            st.info("No recent news found.")
        else:
            for art in news["articles"]:
                with st.container(border=True):
                    st.markdown(f"**{art['title']}**")
                    st.write(art["summary"][:220] + ("…" if len(art["summary"]) > 220 else ""))
                    if art["url"]:
                        st.link_button("Read more", art["url"])

    with right:
        st.markdown(f"#### 📍 {place_type.title()} near centre")
        places = cached_places(city, place_type)
        if "error" in places:
            st.error(places["error"])
        else:
            df = pd.DataFrame(places["places"]).dropna(subset=["lat", "lon"])
            if not df.empty:
                st.map(df, latitude="lat", longitude="lon", size=30)
            with st.expander("List", expanded=True):
                for p in places["places"]:
                    st.markdown(f"**{p['name']}**  \n{p['address']}")