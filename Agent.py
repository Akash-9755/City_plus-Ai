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
    tools = [get_weather,get_news],
    system_prompt= "you are a helpful city assistant.",
    middleware= [human_approval]
)

print("City Agent | type exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break 
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    final_response = result["messages"][-1].content

    cleaned_response = response_runnable.invoke(final_response)

    print("bot : ", cleaned_response)
    
    