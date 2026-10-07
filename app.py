
import os
import uuid
import base64
from html import escape

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import requests

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tavily import TavilyClient

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="CityPulse AI",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling - keeps the CityPulse dark/glass design
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(91, 33, 182, .20), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(14, 165, 233, .14), transparent 25%),
            #070b14;
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: rgba(9, 13, 24, .92);
        border-right: 1px solid rgba(148, 163, 184, .12);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid rgba(148, 163, 184, .14);
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            rgba(30, 41, 59, .78),
            rgba(15, 23, 42, .58)
        );
        box-shadow: 0 20px 60px rgba(0, 0, 0, .25);
        margin-bottom: 24px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        letter-spacing: -.04em;
    }

    .hero p {
        color: #94a3b8;
        margin: 8px 0 0;
        font-size: 1rem;
    }

    .status-card {
        padding: 14px 16px;
        border: 1px solid rgba(148, 163, 184, .12);
        border-radius: 16px;
        background: rgba(15, 23, 42, .68);
        margin: 10px 0;
    }

    .approval-card {
        padding: 18px;
        border-radius: 18px;
        border: 1px solid rgba(245, 158, 11, .35);
        background: rgba(120, 53, 15, .16);
        margin: 14px 0;
    }

    .approval-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #fbbf24;
    }

    .approval-tool {
        margin-top: 7px;
        color: #e2e8f0;
    }

    .tool-args {
        color: #94a3b8;
        font-size: .9rem;
        margin-top: 4px;
        word-break: break-word;
    }

    .feature {
        padding: 14px;
        border: 1px solid rgba(148, 163, 184, .12);
        border-radius: 16px;
        background: rgba(15, 23, 42, .55);
        margin-bottom: 10px;
    }

    .small-muted {
        color: #64748b;
        font-size: .82rem;
    }

    .voice-box {
        padding: 12px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, .12);
        background: rgba(15, 23, 42, .45);
    }

    button[kind="primary"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Tools
# ---------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Weather error: OPENWEATHER_API_KEY is missing."

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if str(data.get("cod")) != "200":
            return f"Weather error: {data.get('message', 'Could not fetch weather')}"

        temp = data["main"]["temp"]
        feels_like = data["main"].get("feels_like")
        desc = data["weather"][0]["description"].capitalize()
        humidity = data["main"].get("humidity")

        return (
            f"Weather in {city}: {desc}, {temp}°C. "
            f"Feels like {feels_like}°C. Humidity: {humidity}%."
        )

    except requests.RequestException as exc:
        return f"Weather error: {exc}"


@tool
def get_news(topic: str) -> str:
    """Search the web for the latest news about a given topic."""
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "News error: TAVILY_API_KEY is missing."

    try:
        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(
            query=f"latest news about {topic}",
            topic="news",
            max_results=5,
        )

        results = response.get("results", [])

        if not results:
            return f"No latest news found about {topic}."

        news = []
        for result in results:
            news.append(
                f"Title: {result.get('title', 'No title')}\n"
                f"Summary: {result.get('content', 'No description')}\n"
                f"Source: {result.get('url', '')}"
            )

        return "\n\n".join(news)

    except Exception as exc:
        return f"News error: {exc}"


# ---------------------------------------------------------
# Agent + real LangChain HITL
#
# IMPORTANT:
# Streamlit reruns after button clicks. The checkpointer + thread_id
# preserves the paused agent state so we can resume the SAME run.
# ---------------------------------------------------------
@st.cache_resource
def build_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b")

    return create_agent(
        llm,
        tools=[get_weather, get_news],
        system_prompt=(
            "You are CityPulse AI, a helpful city information assistant. "
            "Use get_weather for current weather and get_news for latest news. "
            "Use tools when live/current information is requested. "
            "Be concise, accurate, and never invent live information."
        ),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "get_weather": {
                        "allowed_decisions": ["approve", "reject"]
                    },
                    "get_news": {
                        "allowed_decisions": ["approve", "reject"]
                    },
                },
                description_prefix="CityPulse AI wants to use a live tool",
            )
        ],
        checkpointer=InMemorySaver(),
    )


agent = build_agent()

# ---------------------------------------------------------
# Runnable
# ---------------------------------------------------------
def clean_response(response):
    if response is None:
        return ""

    if isinstance(response, list):
        parts = []
        for block in response:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
        response = "\n".join(parts)

    return str(response).strip()


response_runnable = RunnableLambda(clean_response)


def extract_text_from_message(message):
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts).strip()

    return str(content).strip() if content is not None else ""


def extract_final_response(result):
    messages = result.get("messages", []) if isinstance(result, dict) else []

    for message in reversed(messages):
        msg_type = getattr(message, "type", None)
        text_value = extract_text_from_message(message)

        if msg_type == "ai" and text_value:
            return text_value

    for message in reversed(messages):
        text_value = extract_text_from_message(message)
        if text_value:
            return text_value

    return "I could not generate a text response."


def get_interrupt_request(result):
    interrupts = result.get("__interrupt__", []) if isinstance(result, dict) else []

    if not interrupts:
        return None

    interrupt = interrupts[0]

    # LangGraph Interrupt objects expose the payload through .value.
    value = getattr(interrupt, "value", interrupt)

    if isinstance(value, dict):
        actions = value.get("action_requests", value.get("actionRequests", []))
        reviews = value.get("review_configs", value.get("reviewConfigs", []))

        if actions:
            action = actions[0]
            review = reviews[0] if reviews else {}

            return {
                "name": action.get("name", "unknown_tool"),
                "args": action.get("args", action.get("arguments", {})),
                "description": action.get(
                    "description",
                    "This tool call requires your approval.",
                ),
                "allowed_decisions": review.get(
                    "allowed_decisions",
                    review.get("allowedDecisions", ["approve", "reject"]),
                ),
            }

    return {
        "name": "tool",
        "args": {},
        "description": str(value),
        "allowed_decisions": ["approve", "reject"],
    }


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
defaults = {
    "messages": [],
    "thread_id": str(uuid.uuid4()),
    "pending_hitl": None,
    "last_assistant_text": "",
    "voice_text": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌆 CityPulse AI")
    st.caption("Your intelligent city assistant")

    st.markdown("---")

    st.markdown("### Features")

    st.markdown(
        '<div class="feature">🌦️ <b>Live Weather</b><br>'
        '<span class="small-muted">Current weather through OpenWeather</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="feature">📰 <b>Latest News</b><br>'
        '<span class="small-muted">Fresh web results through Tavily</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="feature">🛡️ <b>Human Approval</b><br>'
        '<span class="small-muted">Approve or reject live tool calls</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="feature">🔗 <b>Runnable Pipeline</b><br>'
        '<span class="small-muted">Post-processes the final response</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="feature">🎙️ <b>Speech Input</b><br>'
        '<span class="small-muted">Talk to CityPulse using your microphone</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="feature">🔊 <b>Read Aloud</b><br>'
        '<span class="small-muted">Listen to the assistant response</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_hitl = None
        st.session_state.last_assistant_text = ""
        st.session_state.voice_text = ""
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.caption("Powered by LangChain + Groq + OpenWeather + Tavily")


# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌆 CityPulse AI</h1>
        <p>
            A tool-using city assistant with human approval,
            Runnables, live weather, latest news, and voice interaction.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Show chat history
# ---------------------------------------------------------
for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

        if item["role"] == "assistant" and item["content"].strip():
            # Browser speech is generated by JavaScript below.
            safe_text = escape(item["content"])
            components.html(
                f"""
                <script>
                function speakText(text) {{
                    if ("speechSynthesis" in window) {{
                        window.speechSynthesis.cancel();
                        

                        const utterance = new SpeechSynthesisUtterance(text);

                        const voices = window.speechSynthesis.getVoices();

                        utterance.voice = voices.find(
                            voice =>
                                voice.name.toLowerCase().includes("female") ||
                                voice.name.toLowerCase().includes("zira")
                        );

                        utterance.rate = 1.0;
                        utterance.pitch = 1.1;
                        utterance.volume = 1.0;

                        window.speechSynthesis.speak(utterance);
                    }}
                }}
                </script>
                """,
                height=0,
            )


# ---------------------------------------------------------
# HITL approval panel
# ---------------------------------------------------------
pending = st.session_state.pending_hitl

if pending:
    args_text = escape(str(pending.get("args", {})))
    tool_name = escape(str(pending.get("name", "unknown_tool")))
    description = escape(str(pending.get("description", "")))

    st.markdown(
        f"""
        <div class="approval-card">
            <div class="approval-title">🛡️ Human approval required</div>
            <div class="approval-tool">
                CityPulse AI wants to call <b>{tool_name}</b>
            </div>
            <div class="tool-args"><b>Arguments:</b> {args_text}</div>
            <div class="tool-args">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        approve = st.button(
            "✅ Allow Tool",
            use_container_width=True,
            type="primary",
            key="approve_tool",
        )

    with col2:
        reject = st.button(
            "❌ Reject Tool",
            use_container_width=True,
            key="reject_tool",
        )

    if approve or reject:
        decision_type = "approve" if approve else "reject"

        try:
            resumed = agent.invoke(
                Command(
                    resume={
                        "decisions": [
                            {
                                "type": decision_type,
                            }
                        ]
                    }
                ),
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                },
            )

            st.session_state.pending_hitl = get_interrupt_request(resumed)

            if st.session_state.pending_hitl is None:
                final_text = response_runnable.invoke(
                    extract_final_response(resumed)
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_text,
                    }
                )
                st.session_state.last_assistant_text = final_text

            st.rerun()

        except Exception as exc:
            st.error(f"Approval/resume error: {exc}")


# ---------------------------------------------------------
# Voice input
#
# Streamlit's native audio_input records microphone audio.
# Speech recognition is performed locally with SpeechRecognition.
# ---------------------------------------------------------
st.markdown("### 🎙️ Voice input")

voice_col1, voice_col2 = st.columns([3, 1])

with voice_col1:
    audio_value = st.audio_input(
        "Record your question",
        sample_rate=16000,
        key="citypulse_audio",
    )

with voice_col2:
    st.markdown(
        '<div class="voice-box">'
        '<b>Tip</b><br>'
        '<span class="small-muted">'
        'Record your question, then transcribe it below.'
        '</span></div>',
        unsafe_allow_html=True,
    )

if audio_value:
    if st.button("📝 Transcribe Voice", use_container_width=True):
        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()

            with st.spinner("Transcribing your voice..."):
                audio_bytes = audio_value.getvalue()

                with open("citypulse_voice.wav", "wb") as f:
                    f.write(audio_bytes)

                with sr.AudioFile("citypulse_voice.wav") as source:
                    audio_data = recognizer.record(source)

                text_result = recognizer.recognize_google(audio_data)

            st.session_state.voice_text = text_result
            st.success("Voice converted to text.")

        except ImportError:
            st.error(
                "SpeechRecognition is not installed. "
                "Run: pip install SpeechRecognition"
            )
        except sr.UnknownValueError:
            st.warning("I could not understand the recording. Please try again.")
        except sr.RequestError as exc:
            st.error(f"Speech recognition service error: {exc}")
        except Exception as exc:
            st.error(f"Voice transcription error: {exc}")


if st.session_state.voice_text:
    st.markdown("**Recognized speech:**")
    st.info(st.session_state.voice_text)

    if st.button("💬 Send Voice Message", type="primary"):
        prompt_from_voice = st.session_state.voice_text
        st.session_state.voice_text = ""

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt_from_voice,
            }
        )

        try:
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt_from_voice,
                        }
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                },
            )

            pending_request = get_interrupt_request(result)

            if pending_request:
                st.session_state.pending_hitl = pending_request
            else:
                final_text = response_runnable.invoke(
                    extract_final_response(result)
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_text,
                    }
                )
                st.session_state.last_assistant_text = final_text

            st.rerun()

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")


# ---------------------------------------------------------
# Normal text chat
# ---------------------------------------------------------
prompt = st.chat_input("Ask CityPulse AI about weather, news, or your city...")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            },
        )

        pending_request = get_interrupt_request(result)

        if pending_request:
            st.session_state.pending_hitl = pending_request
        else:
            final_text = response_runnable.invoke(
                extract_final_response(result)
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_text,
                }
            )
            st.session_state.last_assistant_text = final_text

        st.rerun()

    except Exception as exc:
        st.error(f"Something went wrong: {exc}")


# ---------------------------------------------------------
# Read Aloud
# Browser-native SpeechSynthesis = no TTS API key required.
# ---------------------------------------------------------
last_text = st.session_state.last_assistant_text

if last_text:
    st.markdown("### 🔊 Voice output")

    speech_text_b64 = base64.b64encode(
        last_text.encode("utf-8")
    ).decode("ascii")

    components.html(
        f"""
        <div style="
            padding:14px;
            border:1px solid rgba(148,163,184,.14);
            border-radius:16px;
            background:rgba(15,23,42,.55);
            font-family:Arial,sans-serif;
            color:#e2e8f0;
        ">
            <div style="margin-bottom:10px;font-weight:600;">
                Listen to the latest response
            </div>

            <button id="speak"
                style="
                    border:0;
                    border-radius:10px;
                    padding:9px 14px;
                    margin-right:7px;
                    cursor:pointer;
                    background:#7c3aed;
                    color:white;
                    font-weight:600;
                ">
                🔊 Read Aloud
            </button>

            <button id="stop"
                style="
                    border:0;
                    border-radius:10px;
                    padding:9px 14px;
                    cursor:pointer;
                    background:#334155;
                    color:white;
                    font-weight:600;
                ">
                ⏹ Stop
            </button>

            <script>
                const text = decodeURIComponent(
                    escape(atob("{speech_text_b64}"))
                );

                const speakButton = document.getElementById("speak");
                const stopButton = document.getElementById("stop");

                speakButton.onclick = () => {{
                    if (!("speechSynthesis" in window)) {{
                        alert("Speech synthesis is not supported by this browser.");
                        return;
                    }}

                    window.speechSynthesis.cancel();

                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.rate = 1.0;
                    utterance.pitch = 1.0;
                    utterance.volume = 1.0;

                    window.speechSynthesis.speak(utterance);
                }};

                stopButton.onclick = () => {{
                    if ("speechSynthesis" in window) {{
                        window.speechSynthesis.cancel();
                    }}
                }};
            </script>
        </div>
        """,
        height=110,
    )

st.markdown(
    '<div class="small-muted" style="text-align:center;margin-top:30px;">'
    'CityPulse AI • Agent + Tools + Human-in-the-Loop + Runnables + Voice'
    '</div>',
    unsafe_allow_html=True,
)
