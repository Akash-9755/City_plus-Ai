from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rich import print


# creating a tool

@tool
def text_to_length(text : str) -> int:
    """Return the number f character in given text"""
    return len(text)

llm = ChatGroq(
    model = "openai/gpt-oss-120b"
)

# tool binding 

llm_with_tool = llm.bind_tools([text_to_length])
message = []
prompt = input("you : ")
query = HumanMessage(prompt)
message.append(query)


result = llm_with_tool.invoke(message)
message.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tool[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
   

result = llm_with_tool.invoke(message)
print(result.content)
