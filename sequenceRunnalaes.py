from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words."
)
model = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

parser = StrOutputParser()

formeted_prompt = prompt.format_messages(topic = "Machine Learning")

response = model.invoke(formeted_prompt)
# RunnableSequence
chain = prompt | model | parser

result = chain.invoke({
    "topic": "Machine Learning"
})

print(result)