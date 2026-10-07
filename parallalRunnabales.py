import warnings

warnings.filterwarnings(
    "ignore",
    message="Direct use of automatic function calling.*"
)
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
    "Explain the {topic} in 1 to 2 line"
)

long_prompt = ChatPromptTemplate.from_template(
    "Explain the {topic} in details"
)

chain = RunnableParallel(
    {
        "short" : short_prompt | model | parser,
        "long" : long_prompt | model | parser
    }
)

result = chain.invoke({
    "topic" : "Machine Learning"
})

print("SHORT ANSWER:")
print(result["short"])

print("\nLONG ANSWER:")
print(result["long"])