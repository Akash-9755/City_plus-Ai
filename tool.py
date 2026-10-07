from langchain.tools import tool

@tool
def get_greeting(name : str) -> str:
    "Genrate a greeting massage from user"
    
    return f"Hellow {name} ,Welcome to the ai word"

result = get_greeting.invoke({"name":"Akash"})

print(result)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)