from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Create an OpenAI chat model
def initialize_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Answer the user's question using the llm
def get_response(llm, user_input):
    system_message = """You're a helpful librarian and translator. Answer the user's question in the following format:
    <answer to the question>

    German translation: 
    <answer to question in German>

    Shakespeare: 
    <provide the answer using Shakespeare's way of writing>
    """

    messages = [
            ("system", system_message),
            HumanMessage(user_input)
        ]

    response = llm.invoke(messages)    

    return response.content


# Load API keys from .env
env_found = load_dotenv(override=True, verbose=True)

llm = initialize_llm()

print("Hi! Welcome to Demystifying GenAI. How can I help you today?\n")

while True:
    user_input = input ("You: ")
    if user_input.lower() in ("exit", "quit"):
        break
    response = get_response(llm, user_input)
    print(f"AI: {response}\n")