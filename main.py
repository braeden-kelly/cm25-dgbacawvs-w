
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate



env_found = load_dotenv(override=True, verbose=True)
llm = ChatOpenAI(model="gpt-4o-mini")


def get_response(input):
    prompt = PromptTemplate.from_template("""Given the following user question, answer the user question.  
    If you cannot find results, just say 'No results were found' 
    
    Input: {input}
    Answer: """)

    chain = (prompt | llm | StrOutputParser())

    return chain.invoke({"input": input})


print("I'm the meal planner chatbot. How can I help?")

while True:
    user_input = input ("You: ")
    if user_input.lower() == "exit":
        break
    response = get_response(user_input)
    print(response)


