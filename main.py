from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate


# load environment variables
env_found = load_dotenv(override=True, verbose=True)

def get_response(input):

    # Parse CSV file with recipes into documents
    file = "./data/sample_recipes.csv"
    loader = CSVLoader(file_path=file)
    documents = loader.load()

    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create an in-memory vector store using embedding model
    vector_store = InMemoryVectorStore(embedding=embedding_model)

    # Add recipes to the vector store
    vector_store.add_documents(documents)

    # Search for meals based on user input
    retriever = vector_store.as_retriever()
    relevant_recipes = retriever.invoke(input)

    # Set up LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Create a prompt template
    template = ChatPromptTemplate([
        ("system", f"""
         
         You are a a helpful meal planner. 
         
         Your job is to suggest meals based on the recipes and user input. 

         If the user input is a question or request not related to generating a list of meals, reply 'Sorry I can't answer. I can only provide meal suggestions.'
        
         Recipes: {{relevant_recipes}}
         """),
         ("human", "{user_input}")
    ])
    
    # Generate prompt using user input and relevant recipes
    prompt = template.invoke(
        {
        "relevant_recipes": relevant_recipes,
        "user_input": input
        })

    # Get response from LLM
    response = llm.invoke(prompt)

    return response.content
  

print("I'm the meal planner chatbot. How can I help you today?")

while True:
    user_input = input ("You: ")
    if user_input.lower() == "exit":
        break
    response = get_response(user_input)
    print(f"AI: {response}")


