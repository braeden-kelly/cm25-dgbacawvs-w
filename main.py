from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage


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

    # This template will be used to create a system message that
    # indicates to the LLM what task to perform and the context to use (relevant recipes retrieved from our list)
    system_template = """
        You are a a helpful meal planner. Your job is to suggest meals based on the user's questions and the context below. 
        
        If the user's question has nothing to do with recipes, meals, or food, then reply 'Sorry, I don't have the information requested.'
        
        Context: ```{context}```
    """

    # Create a prompt template containing the system message and conversation messages
    prompt = ChatPromptTemplate(
        [
            ("system", system_template),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    # Create a chain that uses relevant recipes (context) to answer questions
    document_chain = create_stuff_documents_chain(llm, prompt)
    response = document_chain.invoke(
        {
            "context": relevant_recipes,
            "messages": [
                HumanMessage(content=input)
            ]
        }
    )

    return response
  

print("I'm the meal planner chatbot. How can I help you today?")

while True:
    user_input = input ("You: ")
    if user_input.lower() == "exit":
        break
    response = get_response(user_input)
    print(f"AI: {response}")


