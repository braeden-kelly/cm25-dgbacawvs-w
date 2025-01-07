import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# Create an OpenAI chat model
def initialize_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Parse CSV file with recipes into a list of documents that can be loaded into a vector store
# Content columns will be included when calculating embeddings
def load_recipes():
    file = "./data/sample_recipes.csv"
    loader = CSVLoader(
        file_path=file,
        content_columns=["description"],
        metadata_columns=["id", "name", "url"],
        source_column="url"
    )
    documents = loader.load()
    return documents

# Initialize embedding model and create a vector store in pgvector that contains the recipes
def initialize_vector_store(documents):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db_url = os.environ["DB_URL"]
    collection_name = "recipes"

    vector_store = PGVector(
        embeddings=embedding_model,
        collection_name=collection_name,
        connection=db_url,
        use_jsonb=True
    )

    # Adding documents by ID will overwrite any existing documents with the same ID during startup
    vector_store.add_documents(documents=documents, ids=[doc.metadata["id"] for doc in documents])

    return vector_store.as_retriever()

# Create a chain that uses the chat history and the user's question to create a “standalone question”. 
# This question is then passed into the retrieval step to fetch relevant documents (context). 
def create_chat_history_aware_retriever(llm, vector_store_retriever):
    rephrase_system_prompt =  """Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question
    which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is."""

    rephrase_prompt = ChatPromptTemplate(
        [
            ("system", rephrase_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    return create_history_aware_retriever(llm, vector_store_retriever, rephrase_prompt)

# Create a chain that takes relevant documents (context), user question, and chat history to return a relevant response
# The system prompt is also used to define the role of the llm and any specific instructions
def create_qa_chain(llm):
    qa_system_prompt = """
        You are a a helpful meal planner. Your job is to suggest meals based on the user's input, chat history, and the context below. 
        If the user requests a meal type you're not familiar with, first obtain information about the meal type, then suggest meals.

        Respond only with recipes included in the context; do not add recipes from other sources.
        
        If the user's question has nothing to do with recipes, meals, or food, then reply 'Sorry, I don't have the information requested.'
        
        {context}
    """

    qa_prompt = ChatPromptTemplate(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    return create_stuff_documents_chain(llm, qa_prompt) 

def get_response(rag_chain, input, chat_history):
    response = rag_chain.invoke(
        {
            "input": input,
            "chat_history": chat_history
        }
    )

    # Add the current input and llm response to the chat history
    chat_history.extend([HumanMessage(content=input), response["answer"]])

    return response["answer"]
  

# Load API keys from .env
env_found = load_dotenv(override=True, verbose=True)

llm = initialize_llm()

documents = load_recipes()
vector_store_retriever = initialize_vector_store(documents)

history_aware_retriever = create_chat_history_aware_retriever(llm, vector_store_retriever)
qa_chain =  create_qa_chain(llm)

# Create a chain that applies the history aware retriever and Q&A chains in sequence, retaining intermediate outputs such
# as the context
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

print("I'm the meal planner chatbot. How can I help you today?")

chat_history = []
while True:
    user_input = input ("You: ")
    if user_input.lower() in ("exit", "quit"):
        break
    response = get_response(rag_chain, user_input, chat_history)
    print(chat_history)
    print(f"AI: {response}")
