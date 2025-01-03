from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain


# load environment variables
env_found = load_dotenv(override=True, verbose=True)

def get_response(input, chat_history):

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

    # System message to request that the user's question be reformulated as a standalone question
    # using the new question and any previous chat history that is relevant to the question
    contextualize_system_prompt =  """Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question
    which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is."""

    # Create a prompt template containing the contextualize system message and conversation messages
    contextualize_prompt = ChatPromptTemplate(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # Construct a chain that accepts input and chat history
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # System message to indicate what task the LLM needs to perform and the context to use (relevant recipes retrieved from our list)
    qa_system_prompt = """
        You are a a helpful meal planner. Your job is to suggest meals based on the user's input, chat history, and the context below. 
        If the user requests a meal type you're not familiar with, first obtain information about the meal type, then suggest meals.

        Respond only with recipes included in the context; do not add recipes from other sources.
        
        If the user's question has nothing to do with recipes, meals, or food, then reply 'Sorry, I don't have the information requested.'
        
        {context}
    """

    # Create a prompt template containing the Q&A system message and conversation messages
    qa_prompt = ChatPromptTemplate(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # Create a chain that uses relevant recipes (context) to answer questions
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a chain that applies the history retriever and Q&A chain in sequence retaining intermediate outputs such
    # as the context
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    response = rag_chain.invoke(
        {
            "input": input,
            "chat_history": chat_history
        }
    )

    # Add the current input and llm response to the chat history
    chat_history.extend([HumanMessage(content=input), response["answer"]])

    return response["answer"]
  

print("I'm the meal planner chatbot. How can I help you today?")

chat_history = []

while True:
    user_input = input ("You: ")
    if user_input.lower() == "exit":
        break
    response = get_response(user_input, chat_history)
    print(f"AI: {response}")


