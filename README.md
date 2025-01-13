## Requirements
* Python 3 (consider installing with python virtual environment like pyenv)
* Python IDE of your choice (Visual Studio Code, PyCharm, etc.)
* Docker
* OpenAI API key
* Langchain API key (to use LangSmith for debugging)
* If you're using Windows, install Git Bash so that you can run the commands below 
(if using VSCode you can select Git Bash after starting the integrated terminal)

## Setup environment variables
Copy the file 'sample.env' and rename it to '.env'.  Add your API keys.

## Python command
You might need to use `python3` instead of `python` in the commands below, if you have more than one version of python installed.

## Create and activate a virtual environment
Create a virtual environment to install dependencies only in this project and not globally

`python -m venv .venv`

To activate the virtual environment:

`source .venv/bin/activate`

To verify virtual environment has been activated:

`which python`

## Install or Upgrade pip

If you're running Python 3.4+ you should already have pip installed. To check if you have pip:

`python -m pip --version`

### Install pip:

Follow installation instructions on pip's website (https://pip.pypa.io/en/stable/installation/#installation)

### Upgrade pip:
`python -m pip install --upgrade pip`

## Install packages
`python -m pip install -r requirements.txt`

## Start dependencies (postgres, pgadmin, etc.) in Docker
Get the docker image for postgres
`docker pull pgvector/pgvector:pg17`

To start containers:
`docker-compose up -d`

To stop and remove containers:
`docker-compose down`

## Log into pgadmin
* Go to http://localhost:8888
* Enter username: user@pgadmin.com and password: pgadmin

## Connect to database in pgadmin
* Servers > Add New Servers
* General tab
    * Name: pgvector
* Connection tab
    * Hostname: host.docker.internal
    * Port: 5440
    * Username: postgres
    * Password: codemash2025

After starting the application the following two tables will be created in the meal_planner database:
`select * from langchain_pg_collection;`
`select * from langchain_pg_embedding;`

## Run application
`python main.py`

## Exit application
Type `exit`

## Review calls to LLM using Langsmith
* Go to https://smith.langchain.com/ and login with your langchain account
* Select the 'default' project
* Click on a run to see more details including input, documents retrieved, and output

# Resources
* [Recipe dataset](https://www.kaggle.com/datasets/paultimothymooney/recipenlg/data)
* [OpenAI Models](https://platform.openai.com/docs/models) - details of models available and their cost
* [Build a simple LLM application](https://python.langchain.com/docs/tutorials/llm_chain/) - intro tutorial on using language models and prompt templates
* [Using a CSV Loader](https://python.langchain.com/docs/integrations/document_loaders/csv/)
* [Vector Stores Overview](https://python.langchain.com/docs/concepts/vectorstores/)
* [OpenAI Embeddings](https://python.langchain.com/docs/integrations/text_embedding/openai/) - uses in-memory vector store and shows the calls made by the vector store and retrievers under the hood to generate embeddings
* [Using langchain-postgres](https://github.com/langchain-ai/langchain-postgres/blob/main/examples/vectorstore.ipynb) - how to connect to PGVector store, add documents with content and metadata, and do a similarity search
* [Chat history](https://python.langchain.com/docs/versions/migrating_chains/conversation_retrieval_chain/#lcel) - create a rag chain that incorporates chat history
* Langchain docs
    * [ChatPromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html#chatprompttemplate)
    * [CSVLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.csv_loader.CSVLoader.html)
    * [PGVector](https://python.langchain.com/api_reference/postgres/vectorstores/langchain_postgres.vectorstores.PGVector.html#pgvector)


# Further reading
* [Build a Chatbot](https://python.langchain.com/docs/tutorials/chatbot/) - tutorial using LangGraph, which is not required to build a RAG application, but can simplify more complex applications.
* [Build a Retrieval Augmented Generation (RAG) App](https://python.langchain.com/docs/tutorials/rag/) - tutorial using LangGraph. Shows how to use a text splitter for handling long documents. 

