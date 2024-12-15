## Requirements
* Python 3
* Python IDE of your choice (Visual Studio Code, PyCharm, etc.)
* OpenAI API key
* Langchain API key (to use LangSmith for debugging)


## Create and activate a virtual environment
Create a virtual environment to install dependencies only in this project and not globally

`python3 -m venv .venv`

To activate the virtual environment:

`source .venv/bin/activate`

To verify virtual environment has been activated:

`which python`

## Install or Upgrade pip

If you're running Python 3.4+ you should already have pip installed. To check if you have pip:

`python3 -m pip --version`

### Install pip:

Follow installation instructions on pip's website (https://pip.pypa.io/en/stable/installation/#installation)


### Upgrade pip:

`python3 -m pip install --upgrade pip`

## Setup environment variables
Copy the file 'sample.env' and rename it to '.env'.  Add your API keys.

## Install packages
`python3 -m pip install -r requirements.txt`

## Run application
`python3 main.py`

## Exit application
Type `exit`