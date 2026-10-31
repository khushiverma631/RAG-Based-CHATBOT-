
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load the API key from the project's .env file
load_dotenv() #loads variables from your .env file.

api_key = os.getenv("GROQ_API_KEY")  #retrieves the key without hardcoding it in Python.

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

# Initialize the Groq chat model
#configures the model connection.
llm = ChatGroq(  
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_tokens=200,
)

# Send a test question

#prints the generated answer.
#llm invoke = sends a question to the model.
response = llm.invoke(
    "Explain Retrieval-Augmented Generation in three simple sentences."
)

print("\nGroq response:\n")
print(response.content)
