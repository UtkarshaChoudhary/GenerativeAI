from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from langchain_community.chat_models import ChatOllama

GROQ_API_KEY= os.getenv("GROQ_API_KEY")

llm=ChatOllama(model="gemma:2b")

question = input("Enter the question")
response = llm.invoke(question)
print(response.content)