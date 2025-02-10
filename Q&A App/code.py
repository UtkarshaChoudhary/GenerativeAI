from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

GROQ_API_KEY= os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

question = input("Enter the question")
response = llm.invoke(question)
print(response.content)