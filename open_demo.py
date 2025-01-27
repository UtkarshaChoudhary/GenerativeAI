import os
from langchain_openai import ChatOpenAI


OPENAI_API_KEY =os.getenv("OPENAI_API_KEY")

llm=ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)


question =input("Enter the question")
response =llm.invoke(question)
print(response)

