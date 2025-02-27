from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from langchain_community.chat_models import ChatOllama
import streamlit as st


GROQ_API_KEY= os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="llama-3.3-70b-versatile",api_key= GROQ_API_KEY)

st.title("Ask Anything :)")

question = st.text_input("Enter the question:")

if question:
    response=llm.invoke(question)
    st.write(response.content)