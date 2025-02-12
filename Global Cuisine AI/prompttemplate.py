from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from langchain_community.chat_models import ChatOllama
import streamlit as st
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate

set_debug(True)

GROQ_API_KEY= os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="llama-3.3-70b-versatile",api_key= GROQ_API_KEY)
prompt_template = PromptTemplate(
    input_variables=["Country","no_of_paras","language"],
    template="""You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country.
    Avoid giving information about fictional places. If the country is fictional 
    or non-existent answer: I don't know.
    Answer the {no_of_paras} short paras in {language}"""
)
st.title("Cuisine Info :)")

country = st.text_input("Enter the country:")
no_of_paras = st.number_input("Enter the number of paras", min_value=1, max_value=5)
language = st.text_input("Enter the language:")

if country:
    response=llm.invoke(prompt_template.format(country=country,
                                               no_of_paras=no_of_paras,
                                               language=language))
    st.write(response.content)