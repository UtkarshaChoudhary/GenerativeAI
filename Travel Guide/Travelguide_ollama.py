from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
#import os
from langchain_community.chat_models import ChatOllama
import streamlit as st
#from langchain.globals import set_debug
from langchain.prompts import PromptTemplate

#set_debug(True)

#GROQ_API_KEY= os.getenv("GROQ_API_KEY")

llm=ChatOllama(model="llama3.2")
prompt_template = PromptTemplate(
    input_variables=["city","month","language","budget"],
    template=f"""Welcome to the city travel guide!
    If you're visiting in month, here's what you can do:
    1. Must-visit attractions.
    2. Local cuisine you must try.
    3. Useful phrases in language.
    4. Tips for traveling on a budget.
    Enjoy your trip!
    """
)
st.title("Travel Guide :)")

city = st.text_input("Enter the city:")
month = st.text_input("Enter the month of travel")
language = st.text_input("Enter the language:")
budget = st.selectbox("Travel Budget",["Low","Medium","High"])

if city and month and language and budget:
    prompt = prompt_template.format(city=city,
                                    month=month,
                                    language=language,
                                    budget=budget)
    response = llm.invoke(prompt)
    st.write(response.content)