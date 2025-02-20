import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from langchain_community.document_loaders import PyPDFLoader

pdf=PyPDFLoader("C:/Users/Utkarsha/PycharmProjects/GenAI using RAG.pdf")
documents = loader.load()

pdfpages = pdf.load_and_split()

from langchain_community. vectorstores import FAISS
#from langchain_groq.embeddings import GroqEmbeddings
import os
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"]=GROQ_API_KEY

mybooks=pdf.load()

text_splitter=CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)

split_text=text_splitter.split_documents(mybooks)

embeddings=GroqEmbeddings()
vectorstore=FAISS.from_documents(split_text,embeddings)

vectorstore

vectorstore_retriever = vectorstore.as_retriever()

from langchain.agents.agent_toolkits import create_retriever_tool

tool=create_retriever_tool(
    "vectorstore_retriever",
    "Retrieve detailed information on the chapters of the book 'Ikigai'"
)

tools=[tool]
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_groq.chat_models import ChatGroq

llm= ChatGroq(temperature=0,model_name="llama-3.3-70b-versatile")

myagent=create_conversational_retrieval_agent(llm, tools, verbose=True)

context="The user is conducting reaserch on the book"
question="who write book in detailed in 200 words"

prompt=f""" Your need to answer the question in the sentence as same as in the pdf content. Given below is the context and question of the user.
context={context}
question={question}

"""

result=myagent.invoke({"input": prompt})