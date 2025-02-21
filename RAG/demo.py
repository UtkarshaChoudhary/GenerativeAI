import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
#from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq.chat_models import ChatGroq
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load PDF
loader = PdfReader(r"Ikigai.pdf")
documents = loader.load()

# Split the text
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
split_text = text_splitter.split_documents(documents)

# Use HuggingFace for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_text, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()

# Create retriever tool
tool = create_retriever_tool(
    retriever=retriever,
    name="vectorstore_retriever",
    description="Retrieve detailed information on the chapters of the book 'Ikigai'"
)

# Define LLM
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

# Create agent
myagent = create_conversational_retrieval_agent(llm, tools=[tool], verbose=True)

# Define question and context
context = "The user is conducting research on the book."
question = "Who wrote the book in detail (200 words)?"

# Ask the agent
input_data = f"{context}\n\n{question}"
result = myagent.invoke({"input": input_data})

# Print result
print(result)
