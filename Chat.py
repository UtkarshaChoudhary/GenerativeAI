import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# OpenAI API Key
OPENAI_API_KEY = "*****Your Open key*********"

# Initialize embeddings
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    st.write("Embeddings initialized successfully.")
except Exception as e:
    st.error(f"Error initializing embeddings: {e}")
    st.stop()

# Upload PDF files
st.header("My First ChatBot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF & start questions", type="pdf")

# Extract text from the uploaded PDF
text = ""
if file is not None:
    reader = PdfReader(file)
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text

    if not text:
        st.error("Failed to extract text. The PDF might be empty or non-readable.")
        st.stop()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_text(text)

# Ensure chunks are valid
if not chunks or not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
    st.error("Chunks must be a non-empty list of strings.")
    st.stop()

# Display chunks for debugging
st.write("Chunks:", chunks[:5])  # Show first 5 chunks for verification

# Generate embeddings and create vector store
try:
    vector_store = FAISS.from_texts(chunks, embeddings)
    st.success("Vector store created successfully!")
except Exception as e:
    st.error(f"Error creating vector store: {e}")
    st.stop()

# Get user question
user_question = st.text_input("Type Your Question here")

# Perform similarity search and respond
if user_question:
    try:
        match = vector_store.similarity_search(user_question)
        st.write(match)

        # Define the LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        # Run the QA chain
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=matches, question=user_question)
        st.write(response)
    except Exception as e:
        st.error(f"Error during question answering: {e}")
