This project demonstrates a Generative AI application that converts text into chunks, generates embeddings, and builds a user-friendly interface to interact with the processed data. Users can upload a PDF, and the system allows them to ask questions about the content, generating accurate answers based on the uploaded file.

Key Features
Text to Chunks:

Extracts text from the uploaded PDF.
Splits the text into manageable chunks using advanced text-splitting techniques.
Chunks to Embeddings:

Generates embeddings for the text chunks using a pre-trained embedding model.
Stores embeddings in a vector database for efficient similarity search.
User Interface:

Developed using Python libraries such as Streamlit to create an interactive UI.
Features a question box where users can input queries to receive contextually accurate answers.
Question-Answering System:

Leverages embeddings and similarity search to find relevant chunks.
Uses an LLM (e.g., OpenAI or an equivalent API) to generate answers from the most relevant chunks.
Libraries and Tools Used
Python: Core programming language for development.
PyPDF2: For extracting text from PDFs.
LangChain: For text splitting, embeddings generation, and question-answering chain.
FAISS: To build and manage the vector store for similarity search.
Streamlit: To design and deploy an intuitive web-based user interface.
How It Works
Upload a PDF document through the UI.
Extract and split the PDF content into chunks of text.
Generate embeddings for these chunks using a pre-trained model.
Store the embeddings in a FAISS vector database.
Type a question in the input box to retrieve answers based on the uploaded document.
