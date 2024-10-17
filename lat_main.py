# Required Imports
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from PyPDF2 import PdfReader

# API Keys and Configuration
pinecone_api_key = '47c3d32b-6fcc-4874-a6bd-cb392f37a9b3'
pinecone_env = 'us-east-1'

# Initialize the Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# Define the index name
index_name = 'langchainchatbot'

# Create or connect to the Pinecone index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Initialize the Pinecone index for querying
index = pc.Index("langchainchatbot")

# Function to process PDF content
def process_pdf_file(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to index the document content into Pinecone
def index_document_content(content):
    # Split the content into chunks for better handling
    chunks = [content[i:i + 500] for i in range(0, len(content), 500)]  # Split into 500-character chunks

    # Iterate over each chunk and encode + upsert into Pinecone
    for idx, chunk in enumerate(chunks):
        chunk_embedding = model.encode(chunk).tolist()
        # Use upsert to add the chunk with a unique ID
        index.upsert([(f"chunk_{idx}", chunk_embedding, {'text': chunk})])

# Function to Query Pinecone for Closest Matches (QA Function)
def answer_question(question):
    question_embedding = model.encode(question).tolist()
    # Query the Pinecone index with the encoded question
    result = index.query(vector=question_embedding, top_k=3, include_metadata=True)
    # Extract the top matches and return the relevant content
    answers = [match['metadata']['text'] for match in result['matches']]
    return "\n\n".join(answers)

# Streamlit App Interface
st.title("Pinecone QA_priyanshu Chatbot ")

# File upload interface
uploaded_file = st.file_uploader("Upload a PDF file:", type=['pdf'])

# Process and index the PDF file content
if uploaded_file:
    # Extract text from the uploaded PDF
    file_content = process_pdf_file(uploaded_file)
    if file_content:
        st.write("PDF content extracted successfully.")
        
        # Index the content into Pinecone
        index_document_content(file_content)
        st.write("PDF content has been indexed into Pinecone database.")
    else:
        st.write("Failed to extract content from the PDF file.")

# Input field for user question
user_input = st.text_input("Ask a question:")

# On Submit, Process the Question
if st.button("Submit") and user_input:
    # Answer the user's question using the indexed content
    answer = answer_question(user_input)
    st.write(f"Answer:\n{answer}")
