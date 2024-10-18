# LangChain Chatbot for Document Retrieval

This project demonstrates a chatbot built using LangChain that can retrieve information from a set of documents. The chatbot uses SentenceTransformer embeddings for semantic similarity search ,RAG based and Pinecone for vector storage.

## Features of part 1 assignment

* Loads PDF and TXT files from a specified directory.
* Splits documents into smaller chunks for efficient processing.
* Embeds documents and queries using SentenceTransformer embeddings and is rag based where datasets goes divided into chunks.
* Stores document embeddings in a Pinecone index.
  
* Performs similarity search to retrieve relevant documents for a given query.

## Requirements

* Python 3.7 or higher
* LangChain
* OpenAI
* Sentence Transformers
* Unstructured
* Detectron2
* Poppler Utils
* LangChain Community
* Pinecone

## features of part2 
* streamlit is used for frontend part to run it type streamlit run [filename.py]
* it has uploading feature that connects to pinecone databse with api key
* you can interact one file is connected
This project implements a Question-Answering (QA) Chatbot using Pinecone for vector-based search, Sentence Transformers for encoding text, and Streamlit for building an interactive web app. The bot allows users to upload PDFs, indexes the content, and answers questions based on the uploaded documents.
Example Usage
*Upload a PDF on machine learning.
*Ask: "What is supervised learning?"
The chatbot will return relevant content from the uploaded PDF.
## Dependencies
* Python 3.7 or higher
* sentence-transformers
* pinecone-client
* streamlit
* PyPDF2
## Future Improvements
* Support multiple PDFs for indexing.
* Caching results to improve response time.
* Explore other transformer models for better performance.
