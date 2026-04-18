import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store():
    # Load the document
    loader = TextLoader("infrastructure_guidelines.md")
    docs = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Free open-source embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS instance and build the vector store
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # Save it locally
    vectorstore.save_local("faiss_index")
    print("Vector database successfully built and saved to 'faiss_index' directory.")

if __name__ == "__main__":
    build_vector_store()
