from dotenv import load_dotenv
load_dotenv()
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_milvus import Milvus
import config as cons
import re


def preprocess_text_for_semantic(text: str) -> str:
    """Preprocess simple, can't change format data"""
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    
    return text.strip()

def create_db_from_file_semantic(embeddings):
    """
    Using  Semantic Chunking 
    """
    try:
        pdf_path = cons.PDF_DATA_PATH
        # Load PDFs
        loader = DirectoryLoader(
            pdf_path, 
            glob="*.pdf", 
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        
        logging.info(f"Loaded {len(documents)} pages from PDFs")
        
        # Preprocess
        for doc in documents:
            doc.page_content = preprocess_text_for_semantic(doc.page_content)
            doc.metadata['char_count'] = len(doc.page_content)
        
        # Semantic Chunking
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # "percentile", "standard_deviation", "interquartile"
            breakpoint_threshold_amount=90,  
            number_of_chunks=None, 
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_length'] = len(chunk.page_content)
        
        logging.info(f"Split into {len(chunks)} semantic chunks")
        
        # create Milvus collection
        db = Milvus.from_documents(
            chunks,
            embeddings,
            connection_args={"uri": cons.MILVUS_URI},
            collection_name="data_vectors_semantic",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
        )
        
        logging.info(f"Created Milvus collection with {len(chunks)} semantic vectors")
        return db
        
    except Exception as e:
        logging.error(f"Error in create_db_from_file_semantic: {e}", exc_info=True)
        raise

def connect_to_milvus(URI_link: str, collection_name: str, embeddings) -> Milvus:
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vectorstore


def insert_pdf_to_milvus(file_paths: List[str], collection_name: str, embeddings):
    """
    Insert data PDF into Milvus
    """
    try:
        docs = []
        for path in file_paths:
            print("path", path)
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        # Split document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        for doc in chunks:
            if "trapped" not in doc.metadata:
                doc.metadata["trapped"] = "Fals" 

        vectorstore = connect_to_milvus(cons.MILVUS_URI, collection_name, embeddings)

        vectorstore.add_documents(chunks)
        return {"status": "success", "inserted_docs": len(chunks)}

    except Exception as e:
        logging.error(f"Error insert PDF {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    
if __name__ == "__main__":
    create_db_from_file_semantic()

