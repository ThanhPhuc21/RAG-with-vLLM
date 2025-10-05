from dotenv import load_dotenv
load_dotenv()
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
import config as cons
import os
import re

# init embedding model
# embeddings = HuggingFaceEmbeddings(
#             model_name="intfloat/multilingual-e5-small",
#             model_kwargs={'device': cons.DEVICE},   # or "cuda" if you have a GPU
#             encode_kwargs={'normalize_embeddings': True}  # E5 models work best when normalized
#         )
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["EMBEDDING_AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["EMBEDDING_AZURE_OPENAI_API_VERSION"],
)

def preprocess_text_for_semantic(text: str) -> str:
    """Preprocess nhẹ, giữ nguyên cấu trúc ngữ nghĩa"""
    # Xóa ký tự điều khiển
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Fix hyphenation từ PDF
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    
    return text.strip()

def create_db_from_file_semantic():
    """
    Sử dụng Semantic Chunking để tự động tìm điểm cắt tối ưu
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
        
        # Preprocess - GIỮ LẠI bước này
        for doc in documents:
            doc.page_content = preprocess_text_for_semantic(doc.page_content)
            doc.metadata['char_count'] = len(doc.page_content)
        
        # Semantic Chunking - tự động tìm điểm cắt dựa trên ngữ nghĩa
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # "percentile", "standard_deviation", "interquartile"
            breakpoint_threshold_amount=90,  # Càng cao càng ít chunks
            number_of_chunks=None,  # Để None cho tự động
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Thêm metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_length'] = len(chunk.page_content)
        
        logging.info(f"Split into {len(chunks)} semantic chunks")
        
        # Tạo Milvus collection
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

def connect_to_milvus(URI_link: str, collection_name: str) -> Milvus:
    """
    Hàm kết nối đến collection có sẵn trong Milvus
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection cần kết nối
    Returns:
        Milvus: Đối tượng Milvus đã được kết nối, sẵn sàng để truy vấn
    Chú ý:
        - Không tạo collection mới hoặc xóa dữ liệu cũ
        - Sử dụng model 'text-embedding-3-large' cho việc tạo embeddings khi truy vấn
    """
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vectorstore

# Hàm insert dữ liệu PDF vào Milvus
def insert_pdf_to_milvus(file_paths: List[str], collection_name: str = "data_vectors"):
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
                doc.metadata["trapped"] = "Fals"  # hoặc giá trị mặc định mà bạn muốn

        vectorstore = connect_to_milvus(cons.MILVUS_URI, collection_name)

        vectorstore.add_documents(chunks)
        return {"status": "success", "inserted_docs": len(chunks)}

    except Exception as e:
        logging.error(f"Error insert PDF {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    
if __name__ == "__main__":
    create_db_from_file_semantic()

