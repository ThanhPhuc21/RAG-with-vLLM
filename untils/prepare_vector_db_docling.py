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

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["EMBEDDING_AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["EMBEDDING_AZURE_OPENAI_API_VERSION"],
)

from docling.document_converter import DocumentConverter
from langchain.schema import Document  # để tạo đối tượng Document
import os, logging, re

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

def load_documents_with_docling(pdf_path: str):
    """
    Duyệt thư mục PDF, dùng Docling để phân tích cấu trúc từng file.
    Trả về danh sách langchain.Document (mỗi section là 1 Document)
    """
    converter = DocumentConverter()
    all_docs = []

    for filename in os.listdir(pdf_path):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(pdf_path, filename)
        logging.info(f"🔍 Converting {file_path} with Docling...")
        result = converter.convert(file_path)

        docling_doc = result.document

        # Cách 1: Sử dụng export_to_markdown() để lấy nội dung có cấu trúc
        try:
            markdown_content = docling_doc.export_to_markdown()
            
            # Split theo heading markdown (##, ###, etc.)
            sections = re.split(r'\n(#{1,6})\s+(.+?)\n', markdown_content)
            
            current_section_title = None
            current_section_text = ""
            
            i = 0
            while i < len(sections):
                if i == 0 and sections[i].strip():
                    # Nội dung trước heading đầu tiên
                    current_section_text = sections[i]
                    i += 1
                elif i + 2 < len(sections) and sections[i].startswith('#'):
                    # Lưu section trước đó
                    if current_section_text.strip():
                        all_docs.append(
                            Document(
                                page_content=preprocess_text_for_semantic(current_section_text),
                                metadata={
                                    "title": current_section_title or "Introduction",
                                    "source": filename
                                }
                            )
                        )
                    
                    # Bắt đầu section mới
                    current_section_title = sections[i + 1].strip()
                    current_section_text = sections[i + 2] if i + 2 < len(sections) else ""
                    i += 3
                else:
                    i += 1
            
            # Lưu section cuối cùng
            if current_section_text.strip():
                all_docs.append(
                    Document(
                        page_content=preprocess_text_for_semantic(current_section_text),
                        metadata={
                            "title": current_section_title or "Content",
                            "source": filename
                        }
                    )
                )
                
        except Exception as e:
            logging.warning(f"Could not parse markdown from {filename}: {e}")
            
            # Cách 2: Fallback - sử dụng export_to_text() và chia theo page
            try:
                text_content = docling_doc.export_to_text()
                
                # Chia theo page nếu có page break
                pages = text_content.split('\f')  # form feed character
                
                for page_num, page_text in enumerate(pages, 1):
                    if page_text.strip():
                        all_docs.append(
                            Document(
                                page_content=preprocess_text_for_semantic(page_text),
                                metadata={
                                    "title": f"Page {page_num}",
                                    "source": filename,
                                    "page": page_num
                                }
                            )
                        )
            except Exception as e2:
                logging.error(f"Failed to process {filename}: {e2}")
                continue

    logging.info(f"✅ Extracted {len(all_docs)} structured sections from PDFs")
    return all_docs

def create_db_from_file_semantic():
    """
    Dùng Docling + Semantic Chunking để lấy ngữ cảnh tốt hơn
    """
    try:
        pdf_path = cons.PDF_DATA_PATH
        
        # 1️⃣ Load & phân tích cấu trúc PDF bằng Docling
        documents = load_documents_with_docling(pdf_path)
        
        logging.info(f"Loaded {len(documents)} structured sections from PDFs")
        
        # 2️⃣ (Optional) thêm metadata về độ dài
        for doc in documents:
            doc.metadata['char_count'] = len(doc.page_content)
        
        # 3️⃣ Semantic Chunking (vẫn giữ nguyên)
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
            number_of_chunks=None,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_length'] = len(chunk.page_content)
        
        logging.info(f"Split into {len(chunks)} semantic chunks")

        # 4️⃣ Tạo Milvus collection
        db = Milvus.from_documents(
            chunks,
            embeddings,
            connection_args={"uri": cons.MILVUS_URI},
            collection_name="data_vectors_docling",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
        )
        
        logging.info(f"✅ Created Milvus collection with {len(chunks)} semantic vectors")
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

