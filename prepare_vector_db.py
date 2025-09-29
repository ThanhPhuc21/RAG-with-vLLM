from dotenv import load_dotenv
load_dotenv()

import sys
import traceback
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
import config as cons

# Thiết lập logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# )
# logging.getLogger("pdfminer").setLevel(logging.DEBUG)
# logging.getLogger("langchain").setLevel(logging.DEBUG)

# # Ghi lại sys.excepthook để in traceback đầy đủ
# def custom_excepthook(exc_type, exc_value, exc_traceback):
#     if issubclass(exc_type, KeyboardInterrupt):
#         sys.__excepthook__(exc_type, exc_value, exc_traceback)
#         return
#     print("=== Uncaught Exception ===")
#     traceback.print_exception(exc_type, exc_value, exc_traceback)

# sys.excepthook = custom_excepthook
# Embedding
embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

def create_db_from_file():
    try:
        # init loader để load pdf folder
        pdf_path = cons.PDF_DATA_PATH
        loader = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        db = Milvus.from_documents(
            chunks,
            embeddings,
            connection_args={"uri": cons.MILVUS_URI},
            collection_name="data_vectors",  # tên collection trong Milvus
        )
        return db

    except Exception as e:
        logging.error("Lỗi khi tạo DB từ file PDF", exc_info=True)
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

