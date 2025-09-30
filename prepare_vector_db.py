from dotenv import load_dotenv
load_dotenv()

import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
import config as cons

#init embedding model
embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={'device': cons.DEVICE},   # or "cuda" if you have a GPU
            encode_kwargs={'normalize_embeddings': True}  # E5 models work best when normalized
        )

def create_db_from_file():
    """
    init loader to load pdf folder
    """
    try:
        pdf_path = cons.PDF_DATA_PATH
        loader = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        db = Milvus.from_documents(
            chunks,
            embeddings,
            connection_args={"uri": cons.MILVUS_URI},
            collection_name="data_vectors",  #  collection name in  Milvus
        )
        return db
    except Exception as e:
        logging.error(f"Error in extract PDF {e}", exc_info=True)
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

if __name__ == "__main__":
    create_db_from_file()

