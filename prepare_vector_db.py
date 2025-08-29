from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import  config as cons

def create_db_from_file():
    #init loader to load pdf folder
    pdf_path = cons.PDF_DATA_PATH
    vector_path = cons.VECTOR_DB_PATH
    
    loader = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = text_splitter.split_documents(documents)

    #Embedding
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vector_path)
    return db

create_db_from_file()