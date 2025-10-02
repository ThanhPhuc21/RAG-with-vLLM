from dotenv import load_dotenv
load_dotenv()

import os
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import EnsembleRetriever
from langfuse.langchain import CallbackHandler
import logging
from prepare_vector_db import connect_to_milvus
import config as cons

if not os.environ.get("AZURE_OPENAI_API_KEY"):
    print("Please Init .evn file end import API KEY")


## Init LLM
# def init_llm():
#     """
#     Initialize llm
#     """
#     return ChatOpenAI(
#         model="Qwen/Qwen1.5-4B-Chat",
#         openai_api_key="EMPTY",
#         openai_api_base="http://104.199.243.127:8000/v1",
#     )

def init_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        # api_key=os.environ["AZURE_OPENAI_API_KEY"]
    )


def get_qa_prompt():
    """
    Get question answering chat prompt template
    """
    return ChatPromptTemplate.from_messages([
        ("system",
        """
            You are a MEDICAL ASSISTANT specialized in answering health-related and clinical questions.

            **Core Instructions:**
            - Use ONLY the provided retrieved context to answer the user's question
            - Stay strictly faithful to the retrieved context - do not paraphrase or reinterpret the information
            - If the retrieved context is insufficient or you do not know the answer, say "I don't know" and suggest appropriate next steps (e.g., consult a specialist, check references, order tests)

            **Answer Format:**
            - Provide complete, easy-to-understand, and practical answers using short sentences
            - Use 1–3 bullet points for actionable recommendations
            - Cite the source or excerpt from the retrieved context when applicable and put it in ()
            - Format your answer in Markdown with line breaks after each bullet point
            - Answer using  Vietnamese
            """),
        ("human", "Question: {question}\nContext: {context}\nAnswer:")
    ])


def format_docs(docs: list[Document]):
    """
    Format documents for prompt
    """
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_docs(query: str):
    """Lấy docs từ hybrid retriever"""
    docs = hybrid_retriever.invoke(query)
    return format_docs(docs)


def conditional_retriever(question: str):
    question_text = question.strip().lower()
    trivial_questions = ["hi", "hello", "what's your name?", "how are you?"]
    if question_text in trivial_questions:
        return "" 
    return retrieve_docs(question)

conditional_retriever_runnable = RunnableLambda(conditional_retriever)

def create_qa_chain(retriever: Any, llm: ChatOpenAI, prompt: ChatPromptTemplate):
    """
    Set up question answering chain
    """
    print("LANGFUSE_SECRET_KEY", os.environ["LANGFUSE_SECRET_KEY"])
    langfuse_handler = CallbackHandler()

    return (
        {
        "question": RunnablePassthrough(),   # giữ nguyên input question
        "context": conditional_retriever_runnable,  # lấy context tùy điều kiện
    }
        | prompt
        | llm
        | StrOutputParser()
    ).with_config(callbacks=[langfuse_handler])


## Init retriever
def init_hybrid_retriever(
    top_k: int = 3,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4
):
    """
    Khởi tạo Hybrid Retriever với Vector Search + BM25 + Reranking
    """
    try:
        vectorstore = connect_to_milvus(cons.MILVUS_URI, "data_vectors")
        
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k,
            }
        )
        
        # 2. BM25 Retriever
        logging.info("Loading all documents for BM25...")
        
        # Cách 1: Nếu collection nhỏ (< 10k docs)
        all_docs = vectorstore.similarity_search(
            "", 
            k=2000,  # Lấy max có thể
            expr=""   # Không filter
        )
        
        # Cách 2: Nếu collection lớn - dùng scroll (tốt hơn)
        # all_docs = []
        # for doc_batch in vectorstore.scroll():  # Nếu Milvus support scroll
        #     all_docs.extend(doc_batch)
        
        logging.info(f"Loaded {len(all_docs)} documents for BM25")
        
        if len(all_docs) == 0:
            raise ValueError("No documents found in Milvus collection")
        
        bm25_retriever = BM25Retriever.from_documents(
            all_docs,
            k=top_k  # Cũng lấy nhiều để rerank
        )
        
        # 3. Ensemble Retriever (kết hợp)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[vector_weight, bm25_weight]
        )
        
        # 4. CrossEncoder Reranker (LOCAL, FREE)
        model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
            # Hoặc dùng model tiếng Việt: "keepitreal/vietnamese-sbert"
        )
        
        compressor = CrossEncoderReranker(
            model=model,
            top_n=top_k
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        # # 3. Kết hợp 2 retrievers (Vector + BM25)
        # compression_retriever = EnsembleRetriever(
        #     retrievers=[vector_retriever, bm25_retriever],
        #     weights=[0.7, 0.3]  # bạn có thể tune tỷ lệ
        # )
        
        logging.info("Hybrid retriever initialized successfully")
        return compression_retriever
        
    except Exception as e:
        logging.error(f"Error initializing hybrid retriever: {e}", exc_info=True)
        raise


# Sử dụng
hybrid_retriever = init_hybrid_retriever(
    top_k=4,
    vector_weight=0.6,
    bm25_weight=0.4
)

   # Initialize llm and prompt
llm = init_llm()
prompt = get_qa_prompt()

# Set up QA chain
retrieval_chain = create_qa_chain(hybrid_retriever, llm, prompt)