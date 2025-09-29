from dotenv import load_dotenv
load_dotenv()

import os
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from prepare_vector_db import connect_to_milvus
import config as cons

if not os.environ.get("AZURE_OPENAI_API_KEY"):
    print("Please Init .evn file end import API KEY")


## Init LLM
def init_llm():
    """
    Initialize llm
    """
    return ChatOpenAI(
        model="Qwen/Qwen1.5-4B-Chat",
        openai_api_key="EMPTY",
        openai_api_base="http://104.199.243.127:8000/v1",
    )
def get_qa_prompt():
    """
    Get question answering chat prompt template
    """
    return ChatPromptTemplate.from_messages([
        ("system", 
         "You are an assistant for question-answering tasks. "
         "Use the following pieces of retrieved context to answer the question. "
         "If you don't know the answer, just say that you don't know. "
         "Use three sentences maximum and keep the answer concise."),
        ("human", "Question: {question}\nContext: {context}\nAnswer:")
    ])

def format_docs(docs: list[Document]):
    """
    Format documents for prompt
    """
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_docs(query: str):
    """Lấy docs từ hybrid retriever"""
    docs = hybrid_retriever.get_relevant_documents(query)
    return format_docs(docs)

# Định nghĩa hàm xử lý
def conditional_retriever(question: str):
    question_text = question.strip().lower()
    trivial_questions = ["hi", "hello", "what's your name?", "how are you?"]
    if question_text in trivial_questions:
        return ""  # trả context rỗng nếu câu trivial
    # ngược lại, gọi hybrid retriever
    return retrieve_docs(question)

# Tạo RunnableLambda từ hàm
conditional_retriever_runnable = RunnableLambda(conditional_retriever)

def create_qa_chain(retriever: Any, llm: ChatOpenAI, prompt: ChatPromptTemplate):
    """
    Set up question answering chain
    """
    return (
        {
        "question": RunnablePassthrough(),   # giữ nguyên input question
        "context": conditional_retriever_runnable,  # lấy context tùy điều kiện
    }
        | prompt
        | llm
        | StrOutputParser()
    )


## Init retriever
vectorstore = connect_to_milvus(cons.MILVUS_URI, "data_vectors")
milvus_retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

#Lấy toàn bộ docs (hoặc ít nhất một batch đủ lớn) để BM25 hoạt động
all_docs = vectorstore.similarity_search("", k=200)  # lấy 200 docs gần như "toàn bộ"

bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 4  # số doc BM25 sẽ trả về

# 3. Kết hợp 2 retrievers (Vector + BM25)
hybrid_retriever = EnsembleRetriever(
    retrievers=[milvus_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # bạn có thể tune tỷ lệ
)

   # Initialize llm and prompt
llm = init_llm()
prompt = get_qa_prompt()

# Set up QA chain
retrieval_chain = create_qa_chain(hybrid_retriever, llm, prompt)


# # #----Query------
# # # query = "Giai đoạn phát triển  2006 - 2010 như nào?"
# # # result = retrieval_chain.invoke({"input": query})
# # # print("Answer:", result.get("answer") or result.get("context"))

# question = "Ngày thành lập ACB "
# output = retrieval_chain.invoke(question)
# print("-" * 50)
# print(output)
# print("-" * 50)