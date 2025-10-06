from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
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
from untils.prepare_vector_db import connect_to_milvus
from llm.llm_service import LLM_Service
import config as cons

class RAG_Service():

    def __init__(self):
    
        self.llm_service = LLM_Service()
        self.embeddings = self.llm_service.init_azure_embedding()

        # Initialize llm and prompt
        self.llm = self.llm_service.llm
        self.prompt = self.llm_service.get_qa_prompt()

        self.hybrid_retriever = self.init_hybrid_retriever(
            top_k=4,
            vector_weight=0.6,
            bm25_weight=0.4
        )
        self.conditional_retriever_runnable = RunnableLambda(self.conditional_retriever)
    
    def init_retrieval_chain(self):
        # Set up QA chain
        return self.create_qa_chain(self.llm, self.prompt)
    
    
    def init_hybrid_retriever(
        self,
        top_k: int = 3,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4):
        """
        Init  Hybrid Retriever with  Vector Search + BM25 + Reranking
        """
        try:
            vectorstore = connect_to_milvus(cons.MILVUS_URI, cons.COLECTION, self.embeddings)
            
            vector_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": top_k * 3,  
                    "score_threshold": 0.3
                }
            )
            
            # 2. BM25 Retriever
            logging.info("Loading all documents for BM25...")
            
            all_docs = vectorstore.similarity_search(
                "", 
                k=2000,
                expr="" 
            )
            
            logging.info(f"Loaded {len(all_docs)} documents for BM25")
            
            if len(all_docs) == 0:
                raise ValueError("No documents found in Milvus collection")
            
            bm25_retriever = BM25Retriever.from_documents(
                all_docs,
                k=top_k *3
            )
            
            # 3. Ensemble Retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[vector_weight, bm25_weight]
            )
            
            # 4. CrossEncoder Reranker (LOCAL, FREE)
            model = HuggingFaceCrossEncoder(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
                # Or using model  VN: "keepitreal/vietnamese-sbert"
            )
            
            compressor = CrossEncoderReranker(
                model=model,
                top_n=top_k
            )
            
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever
            )
            logging.info("Hybrid retriever initialized successfully")
            return compression_retriever
            
        except Exception as e:
            logging.error(f"Error initializing hybrid retriever: {e}", exc_info=True)
            raise


    def format_docs(self, docs: list[Document]):
        """
        Format documents for prompt
        """
        return "\n\n".join(doc.page_content for doc in docs)


    def retrieve_docs(self, query: str):
        """Get docs from  hybrid retriever"""
        docs = self.hybrid_retriever.invoke(query)
        return self.format_docs(docs)


    def conditional_retriever(self, question: str):
        question_text = question.strip().lower()
        trivial_questions = ["hi", "hello", "what's your name?", "how are you?"]
        if question_text in trivial_questions:
            return "" 
        return self.retrieve_docs(question)

    
    def create_qa_chain(self, llm: ChatOpenAI, prompt: ChatPromptTemplate):
        """
        Set up question answering chain
        """
        langfuse_handler = CallbackHandler()

        return (
            {
            "question": RunnablePassthrough(),  
            "context": self.conditional_retriever_runnable,
        }
            | prompt
            | llm
            | StrOutputParser()
        ).with_config(callbacks=[langfuse_handler])


    