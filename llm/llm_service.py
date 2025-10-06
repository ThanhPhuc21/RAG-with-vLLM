from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate
import config as cons

if not os.environ.get("AZURE_OPENAI_API_KEY"):
    print("Please Init .evn file end import API KEY")

class LLM_Service():
    
    def __init__(self):
        self.llm = self.init_azure_llm()
        self.embeddings = self.init_azure_embedding()

    ## Init LLM
    def init_model_vllm(self):
        """
        Initialize llm with vLLM
        """
        return ChatOpenAI(
            model="Qwen/Qwen1.5-4B-Chat",
            openai_api_key="EMPTY",
            openai_api_base="http://104.199.243.127:8000/v1",
        )

    def init_azure_llm(self):
        """
        Initialize llm with vLLM
        """
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            # api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )

    def init_huggingface_embedding(self):
        return  HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-small",
                    model_kwargs={'device': cons.DEVICE},   # or "cuda" if you have a GPU
                    encode_kwargs={'normalize_embeddings': True}  # E5 models work best when normalized
                )
    
    def init_azure_embedding(self):
        return AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["EMBEDDING_AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["EMBEDDING_AZURE_OPENAI_API_VERSION"],
        )

    def get_qa_prompt(self):
        """
        Get question answering chat prompt template
        """
        return ChatPromptTemplate.from_messages([
            ("system",
            """
                You are a ASSISTANT specialized in answering health-related and clinical questions.

                **Core Instructions:**
                - Use ONLY the provided retrieved context to answer the user's question
                - Stay strictly faithful to the retrieved context - do not paraphrase or reinterpret the information
                - If the retrieved context is insufficient or you do not know the answer, say "I don't know" and suggest appropriate next steps (e.g., consult a specialist, check references, order tests)

                **Answer Format:**
                - Provide complete, easy-to-understand, and practical answers using short sentences
                - Use 1–3 bullet points for actionable recommendations
                - Cite the source or excerpt from the retrieved context when applicable and put it in ()
                - Format your answer in Markdown with line breaks after each bullet point
                - If human chat "hi, hello, ... then say hi and introduce yourself as assistant 
                """),
            ("human", "Question: {question}\nContext: {context}\nAnswer:")
        ])
