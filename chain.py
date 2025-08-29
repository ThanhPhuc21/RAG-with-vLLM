from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain

import config as cons

if not os.environ.get("AZURE_OPENAI_API_KEY"):
    print("Please Init .evn file end import API KEY")


## Init LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)


## Load FAISS DB
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
db = FAISS.load_local(
    cons.VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

## Init retriever
retriever = db.as_retriever(search_kwargs = {"k" :3})

## Init prompt
system_prompt = (
    "Use the provided context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# -- Init combine_docs_chain and retrieval chain --
combine_chain = create_stuff_documents_chain(llm=llm, prompt=chat_prompt)
retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)

# #----Query------
# # query = "Giai đoạn phát triển  2006 - 2010 như nào?"
# # result = retrieval_chain.invoke({"input": query})
# # print("Answer:", result.get("answer") or result.get("context"))

