from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, trim_messages

if not os.environ.get("AZURE_OPENAI_API_KEY"):
    print("Can not find API KEY")

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
# print(model.invoke([HumanMessage(content="Hi! I'm Bob")]))
promt_template = ChatPromptTemplate(
    [
        (
            "system",
            "You are an assistant. Please answer the questions briefly and clearly."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)
# simple fallback token counter (không chính xác nhưng dùng để debug)
def simple_token_counter(messages):
    return sum(len(getattr(m, "content", "").split()) for m in messages)

trimmer = trim_messages(
    max_tokens=512,
    strategy="last",
    token_counter=simple_token_counter,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

graph_build = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    print(f"Messages before trimming: {len(state)}")
    trimmed_messages = trimmer.invoke(state["messages"])
    print(f"Messages after trimming: {len(trimmed_messages)}")
    print("Remaining messages:")
    for msg in trimmed_messages:
        print(f"  {type(msg).__name__}: {msg.content}")
    promt = promt_template.invoke(trimmed_messages)
    response = model.invoke(promt)
    return {"messages": response}

graph_build.add_node("model", call_model)
graph_build.add_edge(START, "model")

memory = MemorySaver()
app = graph_build.compile(checkpointer=memory)
