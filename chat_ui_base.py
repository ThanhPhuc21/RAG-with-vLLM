import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from chain_short_term.chain_simple import app


# ===== Streamlit UI =====
st.set_page_config(page_title="Chat with LangGraph", page_icon="💬")

st.title("💬 Chat UI with LangGraph + Streamlit")

# Manage threads
if "threads" not in st.session_state:
    st.session_state.threads = {"default": []}  # thread_id: messages
if "current_thread" not in st.session_state:
    st.session_state.current_thread = "default"

# Sidebar for thread management
st.sidebar.header("🧵 Threads")
thread_id = st.sidebar.selectbox(
    "Select thread", 
    options=list(st.session_state.threads.keys()), 
    index=list(st.session_state.threads.keys()).index(st.session_state.current_thread)
)

# Switch thread
if thread_id != st.session_state.current_thread:
    st.session_state.current_thread = thread_id

# Create new thread
new_thread = st.sidebar.text_input("New thread name")
if st.sidebar.button("➕ Create Thread") and new_thread.strip():
    if new_thread not in st.session_state.threads:
        st.session_state.threads[new_thread] = []
        st.session_state.current_thread = new_thread

st.sidebar.write(f"Current thread: **{st.session_state.current_thread}**")

# Display chat history
messages = st.session_state.threads[st.session_state.current_thread]
for msg in messages:
    role = "👤 You" if msg["role"] == "human" else "🤖 Assistant"
    st.markdown(f"**{role}:** {msg['content']}")

# Input box
user_input = st.chat_input("Type your message...")
if user_input:
    # Save user message
    messages.append({"role": "human", "content": user_input})
     # Hiển thị user message ngay lập tức
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        # Send to LangGraph
        config = {"configurable": {"thread_id": st.session_state.current_thread}}
        input_messages = [HumanMessage(user_input)]
        # output = app.invoke({"messages": input_messages}, config)

        # ai_message = output["messages"][-1]
        ai_message = ""
        for chunk, metadata in app.stream({"messages": input_messages}, config, stream_mode="messages"):
            if isinstance(chunk, AIMessage):
                    placeholder.markdown(ai_message)
                    ai_message += chunk.content
        messages.append({"role": "ai", "content": ai_message})
    