import streamlit as st
from chain import retrieval_chain
from langfuse.langchain import CallbackHandler

handler = CallbackHandler()
# -------- Streamlit UI ----------
st.set_page_config(page_title="Chat với FAISS + Azure OpenAI", page_icon="🤖")
st.title("💬 Chatbot FAISS + Azure OpenAI")

# Lưu lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input chat
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi LLM chain
    result = retrieval_chain.invoke({"input": prompt}, config={
                                                                "callbacks": [handler],
                                                                "metadata": {"user_id": "123", "session": "abc"}
                                                            })
    answer = result.get("answer") or result.get("context")

    # Hiển thị câu trả lời
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)