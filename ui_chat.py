import streamlit as st
import requests
import config as cons
import time
import re

st.set_page_config(page_title="Chat Demo", page_icon="", layout="wide")

def format_markdown(text):
    """
    Chuẩn hóa markdown để hiển thị đúng format trong Streamlit
    """
    if not text:
        return text
    
    # Loại bỏ trailing spaces trước \n
    text = re.sub(r'  +\n', '\n', text)
    
    # Chuyển đổi các dấu bullet không chuẩn
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    
    # CASE 1: "- Title • Sub-item" → Tách thành 2 dòng
    # Ví dụ: "- Chuẩn đoán• Thiết bị" → "- Chuẩn đoán\n  • Thiết bị"
    text = re.sub(r'([-]\s*[^•\n]+?)\s*•\s*', r'\1\n  • ', text)
    
    # CASE 2: Đảm bảo có line break trước mỗi bullet point chính (- hoặc •)
    # Nhưng không thêm nếu đã có line break
    text = re.sub(r'([^\n])(\n?[-•]\s+[A-ZÀ-Ỹ])', lambda m: m.group(1) + '\n\n' + m.group(2).lstrip('\n'), text)
    
    # CASE 3: Sub-items (• sau -) cần indent
    lines = text.split('\n')
    formatted_lines = []
    in_subitem = False
    
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        
        # Main bullet point với -
        if stripped.startswith('- '):
            formatted_lines.append(line)
            in_subitem = True
        # Sub bullet point với •
        elif stripped.startswith('• ') and in_subitem:
            # Đảm bảo indent 2 spaces
            formatted_lines.append('  ' + stripped)
        # Dòng bình thường hoặc continuation
        else:
            formatted_lines.append(line)
            # Reset nếu gặp dòng trống hoặc main bullet mới
            if not stripped or stripped.startswith('- '):
                in_subitem = False
    
    text = '\n'.join(formatted_lines)

    # CASE 4: Đảm bảo heading in đậm và list (- hoặc số) có line break rõ ràng
    # Ví dụ: "**Các trường hợp:**\n\n- ..." thay vì "**Các trường hợp:**- ..."
    text = re.sub(r'(\*\*.*?\*\*):\s*\n?(-|\d+\.)', r'\1:\n\n\2', text)
    
    # Đảm bảo có space sau bullet points
    text = re.sub(r'•([^\s])', r'• \1', text)
    text = re.sub(r'^-([^\s])', r'- \1', text, flags=re.MULTILINE)
    
    # Loại bỏ quá nhiều line breaks (tối đa 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Loại bỏ spaces thừa
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

def stream_markdown(response, placeholder):
    """
    Stream markdown với formatting real-time
    """
    ai_message = ""
    last_update = time.time()
    update_interval = 0.08  # Update mỗi 80ms
    chunk_count = 0
    
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data = line.replace("data: ", "")
            if data == "[DONE]":
                break
            
            ai_message += data
            chunk_count += 1
            
            # Update UI theo interval hoặc sau khi có đủ content
            current_time = time.time()
            if (current_time - last_update >= update_interval) or (chunk_count % 20 == 0):
                formatted = format_markdown(ai_message)
                placeholder.markdown(formatted)
                last_update = current_time
    
    # Render lần cuối với format hoàn chỉnh
    formatted = format_markdown(ai_message)
    placeholder.markdown(formatted)
    
    return formatted

# =====================
# Layout chia 2 cột
# =====================
col1, col2 = st.columns([1, 2])

# ---------------------
# 📂 Bên trái: Upload PDF
# ---------------------
with col1:
    st.header("Upload Data")

    uploaded_files = st.file_uploader(
        "Chọn file PDF để insert vào Milvus",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Insert to DB"):
        if uploaded_files:
            files = [("files", (f.name, f, "application/pdf")) for f in uploaded_files]
            try:
                response = requests.post(f"{cons.API_URL}/insert_pdfs/", files=files)
                if response.status_code == 200:
                    st.success(response.json())
                else:
                    st.error(f"Lỗi insert: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Vui lòng chọn ít nhất 1 file PDF")


# ---------------------
# 💬 Bên phải: Chat UI
# ---------------------
with col2:
    st.header("Chatbot With OpenAI")

    # Khởi tạo lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Container để hiển thị lịch sử chat
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Sử dụng st.chat_input() - nó sẽ tự động nằm ở dưới cùng
    user_input = st.chat_input("Nhập câu hỏi của bạn...")

    # Xử lý khi user gửi tin nhắn
    if user_input:
        # Thêm tin nhắn user vào lịch sử
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Hiển thị tin nhắn user
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        # Gọi API và hiển thị response
        with chat_container:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                ai_message = ""
                buffer = ""

                try:
                    response = requests.post(
                        f"{cons.API_URL}/chat/stream",
                        json={"question": user_input},
                        stream=True,
                    )
                    # Stream Markdown trực tiếp
                    ai_message = stream_markdown(response, placeholder)

                except Exception as e:
                    ai_message = f"Error: {str(e)}"
                    placeholder.markdown(ai_message)

                # Lưu response vào lịch sử
                st.session_state.messages.append({"role": "assistant", "content": ai_message})
        
        # Rerun để cập nhật UI
        st.rerun()