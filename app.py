from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File
import uvicorn
from llm.rag_service import RAG_Service
from untils.prepare_vector_db import insert_pdf_to_milvus
from typing import List
import os
import shutil

app = FastAPI(title="Chatbot RAG API", version="1.0")
rag_service = RAG_Service()
retrieval_chain = rag_service.init_retrieval_chain()

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    question = body.get("question", "")

    if not question:
        return JSONResponse({"error": "Missing question"}, status_code=400)

    try:
        result = retrieval_chain.invoke(question)
        return {"answer": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# API stream
@app.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    question = body.get("question", "")

    if not question:
        return JSONResponse({"error": "Missing question"}, status_code=400)

    async def event_generator():
        try:
            full_response = ""
            async for chunk in retrieval_chain.astream(question):
                full_response += chunk
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/insert_pdfs/")
async def insert_pdfs(files: List[UploadFile] = File(...)):
    saved_files = []
    try:
        os.makedirs("pdf_data", exist_ok=True)
        for file in files:
            file_path = os.path.join("pdf_data", file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
        result = insert_pdf_to_milvus(saved_files, "data_vectors", rag_service.embeddings)
        return result
    finally:
        for path in saved_files:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
