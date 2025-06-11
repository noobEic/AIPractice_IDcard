# api_service.py (流式版本)
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import argparse
import threading
from fastapi.responses import StreamingResponse
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

global_resources = {
    "model": None,
    "tokenizer": None,
    "vector_db": None
}

class InferenceRequest(BaseModel):
    question: str
    max_length: int = 500
    top_k: int = 3

def init_resources(args):
    """初始化模型和向量数据库"""
    print("⚙️ 正在加载资源...")
    
    # 加载模型和tokenizer
    kwargs = {
        "max_memory": {i: "24GiB" for i in range(4)},
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "attn_implementation": "flash_attention_2"
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        args.rag_model_name,
        **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(args.rag_model_name)
    
    # 加载向量数据库
    embedder = HuggingFaceEmbeddings(
        model_name=args.embedding_model_name,
        model_kwargs={'device': args.device},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = FAISS.load_local(args.vector_db_dir, embedder, allow_dangerous_deserialization=True)
    
    global_resources.update({
        "model": model,
        "tokenizer": tokenizer,
        "vector_db": vector_db
    })
    print("✅ 资源加载完成！")

def generate_stream(question: str, max_length: int = 500, top_k: int = 3):
    """流式生成回答"""
    if not all(global_resources.values()):
        yield {"error": "模型未初始化"}
        return

    retriever = global_resources["vector_db"].as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)

    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
    <system>
    你是一个专业问答助手，请基于以下上下文回答用户问题：
    {context}
    </system>
    <user>{question}</user>
    <assistant>
    """

    inputs = global_resources["tokenizer"](
        prompt, 
        return_tensors="pt"
    ).to(global_resources["model"].device)
    
    streamer = TextIteratorStreamer(
        global_resources["tokenizer"],
        skip_prompt=True,
        timeout=60.0,
        skip_special_tokens=True
    )
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        pad_token_id=global_resources["tokenizer"].eos_token_id
    )
    
    # 在单独线程中启动生成
    thread = threading.Thread(target=global_resources["model"].generate, kwargs=generation_kwargs)
    thread.start()
    
    # 4. 流式返回结果
    for new_text in streamer:
        # 返回JSON格式的token
        yield f'data: {{"token": "{new_text}", "finished": false}}\n\n'
    
    # 生成结束标志
    yield 'data: {"token": "", "finished": true}\n\n'
    yield 'data: [DONE]\n\n'

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_model_name", default="/home/zxyu/private_data/pretrain/Qwen2.5-3B-Instruct")
    parser.add_argument('--embedding_model_name', default='BAAI/bge-large-zh')
    parser.add_argument('--vector_db_dir', default='./test_vector_db')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args([])
    
    init_resources(args)

@app.post("/ask")
async def ask_question(request: InferenceRequest):
    """问答接口（流式响应）"""
    try:
        # 创建流式响应
        return StreamingResponse(
            generate_stream(
                question=request.question,
                max_length=request.max_length,
                top_k=request.top_k
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """健康检查端点"""
    return {"status": "active", "models_loaded": bool(global_resources["model"])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)