# api_service.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import argparse

app = FastAPI()

# 允许跨域请求（前端开发需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储加载的模型和资源
global_resources = {
    "generator": None,
    "vector_db": None
}

class InferenceRequest(BaseModel):
    question: str
    max_length: int = 500
    top_k: int = 3

def init_resources(args):
    """初始化模型和向量数据库"""
    print("⚙️ 正在加载资源...")

    kwargs = {"max_memory": {i: "24GiB" for i in range(4)}, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(
        args.rag_model_name,
        torch_dtype='float16',
        **kwargs,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.rag_model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    # 加载向量数据库
    embedder = HuggingFaceEmbeddings(
        model_name=args.embedding_model_name,
        model_kwargs={'device': args.device},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = FAISS.load_local(args.vector_db_dir, embedder, allow_dangerous_deserialization=True)
    
    global_resources.update({
        "generator": generator,
        "vector_db": vector_db
    })
    print("✅ 资源加载完成！")

def rag_answer(question: str, top_k: int = 3, max_length: int = 500) -> str:
    """RAG问答核心逻辑"""
    if not global_resources["generator"] or not global_resources["vector_db"]:
        raise RuntimeError("模型未初始化")
    
    # 1. 检索相关文档
    retriever = global_resources["vector_db"].as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)
    
    # 2. 构建提示词（根据您的PROMPT模板调整）
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
    <system>
    你是一个专业问答助手，请基于以下上下文回答用户问题：
    {context}
    </system>
    <user>{question}</user>
    <assistant>
    """
    
    # 3. 生成回答
    response = global_resources["generator"](
        prompt,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        pad_token_id=global_resources["generator"].tokenizer.eos_token_id
    )
    
    return response[0]['generated_text'].replace(prompt, "").strip()

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_model_name", default="/home/zxyu/private_data/pretrain/Qwen2.5-3B-Instruct")
    parser.add_argument('--embedding_model_name', default='BAAI/bge-large-zh')
    parser.add_argument('--vector_db_dir', default='./test_vector_db')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args([])  # 使用默认参数
    
    init_resources(args)

@app.post("/ask")
async def ask_question(request: InferenceRequest):
    """问答接口"""
    try:
        answer = rag_answer(
            question=request.question,
            top_k=request.top_k,
            max_length=request.max_length
        )
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """健康检查端点"""
    return {"status": "active", "models_loaded": bool(global_resources["generator"])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)