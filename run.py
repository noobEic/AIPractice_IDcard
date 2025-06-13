# api_service.py (流式版本 - 支持RAG-SFT)
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
from FlagEmbedding import FlagReranker  # 新增重排序模型

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
    "vector_db": None,
    "reranker": None  # 新增重排序模型
}

class InferenceRequest(BaseModel):
    question: str
    max_length: int = 500
    top_k: int = 3

def init_resources(args):
    """初始化模型和向量数据库"""
    print("⚙️ 正在加载资源...")
    
    # 加载模型和tokenizer - 使用SFT微调模型
    print(f"加载SFT微调模型: {args.sft_model_path}")
    kwargs = {
        "max_memory": {i: "24GiB" for i in range(4)},
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "attn_implementation": "flash_attention_2"
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        **kwargs
    )
    
    # 检查是否为QLoRA适配器模型
    if os.path.exists(os.path.join(args.sft_model_path, "adapter_config.json")):
        from peft import PeftModel
        print("检测到QLoRA适配器，正在合并...")
        model = PeftModel.from_pretrained(model, args.sft_model_path)
        model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    
    # 加载向量数据库
    embedder = HuggingFaceEmbeddings(
        model_name=args.embedding_model_name,
        model_kwargs={'device': args.device},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = FAISS.load_local(args.vector_db_dir, embedder, allow_dangerous_deserialization=True)
    
    # 加载重排序模型（如果启用）
    reranker = None
    if args.use_reranker:
        print(f"加载重排序模型: {args.reranker_model_name}")
        reranker = FlagReranker(args.reranker_model_name, use_fp16=True)
    
    global_resources.update({
        "model": model,
        "tokenizer": tokenizer,
        "vector_db": vector_db,
        "reranker": reranker
    })
    print("✅ 资源加载完成！")

def retrieve_documents(question, top_k=10, top_k_rerank=3):
    """检索并重排序相关文档"""
    if not global_resources["vector_db"]:
        return []
    
    retriever = global_resources["vector_db"].as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)
    
    # 如果启用了重排序模型
    if global_resources["reranker"]:
        # 准备重排序数据
        pairs = [(question, doc.page_content) for doc in docs]
        # 计算相关性分数
        scores = global_resources["reranker"].compute_score(pairs)
        
        # 组合分数和文档
        scored_docs = list(zip(scores, docs))
        # 按分数降序排序
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        # 取top_k_rerank个文档
        docs = [doc for _, doc in scored_docs[:top_k_rerank]]
    else:
        # 没有重排序时直接取前top_k_rerank个
        docs = docs[:top_k_rerank]
    
    return docs

def generate_stream(question: str, max_length: int = 500, top_k: int = 3, top_k_rerank: int = 3):
    """流式生成回答（RAG-SFT版本）"""
    if not all([global_resources["model"], global_resources["tokenizer"], global_resources["vector_db"]]):
        yield {"error": "模型未初始化"}
        return

    # 1. 检索相关文档（带重排序）
    docs = retrieve_documents(question, top_k=10, top_k_rerank=top_k_rerank)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 2. 构建符合SFT微调格式的prompt
    messages = [
        {"role": "system", "content": "你是一个身份证办理的专家助手。请基于以下上下文回答问题：\n" + context},
        {"role": "user", "content": question}
    ]
    
    # 使用tokenizer的apply_chat_template方法构建prompt
    prompt = global_resources["tokenizer"].apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 3. 编码输入
    inputs = global_resources["tokenizer"](
        prompt, 
        return_tensors="pt"
    ).to(global_resources["model"].device)
    
    # 4. 设置流式生成器
    streamer = TextIteratorStreamer(
        global_resources["tokenizer"],
        skip_prompt=True,
        timeout=60.0,
        skip_special_tokens=True
    )
    
    # 5. 设置生成参数
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        pad_token_id=global_resources["tokenizer"].eos_token_id,
        eos_token_id=global_resources["tokenizer"].eos_token_id,
        repetition_penalty=1.1
    )
    
    # 6. 在单独线程中启动生成
    thread = threading.Thread(target=global_resources["model"].generate, kwargs=generation_kwargs)
    thread.start()
    
    # 7. 流式返回结果
    for new_text in streamer:
        # 返回JSON格式的token
        yield f'data: {{"token": "{new_text}", "finished": false}}\n\n'
    
    # 8. 生成结束标志
    yield 'data: {"token": "", "finished": true}\n\n'
    yield 'data: [DONE]\n\n'

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    parser = argparse.ArgumentParser()
    
    # RAG-SFT相关参数
    parser.add_argument("--sft_model_path", default="./output/Qwen2.5-3B-Instruct/checkpoint-5",
                       help="SFT微调模型的路径")
    parser.add_argument('--embedding_model_name', default='BAAI/bge-large-zh',
                       help="嵌入模型名称")
    parser.add_argument('--vector_db_dir', default='./test_vector_db',
                       help="向量数据库目录")
    parser.add_argument('--use_reranker', action='store_true',
                       help="是否启用重排序模型")
    parser.add_argument('--reranker_model_name', default='BAAI/bge-reranker-large',
                       help="重排序模型名称")
    parser.add_argument('--device', default='cuda',
                       help="运行设备")
    
    # 使用空列表解析参数（实际部署时可通过环境变量设置）
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
                top_k=request.top_k,
                top_k_rerank=3  # 默认重排序后取前3个文档
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """健康检查端点"""
    return {
        "status": "active", 
        "models_loaded": bool(global_resources["model"]),
        "vector_db_loaded": bool(global_resources["vector_db"]),
        "reranker_loaded": bool(global_resources["reranker"])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)