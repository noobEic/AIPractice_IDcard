from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
import argparse
import os
from langchain_community.vectorstores import FAISS


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def rerank_docs(reranker, query: str, retrieved_docs: list, top_k_rerank: int = 3):
    pairs = [[query, doc.page_content] for doc in retrieved_docs]
    scores = reranker.compute_scores(pairs)
    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]
    else:
        rerank_scores = list(rerank_scores)
    
    scored_docs = list(zip(retrieved_docs, rerank_scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = scored_docs[:top_k_rerank]

    return [doc for doc, _ in top_docs]
    
    
    
    
def rag_answer(model, tokenizer, vector_db, query: str, top_k: int = 10, max_length: int = 2048, temperature=0.7 , reranker=None, top_k_rerank=3):
    instruction = "为这个句子生成表示以用于检索相关文章:"

    results = vector_db.similarity_search(instruction + query, k=top_k)
    
    if reranker:
        reranked_results = rerank_docs(reranker, query, results, top_k=top_k_rerank)
        retrieved_docs = [doc.page_content for doc in reranked_results]
    else:
        retrieved_docs = [doc.page_content for doc in results]
        
    context = "\n\n".join(retrieved_docs)
    prompt = f"""
    基于以下上下文回答问题：{context} 问题：{query}，请用简洁的话语回答，不要分段。"""
    messages = [
        {"role": "system", "content": "你是一个身份证办理的专家助手。我将严格遵守以下回答规则：1. 只提供最终答案，不添加解释或思考过程。2. 回答简洁明了，直接回应用户问题。3. 不使用列表、分段或换行。4. 不提出反问或新问题。5. 保持专业、客观。"},
        {"role": "user", "content": prompt} 
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_id = tokenizer.eos_token_id

    outputs = model.generate(
        inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=temperature,
        pad_token_id=eos_token_id,
        eos_token_id=[im_end_id, eos_token_id],
        repetition_penalty=1.1
    )

    response_ids = outputs[0, inputs.shape[1]:]
    generated_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
    return generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model_name', type=str, default='/home/zxyu/private_data/pretrain/Qwen2.5-3B-Instruct')

    parser.add_argument('--embedding_model_name', type=str, default='BAAI/bge-large-zh')
    parser.add_argument('--vector_db_dir', type=str, default='./test_vector_db')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = args.device
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 torch_dtype='float16',
                                                 device_map='auto',
                                                 attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    embedder = HuggingFaceEmbeddings(
        model_name=args.embedding_model_name,
        model_kwargs={'device': args.device},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = FAISS.load_local(args.vector_db_dir, embedder, allow_dangerous_deserialization=True)
    
        
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    query = "如何办理身份证？"
    answer = rag_answer(generator, vector_db, query, top_k=5, max_new_tokens=2048)
    print(answer)
    