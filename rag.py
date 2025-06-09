from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import faiss
import argparse
import pickle
import os
from langchain_community.vectorstores import FAISS


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def rag_answer(generator, vector_db, query: str, top_k: int = 5, max_length: int = 2048):
    instruction = "为这个句子生成表示以用于检索相关文章:"

    results = vector_db.similarity_search(instruction + query, k=top_k)
    retrieved_docs = [doc.page_content for doc in results]


    context = "\n\n".join(retrieved_docs)
    prompt = f"""
    基于以下上下文回答问题：{context} 问题：{query} 答案："""
    
    response = generator(
        prompt,
        max_length=max_length,
        truncation=True,
        do_sample=True,
        temperature=0.7
    )
    return response[0]['generated_text'].replace(prompt, "")


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
    answer = rag_answer(generator, vector_db, query, top_k=5, max_length=2048)
    print(answer)
    