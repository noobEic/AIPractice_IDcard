import os
from openai import OpenAI
import argparse
from tqdm import tqdm
import pandas as pd
from api import API_KEY, BASE_URL
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def process_response(response):
    tag_position = response.find('</think>')
    if tag_position != -1:
        return response[tag_position + len('</think>'):].strip()
    return response

def generate_answer(model, tokenizer, question, max_length=500, temperature=0.7):
    messages = [
        {"role": "system", "content": "你是一个身份证办理的专家助手。我将严格遵守以下回答规则：1. 只提供最终答案，不添加解释或思考过程。2. 回答简洁明了，直接回应用户问题。3. 不使用列表、分段或换行。4. 不提出反问或新问题。5. 保持专业、客观。"},
        {"role": "user", "content": question}
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        return_attention_mask=True
    ).to(model.device)
    
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_id = tokenizer.eos_token_id

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=temperature,
        pad_token_id=eos_token_id,
        eos_token_id=[im_end_id, eos_token_id],
        repetition_penalty=1.1
    )

    response_ids = outputs[0, inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
    return process_response(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--qa_data", type=str,default="synthetic_data_generation_ragas_2.csv")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--setting", type=str, default="RAG")
    parser.add_argument('--device', type=str, default='cuda')
    
    ### Zero-shot setting ###
    parser.add_argument("--zero_shot_model_path", type=str, default="/home/zxyu/private_data/pretrain/Qwen2.5-7B-Instruct")
    
    ### SFT setting ###
    parser.add_argument("--sft_model_path", type=str, default="./output/Qwen2.5-7B-Instruct/checkpoint-5")
    parser.add_argument("--sft_max_length", type=int, default=500)
    parser.add_argument("--sft_temperature", type=float, default=0.7)
    ### RAG ###
    parser.add_argument("--rag_model_path", type=str, default="/home/zxyu/private_data/pretrain/Qwen2.5-7B-Instruct")
    parser.add_argument('--embedding_model_name', type=str, default='BAAI/bge-large-zh')
    parser.add_argument('--reranker_model_name', type=str, default='BAAI/bge-reranker-large')
    parser.add_argument('--vector_db_dir', type=str, default='./test_vector_db')
    
    
    

    args = parser.parse_args()
    qa_data = args.qa_data
    output_dir = args.output_dir

    qa_data = pd.read_csv(qa_data)
    questions = qa_data["user_input"].tolist()

    prompt = "以下是关于身份证办理的问题，请用一段话回答，不要分段或换行。你的答案不应包含多余的信息，包括你的思考过程。"


    results = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.setting == "zero_shot":
        print(f"加载模型: {args.zero_shot_model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            args.zero_shot_model_path,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.zero_shot_model_path,   
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        model.eval()

        for question in tqdm(questions, desc="zero-shot"):
            with torch.no_grad():
                answer = generate_answer(
                    model, 
                    tokenizer, 
                    question,
                    max_length=args.sft_max_length,
                    temperature=args.sft_temperature
                )
            results.append((question, answer))
            print(f"\n问题: {question}\n答案: {answer}\n{'-'*50}")
            
    if args.setting == "SFT":
        
        print(f"加载微调模型: {args.sft_model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            args.sft_model_path,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.sft_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        if os.path.exists(os.path.join(args.sft_model_path, "adapter_config.json")):
            from peft import PeftModel
            print("检测到QLoRA适配器，正在合并...")
            model = PeftModel.from_pretrained(model, args.sft_model_path)
            model = model.merge_and_unload()
        
        model.eval()

        for question in tqdm(questions, desc="SFT"):
            with torch.no_grad():
                answer = generate_answer(
                    model, 
                    tokenizer, 
                    question,
                    max_length=args.sft_max_length,
                    temperature=args.sft_temperature
                )
            results.append((question, answer))
            print(f"\n问题: {question}\n答案: {answer}\n{'-'*50}")
    
    
    if args.setting == "RAG":
        from rag import rag_answer
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        from langchain_community.vectorstores import FAISS
        from FlagEmbedding import FlagReranker
        
        device = args.device
        kwargs = {
            "max_memory": {i: "24GiB" for i in range(4)},
            "device_map": "auto",
        }
        model = AutoModelForCausalLM.from_pretrained(args.rag_model_path,
                                                     torch_dtype='float16',
                                                     **kwargs,
                                                     attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(args.rag_model_path)
        
        embedder = HuggingFaceBgeEmbeddings(
            model_name=args.embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        reranker = FlagReranker(args.reranker_model_name, use_fp16=True)
        
        #reranker = None
        
        vector_db = FAISS.load_local(args.vector_db_dir, embedder, allow_dangerous_deserialization=True)
         
        for question in tqdm(questions):
            answer = rag_answer(model, tokenizer, vector_db, question, top_k=10, max_length=2048, reranker=reranker, top_k_rerank=3)
            print(answer)
            results.append((question, answer))
            
    if args.setting == "RAG-SFT":
        from rag import rag_answer
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        from langchain_community.vectorstores import FAISS
        from FlagEmbedding import FlagReranker
        device = args.device
        kwargs = {
            "max_memory": {i: "24GiB" for i in range(4)},
            "device_map": "auto",
        }
        model = AutoModelForCausalLM.from_pretrained(args.sft_model_path,
                                                     torch_dtype='float16',
                                                     **kwargs,
                                                     attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
        
        embedder = HuggingFaceBgeEmbeddings(
            model_name=args.embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        reranker = FlagReranker(args.reranker_model_name, use_fp16=True)
        
        #reranker = None
        
        vector_db = FAISS.load_local(args.vector_db_dir, embedder, allow_dangerous_deserialization=True)
        
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        
        for question in tqdm(questions):
            answer = rag_answer(model, tokenizer, vector_db, question, top_k=10, max_length=2048, reranker=reranker, top_k_rerank=3)
            print(answer)
            results.append((question, answer))
    
    
    
    with open(f"{output_dir}/result_Qwen2.5-7B-Instruct_{args.setting}.csv", "w") as f:
        writer = csv.writer(f)
        for item in results:
            writer.writerow((item[0], item[1]))    
                
        