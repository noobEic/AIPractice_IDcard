import os
from openai import OpenAI
import argparse
from tqdm import tqdm
import pandas as pd
from api import API_KEY, BASE_URL
import csv
from transformers import from_pretrained
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def process_response(response):
    tag_position = response.find('</think>')
    if tag_position != -1:
        return response[tag_position + len('</think>'):].strip()
    return response

def sft_generate_answer(model, tokenizer, question, max_length=500, temperature=0.7):
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = full_text[len(prompt):].strip()
    
    return process_response(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--qa_data", type=str,default="synthetic_data_generation_ragas_2.csv")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--setting", type=str, default="RAG")
    parser.add_argument('--device', type=str, default='cuda')
    
    ### Zero-shot setting ###
    parser.add_argument("--zero_shot_model", type=str, default="deepseek-r1")
    
    ### SFT setting ###
    parser.add_argument("--sft_model_path", type=str, default="./output/checkpoint-201")
    
    ### RAG ###
    parser.add_argument("--rag_model_name", type=str, default="/home/zxyu/private_data/pretrain/Qwen2.5-3B-Instruct")
    parser.add_argument('--embedding_model_name', type=str, default='BAAI/bge-large-zh')
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
        zero_shot_model = args.zero_shot_model
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        for i in tqdm(range(len(questions))):
            question = questions[i]
            response = client.chat.completions.create(
                model=zero_shot_model,
                messages=[
                    {"role": "system", "content": "你是一个身份证办理的专家助手，回答问题时要准确、简洁。"},
                    {"role": "user", "content": prompt + "问题：" + question}
                ],
                stream=False
            )
            print(response.choices[0].message.content)
            response = process_response(response.choices[0].message.content)
            results.append((question, response))

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
        print(f"模型加载完成，设备: {model.device}")

        for question in tqdm(questions, desc="SFT"):
            with torch.no_grad():
                answer = sft_generate_answer(
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
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        device = args.device
        kwargs = {
            "max_memory": {i: "24GiB" for i in range(4)},
            "device_map": "auto",
        }
        model = AutoModelForCausalLM.from_pretrained(args.rag_model_name,
                                                     torch_dtype='float16',
                                                     **kwargs,
                                                     attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(args.rag_model_name)
        
        embedder = HuggingFaceEmbeddings(
            model_name=args.embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_db = FAISS.load_local(args.vector_db_dir, embedder, allow_dangerous_deserialization=True)
        
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        
        for question in tqdm(questions):
            answer = rag_answer(generator, vector_db, question, top_k=3, max_new_tokens=500)
            print(answer)
            results.append((question, answer))
    
    
    
    with open(f"{output_dir}/result_{args.setting}.csv", "w") as f:
        writer = csv.writer(f)
        for item in results:
            writer.writerow((item[0], item[1]))    
                
        