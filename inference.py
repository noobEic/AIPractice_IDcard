import os
from openai import OpenAI
import argparse
from tqdm import tqdm
import pandas as pd
from api import API_KEY, BASE_URL
import csv
def process_response(response):
    tag_position = response.find('</think>')
    if tag_position != -1:
        return response[tag_position + len('</think>'):].strip()
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--qa_data", type=str,default="id_card_qa.csv")
    parser.add_argument("--model", type=str, default="deepseek-r1")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--setting", type=str, default="zero_shot")

    args = parser.parse_args()
    qa_data = args.qa_data
    model = args.model
    output_dir = args.output_dir

    qa_data = pd.read_csv(qa_data)
    questions = qa_data["question"].tolist()

    prompt = "以下是关于身份证办理的问题，请根据以下格式进行回答：\n\n问题：\n回答：\n参考文件：。你的答案不应包含多余的信息，包括你的思考过程。"


    results = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.setting == "zero_shot":
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        for i in tqdm(range(len(questions))):
            question = questions[i]
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个身份证办理的专家助手，回答问题时要准确、简洁。"},
                    {"role": "user", "content": prompt + "\n问题：" + question}
                ],
                stream=False
            )
            print(response.choices[0].message.content)
            response = process_response(response.choices[0].message.content)
            results.append((question, response))

    with open(f"{output_dir}/result_{args.setting}.csv", "w") as f:
        writer = csv.writer(f)
        for item in results:
            writer.writerow((item[0], item[1]))    
                
        