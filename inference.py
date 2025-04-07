import os
from openai import OpenAI
import argparse
from tqdm import tqdm
import pandas as pd
BASE_URL = "http://505676.proxy.nscc-gz.cn:8888/v1/" 
API_KEY = "sk-GaI4lht7mEdBN75i30FfAe11465e419795156aAc6d3eF7F0"

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

    prompt = "以下是关于身份证办理的问题，请根据以下格式进行回答：\n\n问题：\n回答：\n参考文件："



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
            with open(f"{output_dir}/{i}.txt", "w") as f:
                f.write(response.choices[0].message.content)
        