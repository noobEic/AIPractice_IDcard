import os
import argparse
from openai import OpenAI
import sys
sys.path.append('../')
from api import BASE_URL,API_KEY
num_questions = 10
prompt = "您的任务是根据给定的上下文提出{10个问题，并给出每个问题的答案。 \
在每个问题的末尾加上\"?\" \
提供的上下文写出该问题的答案。 \
每个问题/答案之间用 \"XXX \"隔开。\
每个问题必须以 \"question: \"开头。\
每个答案必须以 \"answer: \"开头。\
问题必须符合以下规则： \
1.即使在没有给定上下文的情况下，问题也应该对人类有意义。\
2.问题应能根据给定的上下文给出完整的答案。\
3.问题应从包含重要信息的上下文中提取。也可以是表格、代码等。\
4.问题答案不应包含任何链接。\
5.问题难度应适中。\
6.问题必须合理，必须为人类所理解和回答。 \
7.不要在问题中使用 \"提供上下文 \"等短语。 \
8.避免在问题中使用 \"和 \"字，因为它可以分解成多个问题。\
9.问题不应超过 10 个单词，尽可能使用缩写。\
  context: {context}   \
    "



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_dir', type=str, default='../reference')
    parser.add_argument('--output_dir',type=str,default='./data')

    args = parser.parse_args()
    reference_dir = args.reference_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for file in os.listdir(reference_dir):
        if file.endswith('.txt'):
            with open(os.path.join(reference_dir, file), 'r') as f:
                text = f.read()
                text = prompt + text
                response = client.completions.create(model="deepseek-r1", prompt=text, max_tokens=1000, stop=["\n"])
                question = response.choices[0].text.strip()
                answer = text.strip()
                with open(os.path.join(output_dir, file.replace('.txt', '.txt')), 'w') as f:
                    f.write(question + '\n' + answer)
                    print(question + '\n' + answer)


    
        

        
    