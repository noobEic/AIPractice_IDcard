import os
import argparse
from openai import OpenAI

API_KEY = os.environ['OPENAI_API_KEY']
BASE_URL = os.environ['OPENAI_API_URL']

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
                prompt = "Please generate a question and answer pair for the following text:\n\n" + text + "\n\n"
                response = client.Completion.create(engine="davinci-instruct-beta", prompt=prompt, max_tokens=100, stop=["\n"])
                question = response.choices[0].text.strip()
                answer = text.strip()
                with open(os.path.join(output_dir, file.replace('.txt', '.txt')), 'w') as f:
                    f.write(question + '\n' + answer)
                    print(question + '\n' + answer)


    
        

        
    