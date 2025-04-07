import os
import argparse
import pandas as pd
import bleu
import rouge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--qa_data", type=str,default="id_card_qa.csv")
    parser.add_argument("--result", type=str, default="results/result_zero_shot.csv")

    args = parser.parse_args()
    qa_data = args.qa_data
    result = args.result
    qa_data = pd.read_csv(qa_data)
    result = pd.read_csv(result)
    
    answers_gt = qa_data["answer"].tolist()
    answers_rs = result["answer"].tolist()

    bleu_score = bleu.bleu(answers_gt, answers_rs)
    rouge_score = rouge.rouge(answers_gt, answers_rs)
    print(f"BLEU score: {bleu_score}")
    print(f"ROUGE score: {rouge_score}")