import os
import argparse
import pandas as pd
import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--qa_data", type=str,default="synthetic_data_generation_ragas_2.csv")
    parser.add_argument("--result", type=str, default="results/result_zero_shot.csv")

    args = parser.parse_args()
    qa_data = args.qa_data
    result = args.result
    qa_data = pd.read_csv(qa_data)
    result = pd.read_csv(result)
    
    answers_gt = qa_data["reference"].tolist()
    answers_rs = result["answer"].tolist()

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bleu_score = bleu.compute(predictions=answers_rs, references=answers_gt)
    rouge_score = rouge.compute(predictions=answers_rs, references=answers_gt, rouge_types=["rougeL"])
    print(f"BLEU score: {bleu_score}")
    print(f"ROUGE score: {rouge_score}")