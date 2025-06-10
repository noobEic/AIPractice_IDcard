import os
import argparse
import pandas as pd
import evaluate
import numpy as np
from bert_score import score as bert_score
from sklearn.metrics import f1_score, accuracy_score
from sentence_transformers import SentenceTransformer, util
import jieba
from fuzzywuzzy import fuzz
def calculate_semantic_similarity(references, predictions):
    """计算语义相似度"""
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    
    # 编码所有句子
    ref_embeddings = model.encode(references, convert_to_tensor=True)
    pred_embeddings = model.encode(predictions, convert_to_tensor=True)
    
    # 计算余弦相似度
    cosine_scores = util.pytorch_cos_sim(pred_embeddings, ref_embeddings)
    
    # 获取每个预测与对应参考的相似度
    return np.diag(cosine_scores.cpu().numpy())

def calculate_metrics(references, predictions):
    exact_matches = [1 if ref.strip().lower() == pred.strip().lower() else 0 
                    for ref, pred in zip(references, predictions)]
    exact_match_rate = np.mean(exact_matches)
    
    semantic_sim = calculate_semantic_similarity(references, predictions)
    
    rouge = evaluate.load('rouge')
    rouge_scores = rouge.compute(
        predictions=predictions, 
        references=references,
        use_stemmer=True
    )
    
    f1_scores = []
    for ref, pred in zip(references, predictions):
        ref_tokens = set(jieba.lcut(ref))
        pred_tokens = set(jieba.lcut(pred))
        common = ref_tokens & pred_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
        f1_scores.append(f1)
    
    return {
        'semantic_similarity': float(np.mean(semantic_sim)),
        'rougeL': rouge_scores['rougeL'],
        'token_f1': float(np.mean(f1_scores))
    }

def classify_error(q, ref, pred):
    norm_pred = pred.strip().lower()
    norm_ref = ref.strip().lower()
    
    if norm_pred == norm_ref:
        return None
    
    if not norm_pred:
        return "无答案"

    if fuzz.partial_ratio(norm_ref, norm_pred) > 70:
        return "部分正确"

    keywords = set(jieba.lcut_for_search(q)[:3])
    if not any(kw in norm_pred for kw in keywords if len(kw) > 1):
        return "不相关"
    
    return "错误"
    
    return error_counts, error_types

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估问答系统性能")
    parser.add_argument("--qa_data", type=str, default="synthetic_data_generation_ragas_2.csv")
    parser.add_argument("--result", type=str, default="results/result_zero_shot.csv")
    parser.add_argument("--output_report", type=str, default="evaluation_report.html")
    
    args = parser.parse_args()
    
    qa_data = pd.read_csv(args.qa_data)
    result_data = pd.read_csv(args.result,header=None, names=["question","answer"])
    
    min_len = min(len(qa_data), len(result_data))
    qa_data = qa_data.iloc[:min_len]
    result_data = result_data.iloc[:min_len]

    global_similarity_model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    metrics = calculate_metrics(qa_data['reference'], result_data['answer'])
    
    error_details = []
    for i, row in qa_data.iterrows():
        err_type = classify_error(row['user_input'], row['reference'], result_data.iloc[i]['answer'])
        if err_type:
            error_details.append((err_type, i, ...))

    print(metrics)
