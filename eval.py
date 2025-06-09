import os
import argparse
import pandas as pd
import evaluate
import numpy as np
from bert_score import score as bert_score
from sklearn.metrics import f1_score, accuracy_score
from sentence_transformers import SentenceTransformer, util

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

def evaluate_qa(qa_data_path, result_path):
    """评估问答结果"""
    # 1. 加载数据
    try:
        qa_data = pd.read_csv(qa_data_path)
        result_data = pd.read_csv(result_path,header=None, names=["question","answer"])
        
        # 确保数据对齐
        if len(qa_data) != len(result_data):
            print(f"警告：数据集长度不一致（参考集: {len(qa_data)}，结果集: {len(result_data)}）")
            min_len = min(len(qa_data), len(result_data))
            qa_data = qa_data.iloc[:min_len]
            result_data = result_data.iloc[:min_len]
        
        # 2. 准备数据
        references = qa_data["reference"].tolist()
        predictions = result_data["answer"].tolist()
        questions = qa_data["user_input"].tolist()
        
        # 3. 计算传统指标
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        
        bleu_scores = bleu.compute(predictions=predictions, references=references)
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        
        # 4. 计算问答专用指标
        
        # 4.2 BERTScore - 语义相似度
        P, R, F1 = bert_score(predictions, references, lang="zh")
        bert_score_f1 = np.mean(F1.numpy())
        
        # 4.3 语义相似度
        semantic_similarities = calculate_semantic_similarity(references, predictions)
        avg_semantic_sim = np.mean(semantic_similarities)
        
        # 5. 返回综合结果
        return {
            "bert_score_f1": float(bert_score_f1),
            "semantic_similarity": float(avg_semantic_sim),
            "bleu": bleu_scores["bleu"],
            "rougeL": rouge_scores["rougeL"],
            "details": {
                "semantic_similarities": semantic_similarities.tolist(),
            }
        }
    
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        return None

def analyze_common_errors(qa_data, result_data):
    """分析常见错误模式"""
    # 1. 准备数据
    references = qa_data["reference"].tolist()
    predictions = result_data["answer"].tolist()
    questions = qa_data["user_input"].tolist()
    
    # 2. 错误分析
    error_types = []
    for i, (q, ref, pred) in enumerate(zip(questions, references, predictions)):
        if pred.strip() == ref.strip():
            continue
            
        # 分类错误类型
        if not pred.strip():
            error_types.append(("无答案", i, q, ref, pred))
        elif ref in pred:
            error_types.append(("部分正确", i, q, ref, pred))
        elif q not in pred and not any(word in pred for word in q.split()[:3]):
            error_types.append(("不相关", i, q, ref, pred))
        else:
            error_types.append(("错误", i, q, ref, pred))
    
    # 3. 统计错误分布
    error_counts = {t: 0 for t in ["无答案", "部分正确", "不相关", "错误"]}
    for error in error_types:
        error_counts[error[0]] += 1
    
    return error_counts, error_types
def save_detailed_report(eval_results, error_details, output_path):
    """生成并保存详细评估报告"""
    from jinja2 import Template
    
    # HTML报告模板
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>问答系统评估报告</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .metric { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>问答系统评估报告</h1>
        
        <h2>总体指标</h2>
        <table>
            <tr><th>指标</th><th>值</th></tr>
            <tr><td>完全匹配率</td><td>{{ "%.4f"|format(metrics.exact_match) }}</td></tr>
            <tr><td>BERTScore F1</td><td>{{ "%.4f"|format(metrics.bert_score_f1) }}</td></tr>
            <tr><td>语义相似度</td><td>{{ "%.4f"|format(metrics.semantic_similarity) }}</td></tr>
            <tr><td>BLEU</td><td>{{ "%.4f"|format(metrics.bleu) }}</td></tr>
            <tr><td>ROUGE-L</td><td>{{ "%.4f"|format(metrics.rougeL) }}</td></tr>
        </table>
        
        <h2>错误分析</h2>
        <table>
            <tr><th>错误类型</th><th>数量</th><th>占比</th></tr>
            {% for err_type, count in error_counts.items() %}
            <tr>
                <td>{{ err_type }}</td>
                <td>{{ count }}</td>
                <td>{{ "%.2f%%"|format(count/total_questions*100) }}</td>
            </tr>
            {% endfor %}
        </table>
        
        <h2>错误详情 (抽样)</h2>
        <table>
            <tr>
                <th>ID</th>
                <th>错误类型</th>
                <th>问题</th>
                <th>参考答案</th>
                <th>模型答案</th>
            </tr>
            {% for error in sample_errors %}
            <tr>
                <td>{{ error[1] }}</td>
                <td>{{ error[0] }}</td>
                <td>{{ error[2] }}</td>
                <td>{{ error[3] }}</td>
                <td>{{ error[4] }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """
    
    # 准备数据
    error_counts = {}
    for err in error_details:
        error_counts[err[0]] = error_counts.get(err[0], 0) + 1
    
    # 随机抽样20个错误案例
    import random
    sample_errors = random.sample(error_details, min(20, len(error_details)))
    
    # 渲染报告
    template = Template(template_str)
    html_report = template.render(
        metrics=eval_results,
        error_counts=error_counts,
        total_questions=len(error_details),
        sample_errors=sample_errors
    )
    
    # 保存报告
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_report)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估问答系统性能")
    parser.add_argument("--qa_data", type=str, default="synthetic_data_generation_ragas_2.csv")
    parser.add_argument("--result", type=str, default="results/result_SFT.csv")
    parser.add_argument("--output_report", type=str, default="evaluation_report.html")
    
    args = parser.parse_args()
    
    # 执行评估
    qa_data = pd.read_csv(args.qa_data)
    result_data = pd.read_csv(args.result,header=None, names=["question","answer"])
    
    evaluation_results = evaluate_qa(args.qa_data, args.result)
    
    if evaluation_results:
        print("\n===== 评估结果 =====")
        print(f"完全匹配率: {evaluation_results['exact_match']:.4f}")
        print(f"BERTScore F1: {evaluation_results['bert_score_f1']:.4f}")
        print(f"语义相似度: {evaluation_results['semantic_similarity']:.4f}")
        print(f"BLEU: {evaluation_results['bleu']:.4f}")
        print(f"ROUGE-L: {evaluation_results['rougeL']:.4f}")
        
        # 错误分析
        error_counts, error_details = analyze_common_errors(qa_data, result_data)
        print("\n===== 错误分析 =====")
        for err_type, count in error_counts.items():
            print(f"{err_type}: {count} 个 ({count/len(qa_data):.2%})")
        
        # 保存详细报告
        save_detailed_report(evaluation_results, error_details, args.output_report)
        print(f"\n详细报告已保存至: {args.output_report}")
    else:
        print("评估失败，请检查输入数据")

