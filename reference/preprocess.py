# -*- coding: utf-8 -*-
import re
from pathlib import Path
from unstructured.partition.doc import partition_doc
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import List, Dict, Any

def process_legal_doc(doc_path: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[Dict[str, Any]]:

    # 1. 提取文本和结构化元素
    elements = partition_doc(doc_path)
    
    # 2. 清洗文本（移除页眉、修订标记等）
    cleaned_text = clean_legal_text("\n".join([e.text for e in elements]))
    
    # 3. 转换为 Markdown 格式（模拟标题结构）
    md_text = convert_to_markdown(elements)
    
    # 4. 按标题层级分块（Section/Clause）
    markdown_chunks = split_markdown_by_headers(md_text)
    
    # 5. 对长段落进一步分块（添加重叠）
    final_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", " "]
    )
    for chunk in markdown_chunks:
        sub_chunks = text_splitter.split_text(chunk["content"])
        for sub_chunk in sub_chunks:
            final_chunks.append({
                "content": sub_chunk,
                "metadata": chunk["metadata"]
            })
    
    return final_chunks

def clean_legal_text(text: str) -> str:
    text = re.sub(r'^.*(CONFIDENTIAL|Page\s+\d+).*$', '', text, flags=re.MULTILINE)
    # 移除修订标记（如 [DEL: ... ] 或 {删除内容}）
    text = re.sub(r'(\[DEL:.*?\])|({.*?})', '', text)
    # 合并多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def convert_to_markdown(elements: List[Any]) -> str:
    """将提取的元素转换为 Markdown 格式（保留标题层级）"""
    md_lines = []
    for element in elements:
        if element.category == "Title":
            md_lines.append(f"# {element.text}")
        elif element.category == "Section":
            md_lines.append(f"## {element.text}")
        elif element.category == "Subsection":
            md_lines.append(f"### {element.text}")
        else:
            md_lines.append(element.text)
    return "\n\n".join(md_lines)

def split_markdown_by_headers(md_text: str) -> List[Dict]:
    """按 Markdown 标题分割文档（保留元数据）"""
    headers_to_split_on = [
        ("##", "Section"),
        ("###", "Subsection")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False 
    )
    chunks = markdown_splitter.split_text(md_text)
    
    return [
        {
            "content": chunk.page_content,
            "metadata": {**chunk.metadata, "source": Path(doc_path).name}
        }
        for chunk in chunks
    ]

if __name__ == "__main__":
    doc_path = "关于传发广东省户口居民身份证管理工作操作规范（2024年版）的通知【正文】4433808.doc"  # 替换为实际文件路径
    chunks = process_legal_doc(doc_path)
    
    # 打印前 3 个分块
    print(f"共生成 {len(chunks)} 个分块\n")
    for i, chunk in enumerate(chunks[:3]):
        print(f"分块 {i+1}:")
        print(f"元数据: {chunk['metadata']}")
        print(f"内容: {chunk['content'][:200]}...\n")