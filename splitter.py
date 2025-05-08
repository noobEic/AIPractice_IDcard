import os
import re
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.docstore.document import Document
from typing import Any, Iterable, List


class LawSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs: Any):
        # 定义中文法律文档层级分隔符（按优先级排序）
        separators = [
            # 一级标题：一、二、...
            r'\n[一二三四五六七八九十]+、\s*',
            
            # 二级标题：（一）（二）...
            r'\n（[一二三四五六七八九十]+）\s*',
            
            # 三级标题：1. 2. ...
            r'\n\d+\.\s*',
            
            # 四级标题：（1）（2）...
            #r'\n（\d+）\s*',
            
            # 五级标题：① ② ...
            #r'\n[①②③④⑤⑥⑦⑧⑨⑩]+\s*',
            

        ]
        
        super().__init__(
            separators=separators,
            is_separator_regex=True,
            keep_separator=True,  # 保留标题标记
            **kwargs
        )
        
        # 标题级别映射
        self.level_mapping = {
            r'^[一二三四五六七八九十]+、': 'level1',
            r'^（[一二三四五六七八九十]+）': 'level2',
            r'^\d+\.': 'level3',
            #r'^（\d+）': 'level4',
            #r'^[①②③④⑤⑥⑦⑧⑨⑩]': 'level5',
        }

    def split_text(self, text: str) -> List[str]:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        chunks = super().split_text(text)
        
        return self._postprocess_chunks(chunks)

    def _postprocess_chunks(self, chunks: List[str]) -> List[str]:
        processed = []
        buffer = ""
        
        for chunk in chunks:
            current_level = self._detect_level(chunk)
            
            if not buffer:
                buffer = chunk
            else:
                if current_level:
                    processed.append(buffer.strip())
                    buffer = chunk
                else:
                    buffer += "\n" + chunk
        
        if buffer:
            processed.append(buffer.strip())
            
        return processed

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        texts = []
        metadatas = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                texts.append(chunk)
                new_metadata = doc.metadata.copy()
                level = self._detect_level(chunk)
                if level:
                    new_metadata["level"] = level
                    title = chunk.split('\n')[0].strip()
                    new_metadata["title"] = title
                
                metadatas.append(new_metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _detect_level(self, text: str) -> str:
        first_line = text.split('\n')[0].strip()
        for pattern, level in self.level_mapping.items():
            if re.search(pattern, first_line):
                return level
        return ""
    
    
if __name__ == "__main__":
    # Test the LawSplitter
    loader = UnstructuredWordDocumentLoader("reference/关于传发广东省户口居民身份证管理工作操作规范（2024年版）的通知【正文】4433808.doc")
    documents = loader.load()
    splitter = LawSplitter()
    split_docs = splitter.split_documents(documents)
    print(split_docs.__len__())
    for doc in split_docs:
        print(doc.metadata)
        print(doc.page_content.__len__())