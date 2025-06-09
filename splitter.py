import os
import re
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.docstore.document import Document
from typing import Any, Iterable, List


import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import Any, Iterable, List

class LawSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32, **kwargs: Any):
        separators = [
            r'\n[一二三四五六七八九十]+、\s*',
            r'\n（[一二三四五六七八九十]+）\s*',
            r'\n\d+\.\s*',
            r'\n（\d+）\s*',
            r'\n[①②③④⑤⑥⑦⑧⑨⑩]+\s*',
            r'\n[●•○▪▫]\s*',
            r'[。！？；]\s*',
            r'\n',
            r'\s{4,}',
        ]
        
        super().__init__(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=True,
            keep_separator=True,
            **kwargs
        )
        
        # 更细粒度的层级映射
        self.level_mapping = {
            r'^[一二三四五六七八九十]+、': 'level1',
            r'^（[一二三四五六七八九十]+）': 'level2',
            r'^\d+\.': 'level3',
            r'^（\d+）': 'level4',
            r'^[①②③④⑤⑥⑦⑧⑨⑩]': 'level5',
            r'^[●•○▪▫]': 'bullet',
        }

    def split_text(self, text: str) -> List[str]:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        chunks = super().split_text(text)
        
        processed = []
        for chunk in chunks:
            if len(chunk) > self._chunk_size * 1.5:
                sentences = re.split(r'(?<=[。！？；])', chunk)
                current_block = ""
                for sentence in sentences:
                    if len(current_block) + len(sentence) > self._chunk_size:
                        if current_block:
                            processed.append(current_block.strip())
                            current_block = sentence
                        else:
                            processed.append(sentence.strip())
                    else:
                        current_block += sentence
                if current_block:
                    processed.append(current_block.strip())
            else:
                processed.append(chunk.strip())
                
        return processed

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        texts = []
        metadatas = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                if not chunk.strip() or len(chunk.strip()) < 10:
                    continue
                    
                texts.append(chunk)
                new_metadata = doc.metadata.copy()
                
                level = self._detect_level(chunk)
                if level:
                    new_metadata["level"] = level

                    title_line = chunk.split('\n')[0].strip()
                    new_metadata["title"] = title_line
                
                metadatas.append(new_metadata)
        
        return self.create_documents(texts, metadatas=metadatas)

    def _detect_level(self, text: str) -> str:
        first_line = text.split('\n')[0].strip()
        for pattern, level in self.level_mapping.items():
            if re.search(pattern, first_line):
                return level
        return ""
    
    
if __name__ == "__main__":
    loader = UnstructuredWordDocumentLoader("reference/关于传发广东省户口居民身份证管理工作操作规范（2024年版）的通知【正文】4433808.doc")
    documents = loader.load()
    splitter = LawSplitter()
    split_docs = splitter.split_documents(documents)
    print(split_docs.__len__())
    