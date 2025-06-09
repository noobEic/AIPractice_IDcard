from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
from typing import List
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from splitter import LawSplitter

def process_and_store(docs: List[Document], 
                      embedding_model: str = "BAAI/bge-large-zh",
                      persist_dir: str = "./vector_db"):

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cuda:0'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_db = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    print(f"索引中文档数量: {vector_db.index.ntotal}")
    print(f"文档存储中文档数量: {len(vector_db.docstore._dict)}")
    
    os.makedirs(persist_dir, exist_ok=True)
    vector_db.save_local(persist_dir)

    print(f"已存储 {len(docs)} 个文档块")
    print(f"向量维度: {vector_db.embedding_function.embed_query('test').__len__()}")
    
    return vector_db

if __name__ == "__main__":

    loader = UnstructuredWordDocumentLoader("reference/关于传发广东省户口居民身份证管理工作操作规范（2024年版）的通知【正文】4433808.doc")
    documents = loader.load()
    splitter = LawSplitter()
    docs = splitter.split_documents(documents)
    
    persist_dir = "./test_vector_db"
    vector_db = process_and_store(docs,persist_dir=persist_dir)