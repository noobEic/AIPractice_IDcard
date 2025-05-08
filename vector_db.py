from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
from typing import List

def process_and_store(docs: List[Document], 
                      embedding_model: str = "BAAI/bge-small-zh",
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

    os.makedirs(persist_dir, exist_ok=True)
    vector_db.save_local(persist_dir)

    print(f"已存储 {len(docs)} 个文档块")
    print(f"向量维度: {vector_db.embedding_function.embed_query('test').__len__()}")
    
    return vector_db

if __name__ == "__main__":

    docs = [
        Document(page_content="testtest", metadata={"source": "test1"}),
        Document(page_content="啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊", metadata={"source": "test2"})
    ]
    
    persist_dir = "./test_vector_db"
    vector_db = process_and_store(docs,persist_dir=persist_dir)