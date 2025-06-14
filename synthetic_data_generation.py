import os
import faiss
import numpy as np
import pickle
import json
from typing import List, Dict, Any, Optional, Type

from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from prompt import (
    SIMPLEQA_PREFIX, SIMPLEQA_SUFFIX,
    COMPLEXQA_PREFIX, COMPLEXQA_SUFFIX,
    FACTCHECK_PREFIX, FACTCHECK_SUFFIX
)

class SimpleQA(BaseModel):
    question: str
    answer: str
    reference: str
    
class FactCheck(BaseModel):
    statement: str
    label: bool
    reference: str
    
class ComplexQA(BaseModel):
    question: str
    answer: str
    reference: list[str]

simpleqa_examples = [
    {
     	"question": "临时居民身份证的有效期是多久？",
     	"answer": "临时居民身份证的有效期为三个月。",
        "reference": "临时居民身份证的有效期限为三个月，有效期限自签发之日起计算。"
    }
]

complexqa_examples = [
    {
     	"question": "广东省内如何办理临时居民身份证？需要哪些材料和办理流程？",
     	"answer": "广东省户籍居民在申领、换领或补领居民身份证期间急需使用身份证件的，可携带《居民身份证领取凭证》原件、户口簿原件及复印件、近期免冠彩色照片1张，前往户籍所在地或现居住地公安机关户政窗口申请办理临时居民身份证，缴纳10元工本费后即可现场领取，该临时身份证有效期为3个月。",
     	"reference": ["临时居民身份证的有效期限为三个月，有效期限自签发之日起计算。", "公民申请领取临时居民身份证应当缴纳证件工本费。工本费标准由国务院价格主管部门会同国务院财政部门核定。", "广东省户籍居民可在省内任一公安机关户政窗口办理临时居民身份证业务，实现全省通办。"]
    }
]
factcheck_examples = [
    {
     	"statement": "所有省份均可通过广东省办理跨省通办身份证业务。",
     	"label": False,
         "reference": "北京、辽宁等未对接地区暂不能受理跨省通办。"
    }
]



class FAISSRetriever(BaseRetriever):
    def __init__(self, faiss_index: faiss.Index, documents: List[Document], embedding_model: Embeddings, k: int = 4):
        if len(documents) != faiss_index.ntotal:
            raise ValueError("Number of documents must match the number of vectors in the FAISS index.")
        self.faiss_index = faiss_index
        self.documents = documents
        self.embedding_model = embedding_model
        self.k = k
        print(f"FAISSRetriever initialized with index containing {faiss_index.ntotal} documents.")

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        print(f"Executing retrieval for query: '{query}' using FAISS...")
        try:
            query_vector = self.embedding_model.embed_query(query)
            query_vector = np.array([query_vector], dtype='float32')
            distances, indices = self.faiss_index.search(query_vector, self.k)
            retrieved_docs = [self.documents[i] for i in indices[0] if i != -1]
            print(f"Retrieved {len(retrieved_docs)} documents from FAISS.")
            return retrieved_docs
        except Exception as e:
            print(f"An error occurred during FAISS retrieval for query '{query}': {e}")
            return []
        
class RAGPromptBuilder:
    def __init__(
        self,
        prefix_content: str,
        suffix_content: str,
        example_template: str = "{example_json_string}"
    ):
        self.prompt_template = PromptTemplate(
            input_variables=[
                "prefix_content", 
                "suffix_content", 
                "context",
                "examples",
                "schema",
                "num_samples",
                "extra",
                "subject"
            ],
            template=(
                "{prefix_content}\n\n"
                "--- 参考上下文 ---\n{context}\n---\n\n"
                "请参考以下示例的格式（JSON数组）：\n{examples}\n\n"
                "Schema:\n{schema}\n\n"
                "请根据上面的参考上下文、示例和Schema，生成 {num_samples} 个新的数据样本。\n"
                "{extra}\n"
                "Subject: {subject}\n\n"
                "Output JSON array here (only the JSON, no markdown or extra text):\n"
                "{suffix_content}"
            )
        )
        self.example_template = example_template
        self.prefix_content = prefix_content
        self.suffix_content = suffix_content
        print("RAGPromptBuilder initialized with specific prefix/suffix content.")


    def build_prompt(
        self,
        context: str,
        schema: str,
        examples: List[Dict[str, Any]],
        num_samples: int = 1,
        extra: Optional[str] = None,
        subject: Optional[str] = None
    ) -> str:

        formatted_examples_list = [
            self.example_template.format(example_json_string=json.dumps(ex, ensure_ascii=False))
            for ex in examples
        ]
        formatted_examples_str = "\n".join(formatted_examples_list)

        full_prompt = self.prompt_template.format(
            prefix_content=self.prefix_content,
            suffix_content=self.suffix_content,
            context=context,
            examples=formatted_examples_str,
            schema=schema,
            num_samples=num_samples,
            extra=extra if extra is not None else "",
            subject=subject if subject is not None else ""
        )

        return full_prompt.strip()


