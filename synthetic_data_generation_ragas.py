import os

from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from langchain_ollama import OllamaEmbeddings
from ragas.testset.persona import Persona
from api import BASE_URL, API_KEY, OPENAI_API_KEY,OPENAI_BASE_URL

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from splitter import LawSplitter
# loader = UnstructuredWordDocumentLoader("reference/关于传发广东省户口居民身份证管理工作操作规范（2024年版）的通知【正文】4433808.doc")
# documents = loader.load()
# splitter = LawSplitter()
# docs = splitter.split_documents(documents)
# print(docs.__len__())

path = "./reference/"
loader = DirectoryLoader(path, glob="**/*.doc")
docs = loader.load()

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="deepseek-r1",base_url=BASE_URL,api_key=API_KEY))
generator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(
    model="bge-large",
    base_url="http://localhost:11434"
))
personas = [
    Persona(
        name="基础咨询者",
        role_description="办理身份证时询问基础政策信息（如有效期、费用等）的普通市民",
    ),
    Persona(
        name="流程咨询者",
        role_description="需要详细了解跨省办理、特殊情形流程等复杂问题的市民",
    ),
    Persona(
        name="政策核实者",
        role_description="对已获取的身份证办理政策信息进行真实性核对的谨慎市民",
    )
]

generator = TestsetGenerator(llm=generator_llm,embedding_model=generator_embeddings,persona_list=personas)
dataset = generator.generate_with_langchain_docs(docs, testset_size=2000)



dataset.to_csv("synthetic_data_generation_ragas_2.csv")
print("Synthetic data generation completed and saved to 'synthetic_data_generation_ragas_2.csv'.")