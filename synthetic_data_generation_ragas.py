import os

from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator

from api import BASE_URL, API_KEY
path = "reference/"
loader = DirectoryLoader(path, glob="**/*.doc")
docs = loader.load()

print(docs.__len__())


generator_llm = LangchainLLMWrapper(ChatOpenAI(model="deepseek-r1",base_url=BASE_URL,api_key=API_KEY))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="deepseek-r1",base_url=BASE_URL,api_key=API_KEY))

generator = TestsetGenerator(llm=generator_llm,embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)


dataset.to_pandas()
dataset.to_csv("synthetic_data_generation_ragas.csv", index=False)
print("Synthetic data generation completed and saved to 'synthetic_data_generation_ragas.csv'.")