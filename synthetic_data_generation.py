from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_openai import ChatOpenAI

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

simpleqa_examples = []
factcheck_examples = []
complexqa_examples = []

OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")
 
simpleqa_prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=simpleqa_examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["question", "answer", "reference"],
    example_prompt=OPENAI_TEMPLATE,
)

factcheck_prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=factcheck_examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["sentence", "label"],
    example_prompt=OPENAI_TEMPLATE,
)

complexqa_prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=complexqa_examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["question", "answer", "reference"],
    example_prompt=OPENAI_TEMPLATE,
)


simpleqa_synthetic_data_generator = create_openai_data_generator(
    output_schema=SimpleQA,
    llm=ChatOpenAI(
        temperature=1
    ), #TODO: llm_config
    prompt=simpleqa_prompt_template,
)

fackcheck_synthetic_data_generator = create_openai_data_generator(
    output_schema=FactCheck,
    llm=ChatOpenAI(
        temperature=1
    ), #TODO: llm_config
    prompt=factcheck_prompt_template,
)

complexqa_synthetic_data_generator = create_openai_data_generator(
    output_schema=ComplexQA,
    llm=ChatOpenAI(
        temperature=1
    ), #TODO: llm_config
    prompt=complexqa_prompt_template,
)


synthetic_results = simpleqa_synthetic_data_generator.generate(
    subject="a question about a legal document",
    extra="the question and answer are related to the document",
    runs=1000,
)

print(synthetic_results)