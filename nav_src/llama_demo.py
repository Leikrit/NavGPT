from langchain import PromptTemplate, LLMChain
from LLMs.Langchain_llama import Custom_Llama
import os

ckpt_dir = "LLMs/llama/Llama-2-7b"
tokenizer_path = "LLMs/llama/Llama-2-7b/tokenizer.model"

llm = Custom_Llama.from_model_id(
        temperature=0.75,
        ckpt_dir = ckpt_dir,
        tokenizer_path = tokenizer_path,
        max_seq_len = 4000,
        max_gen_len = 800,
        max_batch_size = 4,
    )

template = """Question: {question}\nAnswer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is electroencephalography?"
print(llm_chain.run(question))
