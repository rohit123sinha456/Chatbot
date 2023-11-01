#https://docs.llamaindex.ai/en/stable/examples/query_engine/SQLRouterQueryEngine.html
from sqlalchemy import create_engine
from llama_index import SQLDatabase
from llama_index.indices.struct_store import NLSQLTableQueryEngine
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.tools import QueryEngineTool
import os
import torch
# from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from langchain.document_loaders import PyPDFDirectoryLoader
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import LLMSingleSelector
import sys
from langchain.llms import LlamaCpp
from llama_index import SimpleDirectoryReader
from llama_index.embeddings import resolve_embed_model
from llama_index.selectors import EmbeddingSingleSelector
from llama_index.embeddings import HuggingFaceEmbedding

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )


def read_database_and_store(service_context):
    database_path = os.path.join(os.getcwd(),"DB","test.db")
    engine = create_engine("sqlite:///"+database_path, echo=True)
    sql_database = SQLDatabase(engine)
    sql_query_engine = NLSQLTableQueryEngine(sql_database=sql_database,tables=["doctors_records"],service_context=service_context)
    return sql_query_engine

def read_pdf_data_and_store(folder,service_context):
    reader = SimpleDirectoryReader(input_dir="./Data/")
    data = reader.load_data()
    vector_index = VectorStoreIndex.from_documents(data, service_context=service_context)
    vector_query_engine = vector_index.as_query_engine()
    return vector_query_engine

def load_models():
    model_path = os.path.join(".","Model","mistral-7b-instruct-v0.1.Q3_K_M.gguf")
    print(model_path)
    llm = LlamaCpp(
        streaming = True,
        model_path=model_path,#"../Model/mistral-7b-instruct-v0.1.Q3_K_M.gguf",
        temperature=0.70,
        n_gpu_layers = 40,
        n_batch = 512,
        top_p=1,
        verbose=False,
        n_ctx=4096)
#     llm = HuggingFaceLLM(
#     model_name="mistralai/Mistral-7B-Instruct-v0.1",
#     tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
#     query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST] </s>\n"),
#     context_window=3900,
#     max_new_tokens=256,
#     #model_kwargs={"quantization_config": quantization_config},
#     generate_kwargs={"do_sample": True, "top_k": 5},
#     device_map="auto",
# )
    service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
    return service_context

def create_db_and_doc_query_tool(sql_query_engine,vector_query_engine):
    sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        "Useful for translating a natural language query into a SQL query over"
        " a table containing: doctors_records, containing the name,fees anf timings of doctors"
        " useful for answering questions about doctors name, fees and timmings "),
    )
    vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for answering semantic questions about admission policy"),
    )
    return sql_tool,vector_tool


def init_model():
    print("Loading Model..")
    service_context = load_models()
    base_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")#resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    print("Reading database")
    sql_query_engine = read_database_and_store(service_context)
    print("Reading Document")
    vector_query_engine = read_pdf_data_and_store("./Data",service_context)
    print("creating query tool")
    sql_tool,vector_tools = create_db_and_doc_query_tool(sql_query_engine,vector_query_engine)
    print("creating Query Engine")
    base_selector = EmbeddingSingleSelector.from_defaults(
    embed_model=base_embed_model)
    query_engine = RouterQueryEngine(
    selector=base_selector.from_defaults(),
    service_context=service_context,
    query_engine_tools=([sql_tool,vector_tools]),
    )
    return query_engine


if __name__ == "__main__":
    print("Loading Model..")
    service_context = load_models()
    base_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")#resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    print("Reading database")
    sql_query_engine = read_database_and_store(service_context)
    print("Reading Document")
    vector_query_engine = read_pdf_data_and_store("./Data",service_context)
    print("creating query tool")
    sql_tool,vector_tools = create_db_and_doc_query_tool(sql_query_engine,vector_query_engine)
    print("creating Query Engine")
    base_selector = EmbeddingSingleSelector.from_defaults(
    embed_model=base_embed_model)
    query_engine = RouterQueryEngine(
    selector=base_selector.from_defaults(),
    service_context=service_context,
    query_engine_tools=([sql_tool,vector_tools]),
    )

    while True:
        user_input = input(f"Input Prompt: ")
        if user_input == 'exit':
            print('Exiting')
            sys.exit()
        if user_input == '':
            continue
        response = query_engine.query(user_input)
        print(f"Answer: {str(response)}")

