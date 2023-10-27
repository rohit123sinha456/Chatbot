#https://python.langchain.com/docs/modules/chains/foundational/router#legacy-routerchain
from langchain.llms import OpenAI
import yaml
from langchain.chains.router import MultiRetrievalQAChain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
import os
import sys
from langchain.utilities import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate
from langchain.llms import LlamaCpp
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
config = ""
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def read_pdf_data_and_store(folder):
    loader = PyPDFDirectoryLoader(folder)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 2})

def read_sql_data(llm,db):
    query_chain = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return query_chain#SQLDatabaseChain.from_llm(llm, db, verbose=True)

def load_models():
    llm = LlamaCpp(
        streaming = True,
        model_path="./Model/mistral-7b-instruct-v0.1.Q3_K_M.gguf",
        temperature=0.50,
        top_p=1,
        verbose=False,
        n_ctx=4096)
    return llm

def prompt_template():   
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
            """
            Use the information from the below two sources to answer any questions.
            Source 1: a SQL database about employee data
            <source1>
            {source1}
            </source1>
            You are a helpful AI assistant expert in querying SQL Database to find answers to user's question about doctors, fees and time.
            Use the following context to create a SQL query. Context :
            doctors_records : This table conatin doctors information, their name, fees and time.

            Any questions related to doctor should be answered from doctors_record table


            Source 2: a text database that contains information about admission policy
            <source2>
            {source2}
            </source2>
            """
            ),
            ("human", "{question}"),
        ]
    )
    return final_prompt

if __name__ == "__main__":
    prompt = prompt_template()
    print("Learning from Documents")
    print(os.listdir("./Data"))
    vector_store = read_pdf_data_and_store('./Data/')
    llm = load_models()
    print("Reading Database")
    db = SQLDatabase.from_uri("sqlite:///DB/test.db")
    query_chain = read_sql_data(llm,db)
    full_chain = {
    "source1": {"question": lambda x: x["question"]} | query_chain | db.run,
    "source2": (lambda x: x['question']) | vector_store,
    "question": lambda x: x['question'],
    } | prompt | llm
    response = full_chain.invoke({"question":"How many Doctors are there?"})
    print(response)
#     retriever_infos = [
#     {
#         "name": "Admission Policy Documents",
#         "description": "Good for answering questions about semantic questions about admission policy",
#         "retriever": vector_store
#     },
#     {
#         "name": "Doctor Record Database",
#         "description": "Good for answering questions about name,fees anf timings of doctors",
#         "retriever": query_chain
#     }
# ]
#     chain = MultiRetrievalQAChain.from_retrievers(llm, retriever_infos, verbose=True,default_retriever=vector_store)

#     response = chain.run(prompt.format(question="How many Doctors are there?"))
#     print(response)
#     response = chain.run(prompt.format(question="How are patients identified during admission?"))
#     print(response)