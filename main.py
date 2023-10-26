from langchain.llms import OpenAI
import yaml
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
import os
import sys

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
    return vector_store

def load_models():
    llm = LlamaCpp(
        streaming = True,
        model_path="./Model/mistral-7b-instruct-v0.1.Q3_K_M.gguf",
        temperature=0.50,
        top_p=1,
        verbose=False,
        n_ctx=4096)
    return llm
print("Learning from Documents")
print(os.listdir("./Data"))
vector_store = read_pdf_data_and_store('./Data/')
llm = load_models() #HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", huggingfacehub_api_token=config['HUGGINGFACE']['APIKEY'])
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 2}))
# query = "how are patients identified?"
# print(qa.run(query))
while True:
    user_input = input(f"Input Prompt: ")
    if user_input == 'exit':
        print('Exiting')
        sys.exit()
    if user_input == '':
        continue
    result = qa({'query': user_input})
    print(f"Answer: {result['result']}")


