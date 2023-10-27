#https://towardsdatascience.com/talk-to-your-sql-database-using-langchain-and-azure-openai-bb79ad22c5e2
from langchain.utilities import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate
from langchain.llms import LlamaCpp
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

def load_models():
    llm = LlamaCpp(
        streaming = True,
        model_path="./Model/mistral-7b-instruct-v0.1.Q3_K_M.gguf",
        temperature=0.50,
        top_p=1,
        verbose=False,
        n_ctx=4096)
    return llm

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
          You are a helpful AI assistant expert in querying SQL Database to find answers to user's question about doctors, fees and time.
          Use the following context to create a SQL query. Context :
          doctors_records : This table conatin doctors information, their name, fees and time.

          Any questions related to doctor should be answered from doctors_record table
         """
         ),
        ("user", "{question}\n ai: "),
    ]
)


db = SQLDatabase.from_uri("sqlite:///DB/test.db")
llm = load_models()
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)






agent_executor.run(final_prompt.format(
        question="Is there any female doctors?"
  ))