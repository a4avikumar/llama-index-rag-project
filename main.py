import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt,instruction_str,context
from llama_index.llms.groq import Groq
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool,ToolMetadata
from llama_index.core.agent import ReActAgent
from pdf import pakistan_engine

llama2 = Groq(model="llama-3.1-70b-versatile", api_key="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")


population_path=os.path.join("data","population.csv")
population_df=pd.read_csv(population_path)

population_query_engine=PandasQueryEngine(
    df=population_df,verbose=True,instruction_str=instruction_str,llm=llama2
)

population_query_engine.update_prompts({"pandas_prompt":new_prompt})
# response=population_query_engine.query("what is the population of canada")


tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics",
        ),
    ),
    QueryEngineTool(
       query_engine=pakistan_engine,
       metadata=ToolMetadata(
           name="pakistan_data",
           description="this gives detailed information about pakistan the country",
       ),
    ),
]

agent=ReActAgent.from_tools(tools,llm=llama2,verbose=True,context=context)
print(agent)

# while (prompt := input("Enter a prompt (q to quit): ")) != "q":
#     result = agent.query(prompt)
#     print(result)
