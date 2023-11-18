from dotenv import load_dotenv
from llama_index import (
  ObsidianReader, GPTKeywordTableIndex,
  GPTVectorStoreIndex
) 
load_dotenv()



NOTES_PATH = 'E:/CV Dimi/from BA to DS/Lang Chain'

docs = ObsidianReader(NOTES_PATH).load_data()

index = GPTVectorStoreIndex.from_documents(docs)
index.index_struct.index_id = "journals"


query_engine = index.as_query_engine()
res = query_engine.query("How long is the first video of the LLM series")
print(index.index_struct)
