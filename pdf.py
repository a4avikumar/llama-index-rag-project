import os
from llama_index.core import StorageContext,VectorStoreIndex,load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)

# loads BAAI/bge-small-en
# embed_model = HuggingFaceEmbedding()
# loads BAAI/bge-small-en-v1.5
# from main import llama2
from llama_index.llms.groq import Groq

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llama2 = Groq(model="llama-3.1-70b-versatile", api_key="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")
# service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llama2)





def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True,embed_model=embed_model)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name),
            embed_model=embed_model
        )
    return index


pdf_path=os.path.join("data","Pakistan.pdf")
pakistan_pdf=PDFReader().load_data(file=pdf_path)

# text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
# nodes = text_splitter.get_nodes_from_documents(, show_progress=True)


pakistan_index=get_index(pakistan_pdf,"pakistan")
pakistan_engine=pakistan_index.as_query_engine(llm=llama2)


