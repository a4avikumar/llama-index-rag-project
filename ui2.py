import streamlit as st
from streamlit_chatbox import *
from streamlit_option_menu import option_menu
import time
import simplejson as json
from llama_index.readers.file import PDFReader
import fitz
from main import agent  # Import the ReActAgent instance from your main.py
import os


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


from prompts import new_prompt,instruction_str,context
from llama_index.llms.groq import Groq
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool,ToolMetadata
from llama_index.core.agent import ReActAgent



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





# Function to read the content of the note file
def read_note_file():
    note_file = os.path.join("data", "notes.txt")
    with open(note_file, "r") as f:
        return f.read()

def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    # text = ""
    # for page_num in range(len(pdf_document)):
    #     page = pdf_document.load_page(page_num)
    #     text += page.get_text()
    return pdf_document

def save_pdf_data_to_file(text, filename="pdf_data.txt"):
    with open(filename, "w") as file:
        file.write(text)

# Initialize the chatbox
chat_box = ChatBox()
chat_box.use_chat_name("chat1")  # Initialize a chat conversation

def on_chat_change():
    chat_box.use_chat_name(st.session_state["chat_name"])
    chat_box.context_to_session()  # Restore widget values to st.session_state when chat name changes

# Sidebar for chat session selection
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Chat Interface", "Notes"],  # required
        icons=["chat-dots", "file-text"],  # optional - replace with your preferred icons
        menu_icon="cast",  # optional
        default_index=0,  # default option selected
    )



    st.subheader('Start Chatting')
    # chat_name = st.selectbox("Chat Session:", ["default", "chat1"], key="chat_name", on_change=on_chat_change)
    # chat_box.use_chat_name(chat_name)
    # streaming = st.checkbox('Streaming', key="streaming")
    in_expander = st.checkbox('Show Messages in Expander', key="in_expander")
    chat_box.context_from_session(exclude=["chat1"])  # Save widget values to chat context

    st.divider()

    # File uploader to load chat history (optional)
    file = st.file_uploader("Load your pdf file", type=["pdf"])

    if st.button("Load PDF") and file:
        data = read_pdf(file)
        # save_pdf_data_to_file(data)
        st.success("PDF file successfully loaded!")
        pakistan_index=get_index(data,"pakistan")
        pakistan_engine=pakistan_index.as_query_engine(llm=llama2)
        tools = [
        note_engine,
        # QueryEngineTool(
        #     query_engine=population_query_engine,
        #     metadata=ToolMetadata(
        #         name="population_data",
        #       Scaled Dot-Product Attention  description="this gives information at the world population and demographics",
        #     ),
        # ),
        QueryEngineTool(
           query_engine=pakistan_engine,
           metadata=ToolMetadata(
               name="pdf",
               description="this gives detailed information in the pdf file given",
           ),
        ),
        ]
        st.session_state.agent2 = ReActAgent.from_tools(tools, llm=llama2, verbose=True, context=context)

    
        













        
# Initialize chat session and display messages
if selected == "Chat Interface":

    st.title("Chat Interface")

    chat_box.init_session()
    chat_box.output_messages()

    def on_feedback(feedback, chat_history_id: str = "", history_index: int = -1):
        reason = feedback["text"]
        chat_box.set_feedback(feedback=feedback, history_index=history_index)  # Convert emoji to integer
        st.session_state["need_rerun"] = True

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "Provide Feedback",
    }

    # Handle user input
    if query := st.chat_input('Input your question here'):
        chat_box.user_say(query)
        # if streaming:
        #     # Streaming response from agent
        #     generator = agent.query(query, stream=True)
        #     elements = chat_box.ai_say(
        #         [
        #             Markdown("Thinking...", in_expander=in_expander, expanded=True, title="Answer"),
        #             Markdown("", in_expander=in_expander, title="References"),
        #         ]
        #     )
        #     time.sleep(1)
        #     text = ""
        #     for x, docs in generator:
        #         text += x
        #         chat_box.update_msg(text, element_index=0, streaming=True)
        #     chat_box.update_msg(text, element_index=0, streaming=False, state="complete")
        #     chat_box.update_msg("\n\n".join(docs), element_index=1, streaming=False, state="complete")
        #     chat_history_id = "some id"
        #     chat_box.show_feedback(**feedback_kwargs,
        #                             key=chat_history_id,
        #                             on_submit=on_feedback,
        #                             kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})
        # else:
        # Non-streaming response from agent
        if "agent2" in st.session_state:
            response = st.session_state.agent2.query(query)
        else:
            response = agent.query(query)
        text = response.response
        # docs = response.references
        chat_box.ai_say(
            [
                Markdown(text, in_expander=in_expander, expanded=True, title="Answer"),
                # Markdown("\n\n".join(docs), in_expander=in_expander, title="References"),
            ]
        )

    # Button to clear chat history
    # if st.button("Clear History"):
    #     chat_box.init_session(clear=True)
    #     st.experimental_rerun()

    # Optional: Show session state for debugging
    # if st.checkbox('Show Session State'):
    #     st.write(st.session_state)
elif selected == "Notes":
    st.title("Notes Section")
    
    # Display the content of the notes file
    data = read_note_file()
    st.write(data)
