
import streamlit as st
import os
import sys
import tempfile
from dotenv import load_dotenv
from pathlib import Path
#from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import  RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from streamlit_option_menu import option_menu
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.globals import set_verbose, set_debug
from langchain import hub

#Create temporary folder for document Storage
TMP_DIR_PDF = Path(__file__).resolve().parent.parent.joinpath('external_data', 'pdf_docs')

#Sidebar contents
with st.sidebar:
    st.title('LLM Chat App ☕')
    st.markdown('''
    ## About
    This app is an LLM powered chatbot built using
    - [Streamlit] (https://streamlit.io/)
    - [Langchain] 
    - [Open Source LLM, Vector store and embeddings]
    ''')

    add_vertical_space(5)
    st.write('Made by Sumit')


def main():
    st.header('Chat with PDF')

    st.write('Please Upload a document')
    source_doc = st.file_uploader(label="Upload a document", type='pdf')
    # right now we have source_doc which is a BytesIO object
    if source_doc is not None:
        
        load_dotenv()
        os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

        #save the file locally
        with open(f"{TMP_DIR_PDF}\{source_doc.name}", mode='wb') as w:
            w.write(source_doc.getvalue())

        #pass this file path to the loader
        if source_doc: #check if file path is not None
            loader = PyPDFLoader(f"{TMP_DIR_PDF}\{source_doc.name}")
            doc = loader.load()

            #st.write(doc)
        
            #go for chunking
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200, length_function=len)
            splits = text_splitter.split_documents(doc)

            #st.write(splits[:1])

            #convert the chunks into vector embeddings and store them in a vectorStore; We will use FAISS
            db = FAISS.from_documents(splits[:4], OpenAIEmbeddings())
            retriever = db.as_retriever()

            #st.write(retriever)

            prompt = hub.pull("rlm/rag-prompt")
            
            llm = Ollama(model="llama3.1")

            ### Contextualize question ###
            contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )


            ### Answer question ###
            qa_system_prompt = """You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\

            {context}"""
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            ### Statefully manage chat history ###
            #store = {}
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history=[]            

            # def get_session_history(session_id: str) -> BaseChatMessageHistory:
            #     if session_id not in store:
            #         store[session_id] = ChatMessageHistory()
            #     return store[session_id]


            # conversational_rag_chain = RunnableWithMessageHistory(
            #     rag_chain,
            #     get_session_history,
            #     input_messages_key="input",
            #     history_messages_key="chat_history",
            #     output_messages_key="answer",
            # )

            #continued conversation like appearance in UI
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.markdown(message.content)
                else:
                    with st.chat_message("AI"):
                        st.markdown(message.content)

            # if "historical_chat" not in st.session_state:
            #     st.session_state.historical_chat = []
            query = st.chat_input("Ask relevant questions on your document.")
                                        
            if query is not None and query !="":    

                # st.session_state.historical_chat.append(HumanMessage(query))
                st.session_state.chat_history.append(HumanMessage(query))

                with st.chat_message("Human"):
                    st.markdown(query)

                ai_response = rag_chain.invoke({"input":query,"chat_history":st.session_state.chat_history})["answer"]

                # ai_response = conversational_rag_chain.invoke(
                #     {"input": query},
                #     config={
                #         "configurable": {"session_id": "abc123"}
                #     },  # constructs a key "abc123" in `store`.
                # )["answer"]
                
                with st.chat_message("AI"):                    
                    st.markdown(ai_response)

                #st.session_state.historical_chat.append(AIMessage(ai_response))
                st.session_state.chat_history.append(AIMessage(ai_response))

        
        
       
if __name__ == '__main__':
    main()