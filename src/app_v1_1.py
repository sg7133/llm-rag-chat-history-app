
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

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader

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

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            #st.write(llm)
            
            rag_chain =(
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            query = st.text_input("Ask question about your pdf file")

            if query:
                st.write(rag_chain.invoke(query))
        
       
if __name__ == '__main__':
    main()