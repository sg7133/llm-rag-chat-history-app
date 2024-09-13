Version = 1.1

## Release Notes


1. Upload single PDF document to a folder and use RAG and vectorstore to ask relevant questions from the document
2. Made the output of the llm cleaner by hiding the RAG context and question being repeated in the llm response
3. add capability to chat with contex



## Backlog
- add capability to add multiple documents of different formats 
- Divide the logic into 2 tiers and run client and api separately on local machine
- Take the 2 tier app and deploy it on a server on a cloud and give access to client
- Fortify code
    a. check if same PDF is uploaded again to avoid storing in vectorestore
    b. check if the document being uploaded is a pdf


## Infra

1. Langchain for orcehstration
2. Streamlit for UI
3. Local Llama for LLM
4. FAISS for vector store
5. OpenAI for embeddings to store in vector store
6. Recurssive Character Text splitter for chunking


### IMPORTANT
1. Download Ollama to run local Llama
2. Register to OpenAI for API Keys
3. Get Langsmith API key for monitoring
