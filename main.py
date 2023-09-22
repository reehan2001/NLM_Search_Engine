import os
import time
import pickle
import streamlit as st
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS




from dotenv import load_dotenv
load_dotenv()

status_bar = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)
file_path = "faiss_store_openai.pkl"


st.title("Reehans - Ai-bot")
st.sidebar.title("Paste the website URL")

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


process_button = st.sidebar.button("Load")

if process_button:
    loader = UnstructuredURLLoader(urls)
    data = loader.load()
    status_bar.text("Starting to fetch the data from the Website.....")
    time.sleep(2)

    ## spliting the data

    text_spliting = RecursiveCharacterTextSplitter(
            separators=["\n\n","\n",".",","],
            chunk_size=1000
    )
    status_bar.text("Scaning the text from the website.....")
    docs = text_spliting.split_documents(data)
    time.sleep(2)

    ##embedding process...
    embeddings = OpenAIEmbeddings()
    reehan_ai = FAISS.from_documents(docs,embeddings)
    status_bar.text("Reehan-ai is ready to answer the Questions....")
    time.sleep(2)

    with open(file_path,"wb") as f:
         pickle.dump(reehan_ai,f)


query = status_bar.text_input("Questions")
if query:
    if os.path.exists(file_path):
        with open(file_path,f) as f:
            ai_store = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=ai_store.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])



   





    




