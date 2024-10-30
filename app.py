import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
load_dotenv()

from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model= "llama-3.1-70b-versatile",groq_api_key=groq_api_key)

st.title("AI Research Tool 🔎📰")

st.sidebar.title("Add URLS")
urls = []
for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)
clicked = st.sidebar.button("Process URLS")


if clicked:
            
            # LOADING THE ARTICLES
            
            loader = UnstructuredURLLoader(urls= urls)
            st.spinner("Loading the articles...")
            data = loader.load()
            
            # SPLITTING THE LOADED DOCUMENTS
            
            splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000,
                chunk_overlap = 0
            )
            st.spinner("Splitting the articles to chunks...")
            chunks = splitter.split_documents(data)
            
            # Create embeddings of the chunks
            
            hf_embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
            
            # Store the embeddings in a vector database
            
            vector_db = FAISS.from_documents(documents=data,embedding=hf_embeddings)
            st.spinner("Creating the embeddings....")
            
            # Saving the embeddings in a local disk
            
            vector_db.save_local("faiss_index")
                
query = st.text_input("Question: ")

submit = st.button("Ask")
            
if query and submit:
    
                hf_embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
                vector_store =FAISS.load_local("faiss_index",hf_embeddings,allow_dangerous_deserialization=True)
            
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever= vector_store.as_retriever())
                
                result = chain.invoke({"question": query}, return_only_outputs=True)
                
                st.header("Answer")
                st.write(result["answer"])
                
                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                        st.subheader("Sources:")
                        sources_list = sources.split("\n")  # Split the sources by newline
                        for source in sources_list:
                            st.write(source)
                
        
