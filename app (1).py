import streamlit as st
import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv




# API Setup
gemini_api = "<put your api key in environment verialbe or you can past it here" #use os.getenv[] if you use .env file
llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro', gemini_api_key=gemini_api, temperature=0.6)

# Streamlit UI
st.title("📰 Article Reader")
st.sidebar.title("Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
urls = [url for url in urls if url]

log_placeholder = st.empty()           
query_placeholder = st.container()      
#  Event loop fix
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

#  Process URLs
if st.sidebar.button("Process URLs") and urls:
    data = []
    try:
        log_placeholder.info("📡 Fetching data from URLs...")
        for url in urls:
            loader = UnstructuredURLLoader(urls=[url])
            data.extend(loader.load())
        log_placeholder.success("✅ Articles loaded.")

        log_placeholder.info("✂️ Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(data)
        log_placeholder.success(f"✅ Split into {len(docs)} chunks.")

        log_placeholder.info("🧠 Creating embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
        vector_index = FAISS.from_documents(docs, embeddings)
        log_placeholder.success("✅ Embeddings created.")

        log_placeholder.info("💾 Saving index...")
        vector_index.save_local("faiss_index")
        log_placeholder.success("✅ Index saved to 'faiss_index'")
    except Exception as e:
        log_placeholder.error(f"❌ Processing error: {e}")

#  Query Section
if os.path.exists("faiss_index/index.faiss"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    try:
        vector_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_index.as_retriever()
        qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        #  You can ask multiple questions one after another
        query = query_placeholder.text_input("💬 Ask a question based on the articles")
        if query:
            result = qa_chain({"question": query}, return_only_outputs=True)
            query_placeholder.markdown(f"### ✅ Answer:\n{result['answer']}")
            query_placeholder.markdown(f"**📚 Source:** {result.get('sources', 'Not Found')}")
    except Exception as e:
        st.error(f"❌ Failed to load index: {e}")
else:
    st.warning("⚠️ Please process URLs before asking questions.")

