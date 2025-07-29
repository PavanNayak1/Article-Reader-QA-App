# 📰 Article Reader QA App (Streamlit + Gemini + FAISS)

This application lets you:
- 🔗 Input multiple URLs of online articles
- 📥 Automatically fetch and extract article content
- ✂️ Split and embed article text using Google's Gemini embedding model
- 🧠 Build a semantic search index with FAISS
- ❓ Ask natural language questions about the articles
- 📚 Get answers with references to the source material

---

## 🚀 Features

- Built with **Streamlit** for easy interactive UI
- Uses **Google Gemini Pro** for answering questions
- Uses **Gemini Embeddings** (`gemini-embedding-exp-03-07`) for vector representation
- Powered by **LangChain** and **FAISS** for document retrieval and semantic search
- Fully local index creation, storage, and reuse

---

## 📦 Dependencies

Install required Python libraries:

```bash
pip install -r requirements.txt

