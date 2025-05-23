# 📄 Ask the Docs – Retrieval-Augmented Generation (RAG) Web App

Ask the Docs is a fully functional, intelligent, and responsive RAG-based mini web application that allows users to upload documents and ask context-aware questions. Built as part of a Cloud & Backend Engineering internship challenge, the system extracts, embeds, and retrieves relevant data to generate accurate and informative answers using a free, open-source LLM.

## 🚀 Features

- Upload PDF or text files
- Chunk & embed documents into vector format
- Semantic search using FAISS
- Context-aware question answering using a free open-source LLM
- Intuitive UI powered by Streamlit
- Deployed on Render and AWS EC2 with Elastic IP for 24/7 accessibility

---

## 🛠️ Tech Stack & Tools

### 🧠 Backend Logic
- **LangChain** – Simplified RAG pipeline
- **SentenceTransformers** – `all-MiniLM-L6-v2` for high-speed, low-latency embeddings
- **Transformers** – Hugging Face Transformers for model inference
- **FAISS** – Facebook AI Similarity Search for efficient vector storage and retrieval
- **PyPDF2** – Extract text from PDF documents
- **dotenv & requests** – For secure environment variable handling and API communication

### 💻 Frontend & Deployment
- **Streamlit** – UI framework for interactive web interface
- **Python 3.8+**
- **Render** – For quick public deployment
- **AWS EC2 with Elastic IP** – Scalable backend hosting with persistent public access

---

## 🤖 Free LLM Used

- `mistralai/Mistral-7B-Instruct-v0.1` *(or)* `tiiuae/falcon-7b-instruct` *(or)* `HuggingFaceH4/zephyr-7b-beta`
- Hosted via Hugging Face Inference API
- **No API key required** under free tier for limited usage

These LLMs are instruction-tuned and support English Q&A with high performance and relatively low memory use, ideal for RAG workflows.

---

## ⚙️ Workflow

1. **Upload Document**: User selects a PDF or text file.
2. **Text Extraction & Chunking**: Large documents are split into chunks for processing.
3. **Embedding**: Each chunk is converted into a vector using `all-MiniLM-L6-v2`.
4. **Storage in FAISS**: Vectors are stored in a searchable vector database.
5. **User Query Input**: Natural language question is entered by the user.
6. **Relevant Context Retrieval**: FAISS returns top-matching chunks.
7. **Answer Generation**: An LLM (Mistral or Falcon) generates a precise answer using retrieved context.

---

## 🌐 Deployment

This app is hosted on:
- **DEMO Video Drive Link** -  https://drive.google.com/file/d/1K4QhtA9cv68eZOjmklLlcu5CWqM_Ztif/view?usp=sharing
- **Render** – https://rag-app-w9pz.onrender.com
- **AWS EC2 with Elastic IP** – Provides persistent and scalable deployment with public access. The instance runs a production-grade backend configured with Gunicorn and Nginx for better performance and uptime.

---

## 📦 Installation (Local)

```bash
git clone https://github.com/Your-name/ASKtheDOCS_Retrieval-Augmented-Generation
cd ASKtheDOCS_Retrieval-Augmented-Generation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
