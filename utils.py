import os

import faiss
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# HuggingFace cache config
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/hf_cache")

# Load embedding model
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load open LLM model (Flan-T5)
llm_model = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(llm_model)
model = AutoModelForSeq2SeqLM.from_pretrained(llm_model)

class DocumentProcessor:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.index = None

    def process_file(self, file):
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            text = " ".join([page.extract_text() for page in reader.pages])
        else:
            text = file.read().decode("utf-8", errors="ignore")

        self.chunks = self.split_text(text)
        self.embeddings = embedder.encode(self.chunks)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

    @staticmethod
    def split_text(text, chunk_size=500, overlap=50):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    def find_relevant_chunks(self, question, top_k=3):
        question_embed = embedder.encode([question])
        distances, indices = self.index.search(np.array(question_embed), top_k)
        return [self.chunks[i] for i in indices[0]]

def generate_answer(question, context):
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
