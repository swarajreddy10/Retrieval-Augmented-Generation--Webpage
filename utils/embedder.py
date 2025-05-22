import os

import requests
from sentence_transformers import SentenceTransformer

# Load sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    return model.encode(texts, convert_to_numpy=True).tolist()

def query_llm(contexts, question):
    context_joined = "\n---\n".join(contexts)

    prompt = f"""You are an expert document analyst. Based on the context below, provide a detailed, structured, and meaningful answer to the question asked. Make sure to present the answer with clarity, use headings or bullet points if needed, and avoid vague replies.

Context:
{context_joined}

Question: {question}
Answer:
"""

    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        headers=headers,
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 500}
        }
    )
    response.raise_for_status()
    return response.json()[0]["generated_text"].split("Answer:")[-1].strip()

