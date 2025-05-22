import os

from PyPDF2 import PdfReader


def load_and_chunk_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            reader = PdfReader(filepath)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue

        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        documents.extend([(chunk, {"source": filename}) for chunk in chunks])
    return documents