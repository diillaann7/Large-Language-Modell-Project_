import os
from typing import List
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from models import generate_answer

class RAG:
    def __init__(self, pdf_paths: List[str], embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.texts = [self._read_pdf(p) for p in pdf_paths]
        self.model = SentenceTransformer(embedding_model_name)
        self.embeddings = np.array([self.model.encode(t) for t in self.texts])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def _read_pdf(self, file_path: str) -> str:
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Fehler beim Lesen von {file_path}: {e}")
        return text.strip()

    def retrieve(self, query: str, top_k: int = 1) -> List[str]:
        query_emb = np.array([self.model.encode(query)])
        D, I = self.index.search(query_emb, k=top_k)
        return [self.texts[i] for i in I[0]]

    def answer_rag(self, query: str, model_name: str = "gpt4", top_k: int = 1) -> str:
        docs = self.retrieve(query, top_k)
        context = "\n".join(docs)
        prompt = f"Nutze den folgenden Text um die Frage zu beantworten:\n{context}\nFrage: {query}"
        return generate_answer(model_name, prompt)

    def answer_non_rag(self, query: str, model_name: str = "gpt4") -> str:
        return generate_answer(model_name, query)

#Test
if __name__ == "__main__":
    pdfs = ["hunde.pdf", "erneuerbare_energien.pdf", "ki.pdf", "gesundheit.pdf", "weltkulturen.pdf"]  
    frage = "Was fressen Hunde?"

    rag_system = RAG(pdfs)

    print("=== Non-RAG GPT-4 ===")
    print(rag_system.answer_non_rag(frage, model_name="gpt4"))

    print("\n=== RAG GPT-4 ===")
    print(rag_system.answer_rag(frage, model_name="gpt4", top_k=2))

    print("\n=== Non-RAG Gemini ===")
    print(rag_system.answer_non_rag(frage, model_name="gemini"))

    print("\n=== RAG Gemini ===")
    print(rag_system.answer_rag(frage, model_name="gemini", top_k=2))
