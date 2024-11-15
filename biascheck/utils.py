import json
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import torch

def load_terms(terms):
    """
    Load terms from a file or list.
    Parameters:
        terms (str or list): Path to terms file or a list of terms.
    Returns:
        list: Loaded terms.
    """
    if isinstance(terms, str) and os.path.exists(terms):
        with open(terms, "r", encoding="utf-8") as file:
            return file.read().splitlines()
    elif isinstance(terms, list):
        return terms
    return []

def embed_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Embed texts using a sentence transformer model.
    Parameters:
        texts (list): List of texts to embed.
        model_name (str): Model name for embedding.
    Returns:
        np.array: Embedded vectors.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def build_faiss_index(embeddings):
    """
    Build a FAISS index for retrieval.
    Parameters:
        embeddings (np.array): Embedding vectors.
    Returns:
        faiss.IndexFlatL2: FAISS index.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index