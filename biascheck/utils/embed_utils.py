from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=16):
        """
        Embedding utility for text data.
        Parameters:
            model_name (str): Transformer model name.
            batch_size (int): Batch size for efficient embedding.
        """
        if model_name is None:
            raise ValueError("`model_name` cannot be None. Please provide a valid Hugging Face model name.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.batch_size = batch_size

    def embed(self, texts):
        """
        Generate embeddings for a list of texts.
        Parameters:
            texts (list): List of text strings.
        Returns:
            np.array: Embedding matrix.
        """
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.numpy())
        return np.vstack(all_embeddings)