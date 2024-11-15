import faiss

class FAISSRetriever:
    def __init__(self, embedding_dim=384):
        """
        FAISS-based retriever for efficient similarity search.
        Parameters:
            embedding_dim (int): Dimensionality of the embeddings.
        """
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(self, embeddings):
        """
        Add embeddings to the FAISS index.
        Parameters:
            embeddings (np.array): Array of embeddings.
        """
        self.index.add(embeddings)

    def search(self, query_embedding, k=5):
        """
        Perform a search on the FAISS index.
        Parameters:
            query_embedding (np.array): Query vector.
            k (int): Number of results to retrieve.
        Returns:
            list: Indices of the top-k results.
        """
        distances, indices = self.index.search(query_embedding, k)
        return indices