from .utils import load_terms, embed_texts, build_faiss_index
import torch

class ModuCheck:
    def __init__(self, data, model=None, document=None, terms=None, retriever=None, israg=False):
        """
        Parameters:
            data (list): Input queries.
            model (transformers.PreTrainedModel): Pre-loaded LLM model.
            document (str): Path to document for retrieval.
            terms (str or list): Path to terms file or list of terms.
            retriever (faiss.IndexFlatL2): Pre-built FAISS retriever.
            israg (bool): Whether to use RAG pipeline.
        """
        self.data = data
        self.model = model
        self.document = document
        self.terms = load_terms(terms)
        self.retriever = retriever
        self.israg = israg

        if self.israg and document:
            self.pipeline = self._build_rag_pipeline()

    def _build_rag_pipeline(self):
        """
        Build a simple RAG pipeline.
        Returns:
            callable: Retrieval-augmented generation pipeline.
        """
        with open(self.document, "r", encoding="utf-8") as file:
            texts = file.readlines()
        
        embeddings = embed_texts(texts)
        retriever = build_faiss_index(embeddings)

        def pipeline(query):
            query_vec = embed_texts([query])
            _, indices = retriever.search(query_vec, k=5)
            retrieved_texts = [texts[i] for i in indices[0]]
            return " ".join(retrieved_texts)

        return pipeline

    def analyze(self):
        """
        Analyze model responses for bias.
        Returns:
            dict: Bias scores and flagged responses.
        """
        if self.israg:
            responses = [self.pipeline(query) for query in self.data]
        else:
            responses = [self.model.generate(torch.tensor([query])) for query in self.data]

        flagged_responses = [
            response for response in responses if any(term in response for term in self.terms)
        ]
        bias_score = len(flagged_responses) / max(len(responses), 1)
        return {
            "bias_score": bias_score,
            "flagged_responses": flagged_responses,
        }