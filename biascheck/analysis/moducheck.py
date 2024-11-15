from ..utils.embed_utils import Embedder
from ..utils.faiss_utils import FAISSRetriever
from ..utils.terms_loader import load_terms

class ModuCheck:
    def __init__(self, data, model=None, document=None, terms=None, israg=False):
        """
        Model bias analysis class.
        Parameters:
            data (list): Queries or input data.
            model (transformers.PreTrainedModel): Model for generation.
            document (str): Path to document for retrieval.
            terms (str or list): Terms for bias detection.
            israg (bool): Whether to use retrieval-augmented generation.
        """
        self.data = data
        self.model = model
        self.terms = load_terms(terms)
        self.israg = israg
        self.embedder = Embedder()
        self.retriever = None

        if israg and document:
            self._initialize_rag(document)

    def _initialize_rag(self, document):
        """
        Initialize RAG pipeline using FAISS retriever.
        """
        with open(document, "r", encoding="utf-8") as file:
            texts = file.readlines()

        embeddings = self.embedder.embed(texts)
        self.retriever = FAISSRetriever()
        self.retriever.add_embeddings(embeddings)

    def analyze(self):
        """
        Analyze model outputs for bias.
        Returns:
            dict: Bias scores and flagged responses.
        """
        responses = []
        for query in self.data:
            if self.israg:
                query_embedding = self.embedder.embed([query])
                indices = self.retriever.search(query_embedding, k=5)
                retrieved_context = " ".join([self.data[i] for i in indices[0]])
                query = query + retrieved_context

            response = self.model.generate(torch.tensor([query]))
            responses.append(response)

        flagged_responses = [resp for resp in responses if any(term in resp for term in self.terms)]
        bias_score = len(flagged_responses) / max(len(responses), 1)
        return {"bias_score": bias_score, "flagged_responses": flagged_responses}