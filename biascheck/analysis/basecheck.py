from ..utils.embed_utils import Embedder
from ..utils.faiss_utils import FAISSRetriever
from ..utils.terms_loader import load_terms

class BaseCheck:
    def __init__(self, data, inputCols=[], terms=None, model_name=None):
        """
        Database bias analysis class.
        Parameters:
            data (Any): Database connection or raw data.
            inputCols (list): Columns or keys to analyze for bias.
            terms (str or list): Terms for bias detection.
            model_name (str): Transformer model for embedding.
        """
        self.data = data
        self.inputCols = inputCols
        self.terms = load_terms(terms)
        self.embedder = Embedder(model_name=model_name)

    def analyze(self):
        """
        Analyze the database for bias.
        Returns:
            dict: Summary metrics for the database.
        """
        term_embeddings = self.embedder.embed(self.terms)
        metrics = {}

        for col in self.inputCols:
            col_data = self.data[col]  # Assuming it's iterable
            embeddings = self.embedder.embed(col_data)

            term_frequency = {}
            for i, entry in enumerate(col_data):
                similarity = cosine_similarity([embeddings[i]], term_embeddings).mean()
                if similarity > 0.5:  # Threshold
                    for term in self.terms:
                        if term in entry:
                            term_frequency[term] = term_frequency.get(term, 0) + 1

            metrics[col] = {
                "bias_score": len(term_frequency) / max(len(col_data), 1),
                "term_frequency": term_frequency,
            }
        return metrics

    def generate_report(self):
        """
        Generate a human-readable report of the analysis.
        Returns:
            str: Detailed bias report.
        """
        report = "Bias Analysis Report:\n"
        analysis = self.analyze()

        for col, data in analysis.items():
            report += f"\nColumn: {col}\n"
            report += f"Bias Score: {data['bias_score']:.2f}\n"
            report += "Term Frequencies:\n"
            for term, freq in data["term_frequency"].items():
                report += f"  {term}: {freq}\n"

        return report