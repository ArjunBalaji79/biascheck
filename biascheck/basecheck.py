from .utils import load_terms

class BaseCheck:
    def __init__(self, data, inputCols=[], terms=None):
        """
        Parameters:
            data (Any): Graph or vector database connection.
            inputCols (list): Columns or keys to analyze for bias.
            terms (str or list): Path to terms file or list of terms.
        """
        self.data = data
        self.inputCols = inputCols
        self.terms = load_terms(terms)

    def analyze(self):
        """
        Analyze the database for bias.
        Returns:
            dict: Summary metrics.
        """
        term_frequency = {}
        for col in self.inputCols:
            for entry in self.data[col]:
                for term in self.terms:
                    term_frequency[term] = term_frequency.get(term, 0) + entry.count(term)

        return {"term_frequency": term_frequency}