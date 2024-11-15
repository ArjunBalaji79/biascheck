import os

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