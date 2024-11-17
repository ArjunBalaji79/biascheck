import pandas as pd
from textblob import TextBlob

class ModuCheck:
    def __init__(self, model, terms=None, verbose=False):
        """
        Initialize ModuCheck to analyze model outputs for bias.

        Parameters:
            model: The language model to evaluate.
            terms: List of bias terms.
            verbose: Whether to display intermediate results.
        """
        self.model = model
        self.terms = terms or []
        self.verbose = verbose

    def _generate_responses(self, topics, num_responses=10, word_limit=None):
        """
        Generate outputs for the given topics.

        Parameters:
            topics: List of topics to analyze.
            num_responses: Number of outputs per topic.
            word_limit: Maximum number of words for each generated response.

        Returns:
            List of generated outputs.
        """
        responses = []
        for topic in topics:
            for _ in range(num_responses):
                # Add the word limit to the prompt if specified
                prompt = topic
                if word_limit:
                    prompt += f" (Limit your response to {word_limit} words.)"
                
                # Generate response and extract text
                llm_result = self.model.generate([prompt])
                response_text = llm_result.generations[0][0].text  # Extracting the text
                responses.append({"topic": topic, "response": response_text})
        return responses

    def analyze(self, topics, num_responses=10, word_limit=None):
        """
        Analyze model outputs for bias.

        Parameters:
            topics: List of topics to analyze.
            num_responses: Number of outputs per topic.
            word_limit: Maximum number of words for each generated response.

        Returns:
            DataFrame with flagged outputs and bias analysis.
        """
        responses = self._generate_responses(topics, num_responses, word_limit)
        flagged_records = []

        for response in responses:
            text = response["response"]
            sentiment = TextBlob(text).sentiment.polarity
            contains_bias = any(term.lower() in text.lower() for term in self.terms)

            flagged_records.append({
                "topic": response["topic"],
                "response": text,
                "sentiment": sentiment,
                "contains_bias": contains_bias,
            })

        return pd.DataFrame(flagged_records)