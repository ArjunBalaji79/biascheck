import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from PyPDF2 import PdfReader
from ..utils.embed_utils import Embedder


class ModuCheck:
    def __init__(
        self,
        model=None,
        document=None,
        terms=None,
        bias_threshold=0.4,
        model_type="huggingface",  # Options: "huggingface", "ollama", "local"
        verbose=False,
    ):
        """
        Initialize ModuCheck to analyze model outputs for bias.

        Parameters:
            model: The language model (HuggingFace, Ollama, or local).
            document: Optional document for creating a RAG pipeline.
            terms: List of bias terms.
            bias_threshold: Threshold for cosine similarity.
            model_type: Type of model ("huggingface", "ollama", "local").
            verbose: Whether to display intermediate results.
        """
        self.model = model
        self.document = document
        self.terms = terms or []
        self.bias_threshold = bias_threshold
        self.model_type = model_type
        self.verbose = verbose

        # Set up RAG pipeline if a document is provided
        if document:
            self._setup_rag_pipeline(document)

    def _extract_text_from_pdf(self, file_path):
        """
        Extract text from a PDF using PyPDF2.

        Parameters:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def _setup_rag_pipeline(self, document):
        """
        Create a RAG pipeline with the provided document.
        """
        if document.lower().endswith(".pdf"):
            text = self._extract_text_from_pdf(document)
        else:
            with open(document, "r", encoding="utf-8") as f:
                text = f.read()

        # Convert text to LangChain Document objects
        docs = [Document(page_content=text, metadata={"source": document})]
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.model,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )

    def _generate_responses(self, topics, num_responses=50):
        """
        Generate multiple outputs for each topic.

        Parameters:
            topics: List of topics.
            num_responses: Number of outputs per topic.

        Returns:
            List of generated outputs.
        """
        responses = []
        for topic in topics:
            for _ in range(num_responses):
                if self.document:
                    # Extract only the "result" from the RAG pipeline output
                    response = self.rag_chain({"query": topic})["result"]
                else:
                    # Generate output depending on model type
                    if self.model_type == "ollama":
                        result = self.model.generate([topic])  # Ollama LLMResult
                        response = result.generations[0][0].text  # Extract text
                    elif self.model_type == "huggingface":
                        response = self.model(topic)  # HuggingFace models
                    else:
                        response = self.model.generate([topic])[0]  # Local GGUF models
                responses.append({"topic": topic, "response": response})
        return responses

    def analyze(self, topics, num_responses=50):
        """
        Analyze model outputs for bias.

        Parameters:
            topics: List of topics to query.
            num_responses: Number of outputs per topic.

        Returns:
            DataFrame of flagged outputs and their biases.
        """
        responses = self._generate_responses(topics, num_responses)
        flagged_records = []

        for response in responses:
            text = response["response"]
            sentiment = TextBlob(text).sentiment.polarity

            # Check if text contains any of the terms
            contains_bias = any(term.lower() in text.lower() for term in self.terms)

            flagged_records.append({
                "topic": response["topic"],
                "response": text,
                "sentiment": sentiment,
                "contains_bias": contains_bias,
            })

        return pd.DataFrame(flagged_records)