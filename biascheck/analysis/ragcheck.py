import pandas as pd
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from PyPDF2 import PdfReader


class RAGCheck:
    def __init__(self, model, document, terms=None, verbose=False):
        """
        Initialize RAGCheck to analyze bias in RAG pipelines.

        Parameters:
            model: The language model to use in the pipeline.
            document: Path to the document to create a RAG pipeline.
            terms: List of bias terms.
            verbose: Whether to display intermediate results.
        """
        self.model = model
        self.document = document
        self.terms = terms or []
        self.verbose = verbose
        self._setup_rag_pipeline()

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

    def _setup_rag_pipeline(self):
        """
        Create a RAG pipeline using the provided document.
        """
        if self.document.lower().endswith(".pdf"):
            text = self._extract_text_from_pdf(self.document)
        else:
            with open(self.document, "r", encoding="utf-8") as f:
                text = f.read()

        docs = [Document(page_content=text, metadata={"source": self.document})]
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.model,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )

    def analyze(self, topics, num_responses=10, word_limit=None):
        """
        Analyze bias in RAG pipeline outputs.

        Parameters:
            topics: List of topics to query.
            num_responses: Number of outputs per topic.
            word_limit: Maximum number of words for each generated response.

        Returns:
            DataFrame with flagged outputs and their bias analysis.
        """
        responses = []
        for topic in topics:
            for _ in range(num_responses):
                prompt = topic
                if word_limit:
                    prompt += f" (Limit your response to {word_limit} words.)"

                result = self.rag_chain({"query": prompt})
                response = result["result"]
                source_docs = result.get("source_documents", [])
                contains_bias = any(
                    term.lower() in response.lower() for term in self.terms
                )
                responses.append({
                    "topic": topic,
                    "response": response,
                    "source_documents": [doc.page_content for doc in source_docs],
                    "contains_bias": contains_bias,
                })

        return pd.DataFrame(responses)