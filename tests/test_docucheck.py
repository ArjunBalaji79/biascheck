import unittest
from biascheck import DocuCheck

class TestDocuCheck(unittest.TestCase):
    def test_analyze(self):
        data = "This is a test document. It contains biased terms."
        terms = ["biased"]
        analyzer = DocuCheck(data=data, terms=terms)
        result = analyzer.analyze()
        self.assertTrue(result["bias_score"] > 0)

if __name__ == "__main__":
    unittest.main()