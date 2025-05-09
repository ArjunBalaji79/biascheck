from biascheck.analysis.moducheck import ModuCheck
from biascheck.analysis.report_generator import ReportGenerator
from langchain.llms import Ollama
import os
import json


model = Ollama(model="mistral")
topics = ["Gender equality in leadership", "Cultural diversity"]

analyzer = ModuCheck(model=model, terms=["bias", "stereotype"])
result = analyzer.analyze(topics=topics, num_responses=3, word_limit=30)


report_generator = ReportGenerator(result)

os.makedirs("reports", exist_ok=True)

text_report_path = report_generator.generate_text_report(report_path="reports/bias_analysis_report.txt")

# For JSON, convert DataFrame to dict first
df_dict = result.to_dict(orient='records')
# Save JSON manually
with open("reports/bias_analysis_report.json", 'w') as f:
    json.dump(df_dict, f, indent=4)