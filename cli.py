import click
import pandas as pd
from biascheck.analysis.docucheck import DocuCheck
from biascheck.analysis.moducheck import ModuCheck
from biascheck.analysis.setcheck import SetCheck
from biascheck.analysis.basecheck import BaseCheck
from biascheck.analysis.report_generator import ReportGenerator

@click.group()
def cli():
    """BiasCheck CLI - Analyze Bias in Documents, Models, Datasets, and Databases."""
    pass

@cli.command()
@click.option("--type", type=click.Choice(["docu", "modu", "set", "base"]), required=True, help="Type of analysis.")
@click.option("--input", type=click.Path(exists=True), required=True, help="Input file or dataset.")
@click.option("--terms", type=click.Path(exists=True), help="Path to terms file.")
@click.option("--columns", default=None, help="Columns to analyze (comma-separated, for datasets/databases).")
@click.option("--output", type=click.Path(), help="Path to save analysis results (CSV format).")
def analyze(type, input, terms, columns, output):
    """
    Analyze bias in the specified input type.
    """
    if type == "docu":
        analyzer = DocuCheck(data=open(input).read(), terms=terms)
    elif type == "modu":
        # Replace with actual model integration
        model = None
        analyzer = ModuCheck(data=open(input).readlines(), model=model, terms=terms)
    elif type == "set":
        columns = columns.split(",") if columns else []
        analyzer = SetCheck(data=pd.read_csv(input), inputCols=columns, terms=terms)
    elif type == "base":
        # Replace with database integration
        database = {}
        columns = columns.split(",") if columns else []
        analyzer = BaseCheck(data=database, inputCols=columns, terms=terms)

    result = analyzer.analyze()
    
    if output:
        result.to_csv(output, index=False)
        click.echo(f"Analysis results saved to {output}")
    else:
        print(result)

@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("--format", type=click.Choice(["text", "json", "both"]), default="both", help="Report format(s) to generate.")
@click.option("--output-dir", type=click.Path(), default="reports", help="Directory to save reports.")
def generate_report(input, format, output_dir):
    """
    Generate reports from analysis results.
    
    INPUT should be a CSV file containing analysis results.
    """
    try:
        # Read the analysis results
        df = pd.read_csv(input)
        
        # Initialize report generator
        report_gen = ReportGenerator(df)
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate reports based on format
        if format in ["text", "both"]:
            text_path = os.path.join(output_dir, "bias_analysis_report.txt")
            report_gen.generate_text_report(text_path)
            click.echo(f"Text report saved to {text_path}")
            
        if format in ["json", "both"]:
            json_path = os.path.join(output_dir, "bias_analysis_report.json")
            report_gen.generate_json_report(json_path)
            click.echo(f"JSON report saved to {json_path}")
            
    except Exception as e:
        click.echo(f"Error generating report: {str(e)}", err=True)
        raise click.Abort()

if __name__ == "__main__":
    cli()