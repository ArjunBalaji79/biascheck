import click
from biascheck.analysis.docucheck import DocuCheck
from biascheck.analysis.moducheck import ModuCheck
from biascheck.analysis.setcheck import SetCheck
from biascheck.analysis.basecheck import BaseCheck

@click.group()
def cli():
    """BiasCheck CLI - Analyze Bias in Documents, Models, Datasets, and Databases."""
    pass

@cli.command()
@click.option("--type", type=click.Choice(["docu", "modu", "set", "base"]), required=True, help="Type of analysis.")
@click.option("--input", type=click.Path(exists=True), required=True, help="Input file or dataset.")
@click.option("--terms", type=click.Path(exists=True), help="Path to terms file.")
@click.option("--columns", default=None, help="Columns to analyze (comma-separated, for datasets/databases).")
def analyze(type, input, terms, columns):
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
    print(result)

if __name__ == "__main__":
    cli()