import click
from biascheck import DocuCheck, ModuCheck, SetCheck, BaseCheck

@click.group()
def cli():
    """BiasCheck CLI - Analyze Bias in Documents, Models, Datasets, and Databases."""
    pass

@cli.command()
@click.option("--type", type=click.Choice(["docu", "modu", "set", "base"]), required=True, help="Type of analysis.")
@click.option("--input", type=click.Path(exists=True), required=True, help="Input file or dataset.")
@click.option("--terms", type=click.Path(exists=True), help="Path to terms file.")
def analyze(type, input, terms):
    """Analyze bias based on the specified type."""
    if type == "docu":
        analyzer = DocuCheck(data=open(input).read(), terms=terms)
    elif type == "modu":
        # Add LLM model integration here
        pass
    elif type == "set":
        analyzer = SetCheck(data=pd.read_csv(input), terms=terms)
    elif type == "base":
        # Add database integration here
        pass
    result = analyzer.analyze()
    print(result)

if __name__ == "__main__":
    cli()