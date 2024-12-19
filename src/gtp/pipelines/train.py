import click

from gtp.configs.loaders import load_configs
from gtp.configs.project import GenotypeToPhenotypeConfigs


def train():
    pass


@click.command()
@click.option(
    "--configs",
    default=None,
    help="Path to YAML config file to be used in preprocessing.",
)
@click.option(
    "--method",
    default="both",
    type=click.Choice(["phenotype", "genotype", "both"]),
    prompt="Select preprocessing method to run:",
    help="Which preprocessing method to run [phenotype, genotype, both]",
)
@click.option(
    "--verbose/--no-verbose", default=False, help="Whether or not to see logging"
)
def main(configs, verbose):
    cfgs: GenotypeToPhenotypeConfigs = load_configs(configs)


if __name__ == "__main__":
    main()
