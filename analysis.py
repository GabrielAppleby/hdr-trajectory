from pathlib import Path

import click

from cluster import cluster_impl
from config import PARSE_OUTPUT_FILENAME_TEMPLATE, CONVERT_OUTPUT_FILENAME_TEMPLATE
from convert import convert_impl
from parse import parse_impl


@click.command()
@click.option('-f',
              '--filename',
              required=True,
              help='The sim file to parse, convert, and cluster.',
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
def analyze(filename: str):
    parse_impl(filename)
    file_path = Path(filename)
    convert_impl(str(Path(file_path.parent, PARSE_OUTPUT_FILENAME_TEMPLATE.format(file_path.stem))))
    cluster_impl(str(Path(file_path.parent, CONVERT_OUTPUT_FILENAME_TEMPLATE.format(file_path.stem))))


if __name__ == '__main__':
    analyze()
