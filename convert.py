from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np

from config import CONVERT_OUTPUT_FILENAME_TEMPLATE


def get_atom_combos(atom_names: List[str]):
    atoms = defaultdict(lambda: 0)
    for name in atom_names:
        atoms[name] += 1
    return ['{}-{}'.format(a, b) for a, b in
            list(combinations([key + str(i) for key, val in atoms.items() for i in range(val)], 2))]


def load_data(file_name: str) -> Tuple[List[str], np.ndarray]:
    data = np.load(file_name)
    return get_atom_combos(data['atom_names']), data['arr']


def pairwise_atom_dists(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    r = np.sum(arr * arr, -1)
    r = np.expand_dims(r, -1)
    distance = np.sqrt(r - 2 * np.matmul(arr, np.transpose(arr, axes=(0, 1, 3, 2))) + np.transpose(r, axes=(0, 1, 3, 2)))
    triu_indices = np.triu_indices_from(distance[0][0], k=1)

    return distance[(slice(None), slice(None), *triu_indices)].astype(np.float32)


def convert_impl(filename: str):
    atoms, arr = load_data(filename)
    arr = pairwise_atom_dists(arr)
    file_path = Path(filename)
    np.savez(Path(file_path.parent,
                  CONVERT_OUTPUT_FILENAME_TEMPLATE.format('-'.join(file_path.stem.split('-')[:-1]))),
             arr=arr,
             atom_names=atoms)


@click.command()
@click.option('-f',
              '--filename',
              required=True,
              help='The parsed sim file to convert to a distance representation.',
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
def convert(filename: str):
    convert_impl(filename)


if __name__ == '__main__':
    convert()
