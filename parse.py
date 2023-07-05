import json
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from config import PARSE_OUTPUT_FILENAME_TEMPLATE


def get_atoms(data: List[List[List[str]]]) -> List[str]:
    coords = data[0][0][1:]
    return list([c.split()[0] for c in coords])


def load_sim(file_name: str) -> Tuple[List[str], int, List[List[List[str]]]]:
    with open(file_name) as f:
        data = json.load(f)
        return get_atoms(data['coord']), data['nstep'], data['coord']


def parse_coord(coord: str) -> List[float]:
    return [float(x) for x in coord.split()[1:]]


def parse_step(step: List[str]) -> List[List[float]]:
    return [parse_coord(coord) for coord in step[1:]]


def parse_traj(n_steps: int, traj: List[List[str]]) -> List[List[List[float]]]:
    results = [parse_step(step) for step in traj]
    # Extend last result to final time step.
    if len(results) < n_steps:
        results.extend([results[-1]] * (n_steps - len(results)))
    return results


def parse_sim(n_steps: int, sim: List[List[List[str]]]) -> np.ndarray:
    return np.array(Parallel(n_jobs=-1)(
        delayed(parse_traj)(n_steps, traj) for traj in tqdm(sim, desc='Trajectories')), dtype=np.float32)


def parse_impl(filename: str):
    atoms, n_steps, sim = load_sim(filename)
    arr = parse_sim(n_steps, sim)
    file_path = Path(filename)
    np.savez(Path(file_path.parent, PARSE_OUTPUT_FILENAME_TEMPLATE.format(file_path.stem)),
             arr=arr,
             atom_names=atoms)


@click.command()
@click.option('-f',
              '--filename',
              required=True,
              help='The sim file to parse.',
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
def parse(filename: str):
    parse_impl(filename)


if __name__ == '__main__':
    parse()
