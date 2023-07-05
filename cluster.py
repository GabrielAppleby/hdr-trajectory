from pathlib import Path
from typing import List, Dict

import click
import numpy as np
import umap
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import hdbscan

from config import UMAP_NN_PARAMS, PROJECTION_TEMPLATE, HDBSCAN_PARAMS, CLUSTER_TEMPLATE, CLUSTER_OUTPUT_FILENAME_TEMPLATE

RANDOM_SEED: int = 42
N_JOBS: int = -1

METRIC: str = 'euclidean'
CLUSTER: str = 'cluster'


def umap_reducer(metric, param):
    return umap.UMAP(n_neighbors=param, metric=metric, random_state=RANDOM_SEED)


def load_data(file_path: str) -> np.ndarray:
    data = np.load(file_path)

    return data['arr']


def get_pairwise_distance_matrix(arr: np.ndarray) -> np.ndarray:
    return pairwise_distances(arr, metric=METRIC)


def project_data(arr: np.ndarray) -> Dict[str, np.ndarray]:
    constant_dict = {i: i for i in range(arr.shape[0])}
    constant_relations = [constant_dict for i in range(len(UMAP_NN_PARAMS) - 1)]

    neighbors_mapper = umap.AlignedUMAP(
        n_neighbors=UMAP_NN_PARAMS,
        alignment_window_size=2,
        alignment_regularisation=1e-3,
        low_memory=True,
    ).fit(
        [arr for i in range(len(UMAP_NN_PARAMS))], relations=constant_relations
    )
    param_to_umap = {}
    for embedding_index, param in enumerate(UMAP_NN_PARAMS):
        param_to_umap[PROJECTION_TEMPLATE.format(param)] = neighbors_mapper.embeddings_[embedding_index]
    return param_to_umap


def cluster_single(mcs: int, ms: int, cse: float, arr: np.ndarray) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_epsilon=cse, cluster_selection_method='leaf').fit(arr)
    return clusterer.labels_


def cluster_data(arr: np.ndarray) -> Dict[str, np.ndarray]:
    clusters: List[np.ndarray] = list(Parallel(n_jobs=-1)(
        delayed(cluster_single)(mcs, ms, cse, arr) for mcs, ms, cse in tqdm(HDBSCAN_PARAMS)))
    param_to_cluster = {}
    for cluster_index, (mcs, ms, cse) in enumerate(HDBSCAN_PARAMS):
        param_to_cluster[CLUSTER_TEMPLATE.format(mcs, ms, '{:.2f}'.format(cse))] = clusters[cluster_index]
    return param_to_cluster


def cluster_impl(filename: str):
    arr = load_data(filename)
    # For now focus on just clustering products
    arr = arr[:, -1, :]
    param_to_cluster = cluster_data(arr)
    param_to_projection = project_data(arr)
    file_path = Path(filename)
    np.savez(Path(file_path.parent,
                  CLUSTER_OUTPUT_FILENAME_TEMPLATE.format(''.join(file_path.stem.split('-')[:-1]))),
             **param_to_projection,
             **param_to_cluster)


@click.command()
@click.option('-f',
              '--filename',
              required=True,
              help='The distance file to cluster.',
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
def cluster(filename: str):
    cluster_impl(filename)


if __name__ == '__main__':
    cluster()
