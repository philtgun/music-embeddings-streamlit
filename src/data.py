import random
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP


class Loader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load_embeddings(
            self,
            dataset: str,
            architecture: str,
            layer: str,
            total: int,
            shuffle=False,
    ):
        embeddings_dir = self.data_dir / f'{dataset}-{architecture}-{layer}-pca'
        embedding_files = sorted(embeddings_dir.rglob('*.npy'))
        if len(embedding_files) == 0:
            raise RuntimeError('No data found in the provided directory')

        if shuffle:
            random.shuffle(embedding_files)
        embeddings = [np.load(embedding_file) for embedding_file in embedding_files[:total]]
        embeddings_stacked = np.vstack(embeddings)

        return embeddings_stacked


def project(data, projection, input_dims=10):
    if projection == 'PCA':
        return data

    if projection == 't-SNE':
        tsne = TSNE(n_components=2, random_state=0)
        return tsne.fit_transform(data[:, :input_dims])

    if projection == 'UMAP':
        umap = UMAP(n_components=2, init='random', random_state=0)
        return umap.fit_transform(data[:, :input_dims])

    raise ValueError(f'Invalid projection: {projection}')
