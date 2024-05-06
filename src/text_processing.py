from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import dataclasses
import umap.umap_ as umap
import pickle
import hdbscan
import logging


class InvalidPickleData(Exception):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(
            f"Pickle file: {self.filename} does not contain valid Embeddings object."
        )


class EmbeddingsNotGenerated(Exception):
    def __init__(self) -> None:
        super().__init__("Generate embeddings first")


@dataclasses.dataclass(frozen=True)
class Embeddings:
    docs: List[str]
    emb_source: np.ndarray
    emb_low: np.ndarray
    emb_2d: np.ndarray
    model_name: str

    def __len__(self) -> int:
        """
        Return the count of docs.
        """
        return len(self.docs)


class TextProcessor:
    def __init__(self, embedding_model_path: str, docs: List[str], device: str) -> None:
        self.model_name = embedding_model_path.rsplit("/", 1)[-1]
        self.emb_model: SentenceTransformer = SentenceTransformer(
            embedding_model_path, device=device
        )
        self.docs = docs

    def generate_emb(
        self, emb_dir: Optional[str], low_dim: int = 10, random_state: int = 42
    ) -> Embeddings:
        embeddings = self.emb_model.encode(self.docs, convert_to_numpy=True)

        umap_low_dim = umap.UMAP(n_components=low_dim, random_state=random_state)
        umap_2d = umap.UMAP(n_components=2, random_state=random_state)

        emb_low = umap_low_dim.fit(embeddings)
        emb_2d = umap_2d.fit(embeddings)

        self.result = Embeddings(
            docs=self.docs,
            emb_source=embeddings,
            emb_low=emb_low,
            emb_2d=emb_2d,
            model_name=self.model_name,
        )

        try:
            self._save_embeddings(emb_dir)
            logging.info(f"Embeddings saved at location: {emb_dir}")
        except Exception:
            logging.info("No embeddings directory provided. Data not saved.")

        return self.result

    def _save_embeddings(self, embeddings_dir: str) -> None:

        with open(embeddings_dir + "docs.pkl", "wb") as f:
            pickle.dump(self.result.docs, f)

        np.save(file=embeddings_dir + "emb_source.npy", arr=self.result.emb_source)
        np.save(file=embeddings_dir + "emb_low.npy", arr=self.result.emb_low)
        np.save(file=embeddings_dir + "emb_2d.npy", arr=self.result.emb_2d)

    def load_data(self, data_path: str) -> Embeddings:
        raise NotImplementedError


class ClusterAnalytics(TextProcessor):
    def __init__(self, embedding_model_path: str, docs: List[str], device: str) -> None:
        super().__init__(embedding_model_path, docs, device)

    def cluster(self, *args) -> np.ndarray:
        clusterer = hdbscan.HDBSCAN(*args)
        clusterer.fit(self.result.emb_low)
        self.labels: np.ndarray = clusterer.labels_
        return self.labels
    
    def cluster_topics(self) -> List[List[str]]:
        raise NotImplementedError


class ClusterLabeller(ClusterAnalytics):
    def __init__(self, embedding_model_path: str, docs: List[str], device: str) -> None:
        super().__init__(embedding_model_path, docs, device)