from typing import List, Optional, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
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

    def get_stats(self) -> None:
        """
        Prints statistics about:
        - The variance retained during UMAP for intermediate and 2d dim reduction
        - Doc count
        - Source embedding dimensions
        """
        return NotImplementedError


@dataclass(frozen=True)
class ParamHDBSCAN:
    min_samples: List[int] 
    min_cluster_size: List[int]
    cluster_selection_epsilon: List[float]
    cluster_selection_method: List[str]
    metric: List[str]


class Clusterer:
    def __init__(
        self, embedding_model_path: str, docs: List[str], device: str, seed: int = 42
    ) -> None:
        self.model_name = embedding_model_path.rsplit("/", 1)[-1]
        self.emb_model: SentenceTransformer = SentenceTransformer(
            embedding_model_path, device=device
        )
        self.docs = docs
        self.seed = seed

    def generate_emb(self, emb_dir: Optional[str], low_dim: int = 10) -> Embeddings:
        embeddings = self.emb_model.encode(self.docs, convert_to_numpy=True)

        umap_low_dim = umap.UMAP(n_components=low_dim, random_state=self.seed)
        umap_2d = umap.UMAP(n_components=2, random_state=self.seed)

        emb_low = umap_low_dim.fit_transform(embeddings)
        emb_2d = umap_2d.fit_transform(embeddings)

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

        with open(embeddings_dir + "/docs.pkl", "wb") as f:
            pickle.dump(self.result.docs, f)

        np.save(file=embeddings_dir + "/emb_source.npy", arr=self.result.emb_source)
        np.save(file=embeddings_dir + "/emb_low.npy", arr=self.result.emb_low)
        np.save(file=embeddings_dir + "/emb_2d.npy", arr=self.result.emb_2d)

    def load_embeddings(self, embeddings_dir: str) -> Embeddings:

        embeddings: np.ndarray = np.load(file=embeddings_dir + "/emb_source.npy")
        emb_low: np.ndarray = np.load(file=embeddings_dir + "/emb_low.npy")
        emb_2d: np.ndarray = np.load(file=embeddings_dir + "/emb_2d.npy")

        with open(embeddings_dir + "/docs.pkl", "rb") as f:
            docs: List[str] = pickle.load(f)

        self.result = Embeddings(
            docs=docs,
            emb_source=embeddings,
            emb_low=emb_low,
            emb_2d=emb_2d,
            model_name=self.model_name,
        )

        return self.result

    def cluster(self, return_params: bool = True, *args, **kwargs) -> Tuple[np.ndarray, np.float64, Dict[str, Any]]:
        clusterer = hdbscan.HDBSCAN(gen_min_span_tree=True, *args, **kwargs)
        clusterer.fit(self.result.emb_low)
        self.labels: np.ndarray = clusterer.labels_
        score = clusterer.relative_validity_
        return self.labels, score, clusterer.get_params()

    def tune_HDBSCAN(self, params: ParamHDBSCAN) -> Tuple[np.float64, Dict[str, List[Any]]]:
        best_score = 0
        scores = []
        coverage = []

        for min_cluster_size in params.min_cluster_size:
            for min_samples in params.min_samples:
                for cluster_selection_epsilon in params.cluster_selection_epsilon:
                    for cluster_selection_method in params.cluster_selection_method:
                        for metric in params.metric:
                            # for each combination of parameters of hdbscan
                            hdb = hdbscan.HDBSCAN(
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                cluster_selection_method=cluster_selection_method,
                                metric=metric,
                                gen_min_span_tree=True,
                                cluster_selection_epsilon=cluster_selection_epsilon,
                            ).fit(self.result.emb_low)
                            # DBCV score
                            score: np.float64 = hdb.relative_validity_
                            scores.append(score)
                            coverage.append((hdb.labels_>=0).sum()/self.result.emb_low.shape[0])
                            # if we got a better DBCV, store it and the parameters
                            if score > best_score:
                                best_score = score
                                best_parameters = {
                                    "min_cluster_size": min_cluster_size,
                                    "min_samples": min_samples,
                                    "cluster_selection_method": cluster_selection_method,
                                    "metric": metric,
                                }

        return best_score, best_parameters, scores, coverage
