import numpy as np
from scipy.spatial.distance import cosine

"""
move to text_processing.py
"""


def calculate_centroid(embeddings: np.ndarray) -> np.ndarray:
    return embeddings.mean(axis=0)


def calculate_cosine_similarity(
    prompt_emb: np.ndarray, data_emb: np.ndarray
) -> np.ndarray:
    dist: np.ndarray = np.zeros(data_emb.shape[0])
    for idx, emb in enumerate(data_emb):
        dist[(idx)] = 1 - cosine(prompt_emb, emb)

    return dist
