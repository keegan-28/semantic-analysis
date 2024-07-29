from typing import List, Any, Tuple, Union
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def vectorize_docs(
    docs: List[str],
    stop_words: Union[List[str], str] = "english",
    ngram_range: Tuple[int, int] = (1, 1),
):
    """
    Inputs:
        stop_words: list of stop words to use for CountVectoriser.
            Default value is "english"
        ngram_range: Length, in words, of the extracted keywords/keyphrases.
            Default: (1,1)

    Output:
        words: ndarray of vectorized words

    """

    counter = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words,
    )

    token_count = counter.fit(docs)
    words = token_count.get_feature_names_out()
    count_matrix = token_count.fit_transform(docs)

    return words.tolist(), count_matrix


def embed_words(words: List[str], emb_model: SentenceTransformer):

    word_embeddings = emb_model.encode(words)

    return word_embeddings


def embed_docs(docs: List[str], emb_model: SentenceTransformer):

    doc_embeddings = emb_model.encode(docs)

    return doc_embeddings


def extract_keywords(docs, words, count_matrix, doc_emb, word_emb, num_topics: int = 5):
    all_keywords = []

    for index, _ in enumerate(docs):

        candidate_indices = count_matrix[index].nonzero()[1]

        candidates = [words[index] for index in candidate_indices]
        candidate_embeddings = word_emb[candidate_indices]
        doc_embedding: np.ndarray = doc_emb[index].reshape(1, -1)

        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords = [
            (candidates[index], round(float(distances[0][index]), 4))
            for index in distances.argsort()[0][-num_topics:]
        ][::-1]

        all_keywords.append(keywords)

    keywords = [item for sublist in all_keywords for item in sublist]

    deduped_data = {}
    for item in keywords:
        if item[0] not in deduped_data or item[1] > deduped_data[item[0]]:
            deduped_data[item[0]] = item[1]

    # Convert the dictionary back to a list of tuples
    deduped_data = list(deduped_data.items())

    # Sorting the list by the float value in each tuple in descending order
    sorted_keywords = sorted(deduped_data, key=lambda x: x[1], reverse=True)

    sorted_keywords = sorted_keywords[0:num_topics]

    keywords = {"keyword": [], "cosine_sim": []}
    for item in sorted_keywords:
        keywords["keyword"].append(item[0])
        keywords["cosine_sim"].append(item[1])

    return keywords
