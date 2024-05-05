from fasttext.FastText import _FastText
from typing import List
import typing


def language_id(
    fmodel: _FastText, input_text: List[str]
) -> tuple[List[str], List[float]]:
    """ """
    predictions = fmodel.predict(input_text, k=1)
    labels: List[str] = [label[0].replace("__label__", "") for label in predictions[0]]
    confs: List[float] = [conf[0] for conf in predictions[1]]
    return labels, confs


def translate(
    input_text: List[str], labels: List[str], confs: List[float]
) -> List[str]:
    """ """
    raise NotImplementedError()
