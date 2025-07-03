from fasttext.FastText import _FastText
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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


def translation_models(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=model_path
    )
    return tokenizer, model


def translate(
    tokenizer, model, input_text: List[str], labels: List[str], confs: List[float]
) -> List[str]:
    """ """
    tgt_lang_id = tokenizer.lang_code_to_id["eng_Latn"]
    model_inputs = tokenizer()

    raise NotImplementedError()
