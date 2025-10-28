# utils_lid.py


import re

import fasttext
import numpy as np
from huggingface_hub import hf_hub_download

# Global models initialized to None
GLOTLID_MODEL = None
CUSTOM_MODEL = None


def get_model() -> fasttext.FastText._FastText:
    """
    Lazily downloads and loads the main GlotLID model.
    """
    global GLOTLID_MODEL
    if GLOTLID_MODEL is None:
        glotlid_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        GLOTLID_MODEL = fasttext.load_model(str(glotlid_path))
    return GLOTLID_MODEL


def get_custom_model(languages: list[str], mode: str = "before") -> "CustomLID":
    """
    Lazily initializes or re-initializes the CustomLID model wrapper
    with a specific set of languages.
    """
    global CUSTOM_MODEL
    # Ensure labels are prefixed
    languages_pref = [f"__label__{lx}" if not lx.startswith("__label__") else lx for lx in languages]

    if CUSTOM_MODEL is None or set(CUSTOM_MODEL.labels) != set(languages_pref):
        CUSTOM_MODEL = CustomLID(languages=languages_pref, mode=mode)
    return CUSTOM_MODEL


def langid(text: str) -> str:
    """
    Predicts the language of a text string using the full GlotLID model.
    """
    # Normalize text
    pat = re.compile(r"\s+")
    text_c = pat.sub(" ", text)

    # Predict language
    model = get_model()
    (label,), score = model.predict(text_c)

    return label.removeprefix("__label__") if "__label__" in label else "UNK"


def langid_custom(text: str, languages: list[str], mode: str = "before") -> tuple[str, float]:
    """
    Predicts the language of a text, constrained to a specific list of languages.
    """
    # Normalize text
    pat = re.compile(r"\s+")
    text_c = pat.sub(" ", text)

    # Predict language
    model = get_custom_model(languages=languages, mode=mode)
    (label,), score = model.predict(text_c)

    lang_code = label.removeprefix("__label__") if "__label__" in label else "UNK"
    return lang_code, score[0]


class CustomLID:
    """
    A wrapper around the fasttext model to limit predictions to a subset of languages
    by manipulating the output matrix before or after the softmax calculation.
    Source: https://github.com/cisnlp/GlotLID
    """

    def __init__(self, languages: list[str], mode: str = "before"):
        self.model = get_model()
        self.output_matrix = self.model.get_output_matrix()
        all_labels = self.model.get_labels()

        # Get indices for the requested languages
        self.language_indices = [all_labels.index(lx) for lx in set(languages) if lx in all_labels]

        # Store the subset of labels corresponding to the indices
        self.labels = [all_labels[i] for i in self.language_indices]

        # Set the prediction function based on the mode
        self.predict = self.predict_limit_after_softmax if mode == "after" else self.predict_limit_before_softmax

    def predict_limit_before_softmax(self, text, k=1):
        sentence_vector = self.model.get_sentence_vector(text)

        # Dot product with *only* the subset of the output matrix
        result_vector = np.dot(self.output_matrix[self.language_indices, :], sentence_vector)

        # Softmax over the subset
        softmax_result = np.exp(result_vector - np.max(result_vector)) / np.sum(
            np.exp(result_vector - np.max(result_vector))
        )

        top_k_indices = np.argsort(softmax_result)[-k:][::-1]
        top_k_labels = [self.labels[i] for i in top_k_indices]
        top_k_probs = softmax_result[top_k_indices]

        return tuple(top_k_labels), top_k_probs

    def predict_limit_after_softmax(self, text, k=1):
        sentence_vector = self.model.get_sentence_vector(text)

        # Dot product with the *full* output matrix
        result_vector = np.dot(self.output_matrix, sentence_vector)

        # Full softmax
        softmax_result = np.exp(result_vector - np.max(result_vector)) / np.sum(
            np.exp(result_vector - np.max(result_vector))
        )

        # Filter softmax results to *only* the allowed languages
        softmax_result = softmax_result[self.language_indices]

        top_k_indices = np.argsort(softmax_result)[-k:][::-1]
        top_k_labels = [self.labels[i] for i in top_k_indices]
        top_k_probs = softmax_result[top_k_indices]

        return tuple(top_k_labels), top_k_probs
