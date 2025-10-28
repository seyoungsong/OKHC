# text_preprocessing.py


import unicodedata

import pandas as pd
import regex as re
from loguru import logger

from . import tool, utils_lid

# --- Text Normalization Functions ---


def squeeze_whites(text: str, keep_n: bool = True) -> str:
    """Collapses consecutive whitespace characters."""
    if not isinstance(text, str):
        return text
    if keep_n:
        # Remove spaces/tabs around newlines
        text = re.sub(r"[^\S\n]*\n[^\S\n]*", "\n", text)
        # Multiple newlines become single newline
        text = re.sub(r"\n+", "\n", text)
        # Multiple spaces/tabs become single space
        text = re.sub(r"[^\S\n]+", " ", text)
    else:
        # Collapse all whitespace to a single space
        text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_control_chars(s: str) -> str:
    """Removes Unicode control characters (C0, C1) except tab and newline."""
    if not isinstance(s, str):
        return s
    # Regex matches C0 controls (excluding \t, \n), DEL, and C1 controls
    control_char_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
    return control_char_re.sub("", s)


def normalize_text(s: str) -> str:
    """
    Applies a simple text normalization pipeline.
    1. NFKC Unicode Normalization
    2. Whitespace Squeezing
    3. Control Character Removal
    """
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKC", s)  # Unicode normalization
    s = squeeze_whites(s, keep_n=True)  # Whitespace cleaning
    s = remove_control_chars(s)  # Control character removal
    return s


# --- Main Data Processing Function ---


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies normalization and computes metadata for the input DataFrame.

    Note: Assumes `pandarallel.initialize()` has been called by the user.

    Args:
        df: A pandas DataFrame containing a 'text' column.

    Returns:
        A new DataFrame with normalized text and computed
        'script.*', 'language', and 'language_score' columns.
    """
    # Ensure 'text' column exists
    if "text" not in df.columns:
        logger.error("Input DataFrame must contain a 'text' column.")
        raise ValueError("Missing 'text' column")

    df_processed = df.copy()

    logger.info("Normalizing 'text' column...")
    # Apply text normalization in parallel
    df_processed["text"] = df_processed["text"].parallel_apply(normalize_text)

    logger.info("Computing script ratios...")
    # Compute script ratios (e.g., kor, han, lat)
    script_ratios: pd.Series = df_processed["text"].parallel_apply(lambda x: tool.compute_script_ratio(x))
    df_script = pd.json_normalize(script_ratios)
    df_script.rename(columns=lambda x: f"script.{x}", inplace=True)

    logger.info("Computing language identification (LID)...")
    # Define target languages for the LID model
    languages = [
        "cmn_Hani",  # Mandarin Chinese
        "kor_Hang",  # Korean
        "jpn_Jpan",  # Japanese
        "lzh_Hani",  # Literary Chinese
        "eng_Latn",  # English
        "spa_Latn",  # Spanish
    ]
    # Compute language and score
    lid_results: pd.Series = df_processed["text"].parallel_apply(
        lambda x: utils_lid.langid_custom(text=x, languages=languages, mode="before")
    )
    df_lid = pd.DataFrame(lid_results.tolist(), columns=["language", "language_score"])

    logger.info("Merging results...")
    # Drop old metadata columns if they exist, before merging new ones
    cols_to_drop = list(df_script.columns) + list(df_lid.columns)
    df_processed.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Concatenate original df with new metadata
    df_processed = pd.concat([df_processed, df_script, df_lid], axis=1)

    return df_processed
