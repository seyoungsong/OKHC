# tool.py


import unicodedata

import regex as re

CLEAN_REGEX = re.compile(r"[\p{P}\p{Z}\s\dᆞㆍ]+")
KOR_REGEX = re.compile(
    r"""
    \p{Script=Hangul}|
    [\u3130-\u318F]|
    [\u1100-\u11FF]|   # Hangul Jamo (Combining)
    [\uA960-\uA97F]|   # Hangul Jamo Extended-A
    [\uD7B0-\uD7FF]|   # Hangul Jamo Extended-B
    [\u302E-\u302F]|   # Tone marks (방점)
    [\uE0BC-\uEFFF]|   # Hanyang PUA
    [\uF100-\uF66E]|   # Hanyang PUA
    [\uF784-\uF800]|   # Hanyang PUA
    [\uF806-\uF864]|   # Hanyang PUA
    [\uF86A-\uF8F7]|   # Hanyang PUA
    [\uE001-\uE0A0]    # Jeju font PUA
    """,
    re.VERBOSE,
)
OKO_REGEX = re.compile(
    r"""
    [\u1100-\u11FF]|   # Hangul Jamo (Combining)
    [\uA960-\uA97F]|   # Hangul Jamo Extended-A
    [\uD7B0-\uD7FF]|   # Hangul Jamo Extended-B
    [\u302E-\u302F]|   # Tone marks (방점)
    [\uE0BC-\uEFFF]|   # Hanyang PUA
    [\uF100-\uF66E]|   # Hanyang PUA
    [\uF784-\uF800]|   # Hanyang PUA
    [\uF806-\uF864]|   # Hanyang PUA
    [\uF86A-\uF8F7]|   # Hanyang PUA
    [\uE001-\uE0A0]    # Jeju font PUA
    """,
    re.VERBOSE,
)
JPN_REGEX = re.compile(r"\p{Script=Hiragana}|\p{Script=Katakana}")
HAN_REGEX = re.compile(r"\p{Script=Han}")
LAT_REGEX = re.compile(r"\p{Script=Latin}")


def compute_script_ratio(text: str, debug: bool = False, len_only: bool = False):
    """
    Computes the ratio of different scripts in a text.
    """
    empty_result = {"kor": 0.0, "oko": 0.0, "jpn": 0.0, "han": 0.0, "lat": 0.0}
    if not isinstance(text, str):
        return empty_result

    # Normalize and clean text
    text_c = unicodedata.normalize("NFKC", text)
    text_c = re.sub(CLEAN_REGEX, "", text_c)
    if len(text_c) == 0:
        return empty_result

    # Get character counts
    kor_len = len(KOR_REGEX.findall(text_c))
    oko_len = len(OKO_REGEX.findall(text_c))
    jpn_len = len(JPN_REGEX.findall(text_c))
    han_len = len(HAN_REGEX.findall(text_c))
    lat_len = len(LAT_REGEX.findall(text_c))

    # Calculate ratios
    total_len = kor_len + jpn_len + han_len + lat_len
    if total_len == 0:
        return empty_result

    kor_ratio = kor_len / total_len
    oko_ratio = oko_len / total_len
    jpn_ratio = jpn_len / total_len
    han_ratio = han_len / total_len
    lat_ratio = lat_len / total_len

    out = {"kor": kor_ratio, "oko": oko_ratio, "jpn": jpn_ratio, "han": han_ratio, "lat": lat_ratio}
    return out


def count_valid_script(text: str):
    """
    Counts characters from a basic set of "valid" scripts vs. "other" scripts.
    """
    if not isinstance(text, str) or not text:
        return {"count_all": 0, "count_val": 0, "count_etc": 0}

    # Clean numbers and horizontal whitespace
    pat2 = re.compile(r"\p{Number}|\p{Zs}")
    text = pat2.sub("", text)

    total_len = len(text)

    # Count basic scripts (note: excludes Old Korean PUA)
    pat1 = re.compile(r"\p{Script=Hangul}|\p{Script=Hiragana}|\p{Script=Katakana}|\p{Script=Han}|\p{Script=Latin}")
    val_count = len(pat1.findall(text))
    etc_count = total_len - val_count

    out = {"count_all": total_len, "count_val": val_count, "count_etc": etc_count}
    return out
