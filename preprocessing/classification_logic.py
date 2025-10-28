# classification_logic.py


import pandas as pd
import regex as re

from . import tool, utils_lid

# --- Regex patterns compiled once ---

PAT_NEWS_JUNK = re.compile(r"(\p{N}|\p{P}|\p{Z}|\n|\s)+", re.UNICODE)
PAT_GONGU_MT = re.compile(r"((\p{Hangul}+\s*[\(\[]\p{Han}+[\)\]])|(\p{Han}+\s*[\(\[]\p{Hangul}+[\)\]]))", re.UNICODE)
PAT_NUMERIC = re.compile(r"(\p{N}|\p{P}|\p{Z}|\n|\s)+")

# --- Heuristic Rule Functions (from gen_category.py) ---


def cat_rule_gaksa(row: pd.Series) -> str:
    x1 = row.to_dict()
    cid = None
    if x1["script.han"] >= 1.0:
        cid = "hj"
    elif (x1["script.kor"] > 0) and (x1["script.han"] > 0) and (x1["script.oko"] == 0):
        cid = "hj+ko"
        if "주-D" in x1["text"]:
            cid = "hj"
    elif (x1["script.oko"] > 0) and (x1["script.han"] > 0):
        cid = "hj+oko"
    elif (x1["script.kor"] > 0) and (x1["script.han"] == 0) and (x1["script.oko"] == 0):
        cid = "ko"
    elif (x1["script.oko"] > 0) and (x1["script.han"] == 0):
        cid = "oko"
    elif (x1["script.jpn"] > 0) and (x1["script.kor"] == 0):
        cid = "ja"
    else:
        cid = "unk"
    return cid


def cat_rule_jpn(row: pd.Series) -> str:
    x1 = row.to_dict()
    cid = None
    if (x1["language"] == "jpn_Jpan") and (x1["script.jpn"] > 0):
        cid = "ja"
    elif (x1["language"] == "zho_Hani") and (x1["script.jpn"] > 0):
        cid = "ja"
    elif (x1["language"] == "eng_Latn") and (x1["script.lat"] > 0):
        cid = "en"
    elif (x1["language"] == "fra_Latn") and (x1["script.lat"] > 0):
        cid = "fr"
    elif (x1["script.jpn"] == 0) and (x1["script.oko"] > 0) and (x1["script.han"] > 0):
        cid = "hj+oko"
    elif (x1["script.jpn"] == 0) and (x1["script.oko"] > 0) and (x1["script.han"] == 0):
        cid = "oko"
    elif (x1["script.jpn"] == 0) and (x1["script.kor"] > 0) and (x1["script.han"] > 0):
        cid = "hj+ko"
        if "문서제목" in x1["text"]:
            cid = "hj"
    elif (x1["script.jpn"] == 0) and (x1["script.kor"] > 0) and (x1["script.han"] == 0):
        cid = "ko"
    elif (x1["script.jpn"] == 0) and (x1["script.han"] > 0) and (x1["script.kor"] == 0):
        cid = "hj"
    else:
        cid = "unk"
    return cid


def cat_rule_magazine(row: pd.Series) -> str:
    x1 = row.to_dict()
    cid = None
    if (x1["language"] == "jpn_Jpan") and (x1["script.jpn"] > 0):
        cid = "ja"
    elif x1["script.han"] >= 1.0:
        cid = "hj"
    elif (x1["script.han"] > 0) and (x1["script.kor"] > 0) and (x1["script.oko"] == 0):
        cid = "hj+ko"
    elif (x1["script.han"] > 0) and (x1["script.oko"] > 0):
        cid = "hj+oko"
    elif (x1["language"] == "eng_Latn") and (x1["script.lat"] > 0):
        cid = "en"
    elif (x1["script.oko"] > 0) and (x1["script.han"] == 0):
        cid = "oko"
    elif (x1["script.kor"] > 0) and (x1["script.han"] == 0):
        cid = "ko"
    elif x1["script.jpn"] > 0:
        cid = "ja"
    elif x1["script.han"] > 0:
        cid = "hj"
    else:
        cid = "unk"
    return cid


def cat_rule_news(row: pd.Series) -> str:
    x1 = row.to_dict()
    cid = "unk"
    if x1["script.han"] >= 1.0:
        cid = "hj"
    elif (x1["script.han"] > 0) and (x1["script.kor"] > 0) and (x1["script.oko"] == 0):
        cid = "hj+ko"
    elif (x1["script.han"] > 0) and (x1["script.oko"] > 0):
        cid = "hj+oko"
    elif (x1["script.kor"] > 0) and (x1["script.han"] == 0):
        cid = "ko"
    elif (x1["script.oko"] > 0) and (x1["script.han"] == 0):
        cid = "oko"
    elif (x1["language"] == "eng_Latn") and (x1["script.lat"] > 0):
        cid = "en"
    elif (x1["language"] == "jpn_Jpan") and (x1["script.jpn"] > 0):
        cid = "ja"
    elif x1["script.jpn"] > 0:
        cid = "ja"
    elif x1["script.lat"] >= 1.0:
        cid = "en"
    elif (x1["script.lat"] > 0) and (x1["script.han"] > 0):
        cid = "hj+en"
    else:
        if PAT_NEWS_JUNK.fullmatch(x1["text"].strip()):
            cid = "num"
    return cid


def cat_rule_gongu(row: pd.Series) -> str:
    x1 = row.to_dict()
    cid = "unk"
    text = x1["text"]
    match_concat = "".join([m.group(0) for m in PAT_GONGU_MT.finditer(text)])
    if len(match_concat) > 0 and ((len(match_concat) / len(text)) >= 0.7):
        cid = "hj-mt"
        return cid

    if (x1["script.kor"] > 0) and (x1["script.han"] == 0) and (x1["script.oko"] == 0):
        cid = "ko"
    elif (x1["script.oko"] > 0) and (x1["script.han"] == 0):
        cid = "oko"
    elif (x1["script.jpn"] > 0) and (x1["script.kor"] == 0) and (x1["script.oko"] == 0):
        cid = "ja"
    elif (x1["script.han"] > 0) and (x1["script.kor"] == 0) and (x1["script.oko"] == 0):
        cid = "hj"
    elif (x1["language"] == "eng_Latn") and (x1["script.lat"] > 0):
        cid = "en"
    elif (x1["script.han"] > 0) and (x1["script.kor"] > 0) and (x1["script.oko"] == 0):
        cid = "hj+ko"
    elif (x1["script.han"] > 0) and (x1["script.oko"] > 0):
        cid = "hj+oko"

    return cid


def cat_rule_aks_etc(row: pd.Series) -> str:
    x1 = row.to_dict()
    cid = "unk"
    if x1["script.han"] >= 1.0:
        cid = "hj"
    elif (x1["script.han"] > 0) and (x1["script.oko"] > 0):
        cid = "hj+oko"
    elif (x1["script.han"] > 0) and (x1["script.kor"] > 0) and (x1["script.oko"] == 0):
        cid = "hj+ko"
    elif (x1["script.oko"] > 0) and (x1["script.han"] == 0):
        cid = "oko"
    elif (x1["script.kor"] > 0) and (x1["script.han"] == 0):
        cid = "ko"
    elif (x1["language"] == "jpn_Jpan") and (x1["script.jpn"] > 0):
        cid = "ja"
    elif (x1["language"] == "eng_Latn") and (x1["script.lat"] > 0):
        cid = "en"
    elif (x1["script.han"] > 0) and (x1["script.kor"] == 0) and (x1["script.oko"] == 0):
        cid = "hj"

    return cid


# --- Main Classification Function ---


def classify_dataframe(df1: pd.DataFrame) -> pd.DataFrame:
    """
    Applies heuristic categorization and schema finalization to a DataFrame.
    Assumes df1 has columns: id, year, text, language,
    script.kor, script.oko, script.jpn, script.han, script.lat
    """

    # --- Part 1: Heuristic Category (from gen_category.py) ---
    df1 = df1.copy()
    df1["src_id"] = df1["id"].str.split(":").str[0]
    df1["cid"] = None

    idx = df1["text"].str.len() == 0
    df1.loc[idx, "cid"] = "empty"

    # 1. all Hanja
    tgt_src_ids = ["bibyeonsa", "goryeosa", "ilseongnok", "sagi", "sillok", "sjw"]
    idx = df1["src_id"].isin(tgt_src_ids)
    df1.loc[idx, "cid"] = "hj"

    # 2. all nko
    tgt_src_ids = ["kisu_journal", "kisu_literary"]
    idx = df1["src_id"].isin(tgt_src_ids)
    df1.loc[idx, "cid"] = "nko"

    # 3. nko and others
    tgt_src_ids = ["kcna_jp"]
    idx = df1["src_id"].isin(tgt_src_ids)
    lang_map = {"kor_Hang": "nko", "eng_Latn": "en", "spa_Latn": "es"}
    df1.loc[idx, "cid"] = df1.loc[idx, "language"].map(lang_map)

    # 4. hj, hj_oko, hj+ko
    tgt_src_ids = ["klc", "gaksa", "gaksa_modern"]
    idx = df1["src_id"].isin(tgt_src_ids) & (df1["text"].str.len() > 0)
    df1.loc[idx, "cid"] = df1[idx].apply(cat_rule_gaksa, axis=1)
    lang_fix = {"hj+ko": "hj+oko", "oko": "hj+oko", "unk": "hj"}
    df1.loc[idx, "cid"] = df1.loc[idx, "cid"].replace(lang_fix)

    # 5. jpn_records
    tgt_src_ids = ["jpn_records"]
    idx = df1["src_id"].isin(tgt_src_ids) & (df1["text"].str.len() > 0)
    if idx.sum() > 0:
        languages = ["eng_Latn", "fra_Latn", "jpn_Jpan"]
        col_lang = df1.loc[idx, "text"].apply(
            lambda x: utils_lid.langid_custom(text=x, languages=languages, mode="before")
        )
        df1_lang = pd.DataFrame(col_lang.tolist(), index=df1.loc[idx].index, columns=["language", "language_score"])
        df1.loc[idx, "language"] = df1_lang["language"]
        df1.loc[idx, "language_score"] = df1_lang["language_score"]
    df1.loc[idx, "cid"] = df1[idx].apply(cat_rule_jpn, axis=1)
    lang_fix = {"unk": "hj"}
    df1.loc[idx, "cid"] = df1.loc[idx, "cid"].replace(lang_fix)

    # 6. magazine
    tgt_src_ids = ["magazine"]
    idx = df1["src_id"].isin(tgt_src_ids) & (df1["text"].str.len() > 0)
    df1.loc[idx, "cid"] = df1[idx].apply(cat_rule_magazine, axis=1)
    lang_fix = {"unk": "hj"}
    df1.loc[idx, "cid"] = df1.loc[idx, "cid"].replace(lang_fix)

    # 7. news_archive, newslibrary
    tgt_src_ids = ["news_archive", "newslibrary"]
    idx = df1["src_id"].isin(tgt_src_ids) & (df1["text"].str.len() > 0)
    if idx.sum() > 0:
        df1.loc[idx, "cid"] = df1[idx].apply(cat_rule_news, axis=1)
    lang_fix = {"unk": "zxx"}
    df1.loc[idx, "cid"] = df1.loc[idx, "cid"].replace(lang_fix)

    # 8. gongu
    tgt_src_ids = ["gongu"]
    idx = df1["src_id"].isin(tgt_src_ids) & (df1["text"].str.len() > 0)
    df1.loc[idx, "cid"] = df1[idx].apply(cat_rule_gongu, axis=1)
    lang_fix = {"hj-mt": "hj"}
    df1.loc[idx, "cid"] = df1.loc[idx, "cid"].replace(lang_fix)

    # 9. ext_aks_etc
    tgt_src_ids = ["aks_book", "aks_collection", "aks_letter", "aks_royal", "kyu_old", "nhm_archive"]
    idx = df1["src_id"].isin(tgt_src_ids) & (df1["text"].str.len() > 0)
    df1.loc[idx, "cid"] = df1[idx].apply(cat_rule_aks_etc, axis=1)
    lang_fix = {"unk": "zxx"}
    df1.loc[idx, "cid"] = df1.loc[idx, "cid"].replace(lang_fix)

    idx_fix = (df1["cid"] == "hj+ko") & (df1["src_id"].isin(["kyu_old", "aks_collection"]))
    df1.loc[idx_fix, "cid"] = "hj"

    # Final fixes for empty/numeric
    idx = df1["text"].str.strip() == ""
    df1.loc[idx, "cid"] = "empty"

    idx_num = df1["text"].apply(lambda x: bool(PAT_NUMERIC.fullmatch(str(x).strip())))
    df1.loc[idx_num, "cid"] = "num"

    df1["category"] = df1["cid"]

    # --- Part 2: Schema Finalization (from gen_internal.py) ---

    # Drop rows with null-or-undefined text
    idx = df1["category"].isin(["empty", "num", "zxx"])
    df1 = df1[~idx].reset_index(drop=True)

    # Fix: ko/oko cannot exist before 1446
    idx = df1["category"].str.contains("ko") & (df1["year"] < 1446)
    df1.loc[idx, "category"] = df1.loc[idx, "category"].apply(
        lambda x: str(x).replace("+oko", "").replace("+ko", "").strip()
    )

    # Convert category to language and script
    conv_lang_map = {
        "hj": "Hanmun",
        "hj+oko": None,
        "hj+ko": None,
        "nko": "North Korean",
        "ko": None,
        "en": "English",
        "ja": "Japanese",
        "es": "Spanish",
        "hj+en": "Hanmun",
        "oko": None,
        "fr": "French",
    }
    conv_script_map = {
        "hj": "Hanja",
        "hj+oko": "Hanja, Old Hangeul",
        "hj+ko": "Hanja, Hangeul",
        "nko": "Hangeul",
        "ko": "Hangeul",
        "en": "Latin",
        "ja": "Kanji, Kana",
        "es": "Latin",
        "hj+en": "Hanja, Latin",
        "oko": "Old Hangeul",
        "fr": "Latin",
    }
    df1["language"] = df1["category"].map(conv_lang_map)
    df1["script"] = df1["category"].map(conv_script_map)

    # Add language to Korean categories based on year
    idx = df1["category"].isin(["oko", "hj+oko", "hj+ko", "ko"])
    idx_mid = idx & (df1["year"].notna()) & df1["year"].apply(lambda x: 918 <= x < 1600)
    idx_emk = idx & (df1["year"].notna()) & df1["year"].apply(lambda x: 1600 <= x < 1894)
    idx_mok = idx & (df1["year"].notna()) & df1["year"].apply(lambda x: x >= 1894)
    idx_etc = idx & (df1["year"].isna())

    df1.loc[idx_mid, "language"] = "Middle Korean"
    df1.loc[idx_emk, "language"] = "Early Modern Korean"
    df1.loc[idx_mok, "language"] = "Modern Korean"
    df1.loc[idx_etc, "language"] = "Korean"

    # Fix: Re-classify Middle Korean with high Hanja ratio as Hanmun
    idx = df1["language"] == "Middle Korean"
    if idx.sum() > 0:
        df2 = df1[idx].copy().reset_index(drop=True)
        df2_x = pd.json_normalize(df2["text"].apply(tool.compute_script_ratio))
        df2 = pd.concat([df2.reset_index(drop=True), df2_x.reset_index(drop=True)], axis=1)

        han_id = df2[df2["han"] > 0.8]["id"]
        han_idx = df1["id"].isin(han_id)
        df1.loc[han_idx, "language"] = "Hanmun"

    # Clean up intermediate columns
    df_final = df1.drop(columns=["cid", "src_id"], errors="ignore")

    return df_final
