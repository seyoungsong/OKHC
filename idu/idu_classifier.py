# idu_classifier.py


from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import ahocorasick  # type: ignore

# Key for the prediction in the output dictionary
PRED_KEY = "pred"


# --- Constants ---


class C:
    """Holds constants for Unicode ranges and grammatical rules."""

    CATS = ("kr", "cc", "other")
    CAT_NAME = {"other": "Other", "kr": "Idu", "cc": "Classical Chinese"}

    # Unicode ranges for Hanja (CJK Ideographs)
    HANJA = (
        (0x4E00, 0x9FFF),
        (0x3400, 0x4DBF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x30000, 0x3134F),
        (0xF900, 0xFAFF),
    )

    # Unicode ranges for Hangul (Korean)
    HANGUL = ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0xA960, 0xA97F), (0xD7B0, 0xD7FF))

    # Earthly Branches (for '乙' guard rule)
    # Negative lookahead: (?:HEAD)乙(?![EARTHLY])
    EARTHLY = "丑卯巳未酉亥子寅辰午申戌"


# --- Configuration ---


@dataclass
class Config:
    """Configuration for the classifier."""

    # Gate/preprocess
    hanja_ratio_min: float = 0.88
    token_len_min: int = 2

    # Resource paths
    resource_dir: Path = Path(__file__).resolve().parent / "resources"
    idu_dictionary_path: Path = resource_dir / "idu_dictionary.jsonl"
    idu_heads_path: Path = resource_dir / "idu_heads.json"
    idu_exclusions_path: Path = resource_dir / "idu_exclusions.json"
    idu_noun_exclusions_path: Path = resource_dir / "idu_noun_exclusions.json"

    # Length buckets & thresholds
    use_length_buckets: bool = True
    length_thresholds: Dict[Literal["short", "medium", "long"], float] = field(
        default_factory=lambda: {
            "short": 0.0104,  # <= 100 chars
            "medium": 0.008622,  # <= 300 chars
            "long": 0.003813,  # > 300 chars
        }
    )

    # Scoring
    count_by_frequency: bool = True
    score_denominator: Literal["hanja", "total"] = "hanja"
    enforce_grammar_hits: bool = True
    min_morph_hits_short: int = 1
    short_single_morph_delta: float = 0.0010

    def __post_init__(self):
        if self.score_denominator not in {"total", "hanja"}:
            raise ValueError("score_denominator must be 'total' or 'hanja'")


# --- Core Utilities ---


def _is_letter(ch: str) -> bool:
    """Check if a character is a Unicode letter."""
    return unicodedata.category(ch).startswith("L")


def _is_hanja(ch: str) -> bool:
    """Check if a character is a Hanja (CJK ideograph)."""
    cp = ord(ch)
    return any(a <= cp <= b for a, b in C.HANJA)


def _is_hangul(ch: str) -> bool:
    """Check if a character is a Hangul character."""
    cp = ord(ch)
    return any(a <= cp <= b for a, b in C.HANGUL)


def clean_letters_with_map(s: str) -> Tuple[str, List[int]]:
    """
    Normalize a string and strip non-letters.

    Returns:
        A tuple of (cleaned_string, position_map), where position_map
        maps indices from the cleaned string back to the original.
    """
    s = unicodedata.normalize("NFKC", s or "")
    out_chars: List[str] = []
    pos_map: List[int] = []
    for i, ch in enumerate(s):
        if _is_letter(ch):
            out_chars.append(ch)
            pos_map.append(i)
    return "".join(out_chars), pos_map


# --- Resource Loading Utilities ---


def _load_json_or_jsonl(path: Path) -> List[Any]:
    """Loads a JSON or JSONL file into a list."""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty resource file: {path}")

    try:
        if path.suffix.lower() == ".jsonl":
            out = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        out.append(json.loads(s))
            return out
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            raise ValueError(f"JSON at {path} must be a list")
    except Exception as e:
        raise ValueError(f"Parse error in {path}: {e}") from e


def _norm_token(w: str, min_len: int) -> str:
    """Normalize a token and filter by length."""
    if not w:
        return ""
    w2 = unicodedata.normalize("NFKC", str(w).strip())
    w2 = "".join(ch for ch in w2 if _is_letter(ch))
    return w2 if len(w2) >= min_len else ""


def _load_dictionary_set(path: Path, min_len: int, exclusions: Set[str]) -> Set[str]:
    """Loads an Idu dictionary from JSON/JSONL."""
    raw = _load_json_or_jsonl(path)
    out: Set[str] = set()
    for x in raw:
        token = x.get("idu_text", x) if isinstance(x, dict) else x
        n = _norm_token(token, min_len)
        if n and n not in exclusions:
            out.add(n)
    if not out:
        print(f"Warning: No valid dictionary entries loaded from {path}")
    return out


def _load_exclusions_set(path: Path) -> Set[str]:
    """Loads exclusion terms from a JSON/JSONL file."""
    data = _load_json_or_jsonl(path)
    it = data if isinstance(data, list) else [x for v in data.values() for x in (v if isinstance(v, list) else [])]
    out: Set[str] = set()
    for w in it:
        s = unicodedata.normalize("NFKC", str(w).strip())
        if s:
            out.add(s)
    if not out:
        print(f"Warning: No valid exclusion entries loaded from {path}")
    return out


def _load_noun_exclusions_set(path: Path) -> Set[str]:
    """Loads noun exclusion terms from a JSON/JSONL file."""
    items = _load_json_or_jsonl(path)
    out: Set[str] = set()
    for x in items:
        w = x.get("text", x) if isinstance(x, dict) else x
        s = unicodedata.normalize("NFKC", str(w).strip())
        if s:
            out.add(s)
    if not out:
        print(f"Warning: No valid noun exclusion entries loaded from {path}")
    return out


def _load_heads_list(path: Path) -> List[str]:
    """Loads Idu head terms from a JSON/JSONL file."""
    data = _load_json_or_jsonl(path)
    seen: Set[str] = set()
    out: List[str] = []
    for x in data:
        head = str(x).strip()
        if head and head not in seen:
            seen.add(head)
            out.append(head)
    if not out:
        print(f"Warning: No valid Idu head entries loaded from {path}")
    return out


# --- Matcher Class ---


class Matcher:
    """Finds Idu tokens using Aho-Corasick and Regex."""

    def __init__(self, dictionary: Set[str], idu_heads: List[str]):
        self.ac = self._build_automaton(dictionary)
        self.idu_head_guard_re = self._compile_idu_head_guard_regex(idu_heads)

    def _build_automaton(self, dictionary: Set[str]) -> Optional[ahocorasick.Automaton]:
        """Builds the Aho-Corasick automaton from the dictionary."""
        if not dictionary:
            print("Warning: Dictionary is empty; automaton not built.")
            return None
        ac = ahocorasick.Automaton()
        for token in dictionary:
            ac.add_word(token, token)
        ac.make_automaton()
        return ac

    def _compile_idu_head_guard_regex(self, heads: List[str]) -> re.Pattern[str]:
        """Builds the regex for Idu head + '乙' guard rule."""
        if not heads:
            print("Warning: Idu heads list is empty; guard regex will not match.")
            heads_alt = ""
        else:
            heads_alt = "|".join(map(re.escape, heads))
        # This regex finds (HEAD) + '乙' NOT followed by an Earthly Branch
        return re.compile(rf"(?:{heads_alt})乙(?![{C.EARTHLY}])")

    def find_spans(self, s: str) -> List[Tuple[int, int, str, str]]:
        """Finds all Idu-related spans in the text."""
        if not s:
            return []

        spans: List[Tuple[int, int, str, str]] = []

        # 1. Find dictionary matches
        if self.ac:
            for end_idx, token in self.ac.iter(s):
                start_idx = end_idx - len(token) + 1
                spans.append((start_idx, end_idx + 1, token, "dictionary"))

        # 2. Find Idu head guard matches
        for m in self.idu_head_guard_re.finditer(s):
            spans.append((m.start(), m.end(), m.group(0), "idu_head_guard"))

        # 3. Resolve overlaps, prioritizing guards and longer matches
        spans.sort(key=lambda t: (t[3] != "idu_head_guard", -(t[1] - t[0]), t[0]))
        used: Set[int] = set()
        out: List[Tuple[int, int, str, str]] = []
        for start_idx, end_idx, tok, src in spans:
            if any(i in used for i in range(start_idx, end_idx)):
                continue
            out.append((start_idx, end_idx, tok, src))
            used.update(range(start_idx, end_idx))
        return out


# --- Main Classifier ---


class IduClassifier:
    """
    Core classifier for Idu ('kr'), Classical Chinese ('cc'), or 'other'.
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self._load_resources()
        self.matcher = Matcher(self.dictionary, self.idu_heads)

    def _load_resources(self) -> None:
        """Loads all dictionaries and exclusion lists."""
        cfg = self.cfg
        self.exclusions = _load_exclusions_set(cfg.idu_exclusions_path)
        self.noun_exclusions = _load_noun_exclusions_set(cfg.idu_noun_exclusions_path)
        self.dictionary = _load_dictionary_set(cfg.idu_dictionary_path, cfg.token_len_min, self.exclusions)
        self.idu_heads = _load_heads_list(cfg.idu_heads_path)

    def _score(self, toks: List[str]) -> Tuple[int, int, List[str], List[str]]:
        """
        Scores a list of tokens, filtering exclusions.
        Returns (raw_count, weighted_count, valid_tokens, excluded_tokens)
        """
        if not toks:
            return 0, 0, [], []
        cnt = Counter(toks)
        raw = wgt = 0
        valid: List[str] = []
        excluded: List[str] = []
        for w, c in cnt.items():
            if w in self.exclusions:
                excluded.append(w)
                continue
            # Note: noun_exclusions are filtered *before* calling _score
            valid.append(w)
            raw += c
            wgt += c if self.cfg.count_by_frequency else 1
        return raw, wgt, valid, excluded

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies a single text string.

        Returns:
            A dictionary with the prediction and supporting metrics/features.
        """
        if not text:
            return self._build_result("other", reason="no_text")

        text = str(text)
        s, pos_map = clean_letters_with_map(text)
        n = len(s)
        if n == 0:
            return self._build_result("other", reason="no_valid_letters", orig_len=len(text))

        # --- Gating ---
        hg = sum(1 for ch in s if _is_hangul(ch))
        hj = sum(1 for ch in s if _is_hanja(ch))
        rhj = (hj / n) if n else 0.0

        if rhj < self.cfg.hanja_ratio_min:
            return self._build_result(
                "other",
                reason=f"low_hanja_{hj}({rhj:.3f})",
                orig_len=len(text),
                norm_len=n,
                hanja=hj,
                hangul=hg,
                metrics={"ratio": rhj, "threshold": self.cfg.hanja_ratio_min},
            )

        # --- KR/CC Classification Stage ---

        # 1. Find all potential Idu spans
        spans = self.matcher.find_spans(s)

        # 2. Filter spans based on context (e.g., '乙' guard rule)
        filtered_spans = []
        for st, en, tok, src in spans:
            if src == "dictionary" and tok.endswith("乙"):
                next_char = s[en : en + 1]
                if next_char and next_char in C.EARTHLY:
                    continue  # This is a false positive
            filtered_spans.append((st, en, tok, src))

        # 3. Separate dictionary hits for scoring
        toks = [t for _, _, t, src in filtered_spans if src in ["dictionary","idu_head_guard"]]
        toks = [t for t in toks if t not in self.exclusions]

        # 4. Separate hits into lexical (noun) vs. grammatical (morpheme)
        morph_hits_all = [t for t in toks if t in self.dictionary]
        lex_hits = [t for t in morph_hits_all if t in self.noun_exclusions]
        morph_hits_scored = [t for t in morph_hits_all if t not in self.noun_exclusions]

        # 5. Score the grammatical hits
        raw, wgt, valid, excluded = self._score(morph_hits_scored)

        # 6. Calculate final score ratio
        denom = hj if self.cfg.score_denominator == "hanja" else n
        denom = max(int(denom or 0), 1)
        score = wgt / denom

        # 7. Determine threshold based on text length
        if self.cfg.use_length_buckets:
            if n <= 100:
                thr = self.cfg.length_thresholds["short"]
            elif n <= 300:
                thr = self.cfg.length_thresholds["medium"]
            else:
                thr = self.cfg.length_thresholds["long"]
        else:
            thr = self.cfg.length_thresholds["medium"]

        # 8. Apply scoring rules and make decision
        if self.cfg.enforce_grammar_hits and not morph_hits_scored:
            score = 0.0  # Must have at least one grammar hit

        # Handle cases with only noun hits
        if morph_hits_all and not morph_hits_scored:
            score = 0.0

        uniq_morph = list(morph_hits_scored)
        eff_thr = thr

        # Apply special rules for short texts
        if n <= 100:
            if len(uniq_morph) < self.cfg.min_morph_hits_short:
                score = 0.0  # Not enough unique hits
            elif len(uniq_morph) == 1:
                # Be stricter if only one unique morph
                eff_thr = thr + self.cfg.short_single_morph_delta

        decision = "kr" if score >= eff_thr else "cc"

        # --- Format Output ---

        # Helper to map normalized span indices back to original text
        def _to_orig(start_norm: int, end_norm: int) -> Tuple[int, int]:
            start_norm = max(0, min(start_norm, len(pos_map) - 1))
            end_norm = max(1, min(end_norm, len(pos_map)))
            if not pos_map:
                return 0, 0
            return pos_map[start_norm], pos_map[end_norm - 1] + 1

        spans_out: List[Dict[str, Any]] = []
        for start_norm, end_norm, tok, src in spans:
            o_st, o_en = _to_orig(start_norm, end_norm)
            spans_out.append(
                {"start": o_st, "end": o_en, "start_norm": start_norm, "end_norm": end_norm, "text": tok, "source": src}
            )

        features = {
            "matched_scored_tokens": valid,
            "lex_only_hits": lex_hits,
            "excluded": excluded,
            "orig_text_len": len(text),
            "norm_text_len": n,
            "hanja": hj,
            "hangul": hg,
            "spans": spans_out,
        }
        metrics = {"raw": raw, "weighted": wgt, "ratio": score, "threshold": thr}
        return {PRED_KEY: decision, "metrics": metrics, "features": features}

    def _build_result(
        self,
        pred: str,
        reason: str,
        orig_len: int = 0,
        norm_len: int = 0,
        hanja: int = 0,
        hangul: int = 0,
        metrics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Helper to create a standardized result dictionary."""
        return {
            PRED_KEY: pred,
            "metrics": metrics or {"raw": 0, "weighted": 0, "ratio": 0.0, "threshold": None},
            "features": {
                "reason": reason,
                "orig_text_len": orig_len,
                "norm_text_len": norm_len,
                "hanja": hanja,
                "hangul": hangul,
            },
        }


# --- Demonstration ---


def _create_dummy_resources(res_dir: Path):
    """Creates empty placeholder resource files for the demo."""
    res_dir.mkdir(exist_ok=True)

    # Idu dictionary (JSONL format)
    # '等乙' (dunguhl) - common grammar
    # '去乙' (guyeol) - common grammar
    # '爲去乎' (wigeoho) - common grammar
    # '水刺' (sura) - common noun
    (res_dir / "idu_dictionary.jsonl").write_text(
        '{"idu_text": "等乙"}\n{"idu_text": "去乙"}\n{"idu_text": "爲去乎"}\n{"idu_text": "水刺"}\n', encoding="utf-8"
    )

    # Idu heads (for '乙' rule)
    (res_dir / "idu_heads.json").write_text('["爲", "去"]', encoding="utf-8")

    # General exclusions (common CC words)
    (res_dir / "idu_exclusions.json").write_text('["不得"]', encoding="utf-8")

    # Noun exclusions (words that are Idu but not grammar)
    (res_dir / "idu_noun_exclusions.json").write_text('[{"text": "水刺"}]', encoding="utf-8")
    print(f"Created dummy resource files in: {res_dir.resolve()}")


if __name__ == "__main__":
    """
    Demonstrates how to use the IduClassifier.
    This will create dummy resource files in a 'resources/' subdirectory.
    """
    print("--- Idu Classifier Demo ---")

    # 1. Setup Config and create dummy files
    # We point the config to a local 'resources' directory
    demo_res_dir = Path(__file__).resolve().parent / "resources"
    _create_dummy_resources(demo_res_dir)

    cfg = Config(resource_dir=demo_res_dir)

    # 2. Initialize the classifier
    # This will load the resources we just created
    try:
        classifier = IduClassifier(cfg)
        print("Classifier initialized successfully.")
    except Exception as e:
        print(f"\n[ERROR] Could not initialize classifier: {e}")
        print("Please ensure 'ahocorasick' is installed (`pip install pyahocorasick`)")
        exit(1)

    # 3. Define test cases
    test_cases = {
        "pure_cc": "事勢亦無所不得已也",  # Classical Chinese, low Idu score
        "idu_grammar": "田地等乙許與爲去乎",  # '等乙' and '爲去乎' are Idu grammar
        "idu_noun": "水刺間所用銀鉢里蓋臺一坐",  # '水刺' is an Idu noun, should score 0
        "idu_guard_false": "去乙丑年徙于安邊",  # '去乙' + Earthly Branch = CC
        "idu_guard_true": "圍犯爲去乙多數射中勝戰追擊",  # '去乙' + other = Idu
        "low_hanja": "Hello this is English.",  # Should be 'other'
        "hangul_mix": "王曰. 吾遣汝爲. 그렇다.",  # Mixed, but should pass Hanja gate
    }

    print("\n--- Running Classifications ---")

    # 4. Run classification
    for name, text in test_cases.items():
        result = classifier.classify(text)

        pred = result[PRED_KEY]
        score = result["metrics"]["ratio"]
        reason = result["features"].get("reason", "n/a")

        print(f"\nTest: '{name}'")
        print(f'  Text: "{text}"')
        print(f"  -> Prediction: {pred.upper()} (Score: {score:.4f}, Reason: {reason})")

        # Show matched spans for Idu/CC cases
        if pred in ("kr", "cc"):
            spans = result["features"]["spans"]
            if spans:
                print("  Matches:")
                for span in spans:
                    print(f"    - '{span['text']}' ({span['source']})")
            else:
                print("  Matches: (None)")
