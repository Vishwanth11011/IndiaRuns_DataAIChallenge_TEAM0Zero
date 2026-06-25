"""
pipeline/data_loader.py
-----------------------
Efficient loading of candidates from:
  - candidates.jsonl  (~487 MB, one JSON object per line)
  - sample_candidates.json  (list of JSON objects, for quick testing)

Each loaded candidate is a plain dict matching candidate_schema.json.
"""

import json
import logging
from pathlib import Path
from typing import Generator, List

logger = logging.getLogger(__name__)


def stream_candidates_jsonl(filepath: str) -> Generator[dict, None, None]:
    """
    Stream candidates from a JSONL file one-by-one.
    Memory-efficient for the full 487 MB candidates.jsonl.
    Yields each parsed candidate dict.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {filepath}")

    total = 0
    errors = 0
    logger.info(f"Streaming candidates from: {path.name}")

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                total += 1
                yield candidate
            except json.JSONDecodeError as e:
                errors += 1
                logger.warning(f"Line {line_no}: JSON parse error — {e}")

    logger.info(f"Streamed {total:,} candidates ({errors} errors) from {path.name}")


def load_candidates_json(filepath: str) -> List[dict]:
    """
    Load all candidates from a JSON array file (sample_candidates.json).
    Returns a list of candidate dicts.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {filepath}")

    logger.info(f"Loading candidates from: {path.name}")
    with open(path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    if not isinstance(candidates, list):
        raise ValueError(f"Expected a JSON array in {path.name}")

    logger.info(f"Loaded {len(candidates):,} candidates from {path.name}")
    return candidates


def load_candidates(filepath: str, sample_mode: bool = False) -> List[dict]:
    """
    Auto-detect file format and load all candidates into memory.
    For JSONL files, this loads everything — use stream_candidates_jsonl()
    if memory is a concern (e.g. for the full 487 MB file).

    Args:
        filepath: Path to candidates.jsonl or sample_candidates.json
        sample_mode: If True, only load first 5000 candidates (for fast testing)
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return load_candidates_json(filepath)
    elif suffix == ".jsonl":
        candidates = []
        for i, c in enumerate(stream_candidates_jsonl(filepath)):
            if sample_mode and i >= 5000:
                logger.info(f"Sample mode: stopping at {i} candidates")
                break
            candidates.append(c)
        return candidates
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def get_candidate_count(filepath: str) -> int:
    """
    Count lines in a JSONL file without loading everything into memory.
    Fast O(n) line scan.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".json":
        candidates = load_candidates_json(filepath)
        return len(candidates)
    else:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
