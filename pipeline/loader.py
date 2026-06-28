import gzip, json
from tqdm import tqdm

def stream_candidates(path: str):
    """Stream candidates one by one from .jsonl.gz or load from .json."""
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
            for c in candidates:
                yield c
        return

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_all_candidates(path: str) -> list[dict]:
    """Load all candidates into memory."""
    print(f"Loading candidates from {path}...")
    candidates = []
    for c in tqdm(stream_candidates(path)):
        candidates.append(c)
    print(f"Loaded {len(candidates):,} candidates.")
    return candidates
