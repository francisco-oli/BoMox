"""Train/val/test/held-out split of the Motion-X subsets, with triples matched by file name."""
import os
import random

BASE_SMPL_PATH = './Motion-X++/Motion-X++/motion/global_motion'
BASE_TEXT_PATH = './Motion-X++/Motion-X++/text'
BASE_VIDEO_PATH = './Motion-X++/Motion-X++/preprocessed_videos'

# In-distribution subsets (80/10/10 split)
SUBSETS = ['idea400', 'haa500', 'humman', 'music', 'perform']

# Held-out subsets (never trained on)
HELD_OUT_SUBSETS = ['animation', 'kungfu']

SPLIT_SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1  # remainder (0.1) goes to test


def get_samples(directory, extension):
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                all_files.append(os.path.join(root, file))
    return all_files, len(all_files)


def stem(path):
    """File name without directory or extension."""
    return os.path.splitext(os.path.basename(path))[0]


def _list_subset(subset):
    m, _ = get_samples(os.path.join(BASE_SMPL_PATH, subset), '.json')
    t, _ = get_samples(os.path.join(BASE_TEXT_PATH, subset), '.txt')
    v, _ = get_samples(os.path.join(BASE_VIDEO_PATH, subset), '.pt')
    return sorted(m), sorted(t), sorted(v)


def _index_by_stem(paths, subset, modality):
    """name -> path; duplicate names are dropped."""
    index, dups = {}, set()
    for p in paths:
        s = stem(p)
        if s in index:
            dups.add(s)
        index[s] = p
    for s in dups:
        del index[s]
    if dups:
        print(f"[WARNING] {subset}/{modality}: {len(dups)} duplicate names dropped, "
              f"e.g. {sorted(dups)[:3]}")
    return index


def _gather_subset(subset, verbose=True):
    """Name-matched (motion, text, video) triples for one subset."""
    m, t, v = _list_subset(subset)
    M = _index_by_stem(m, subset, 'motion')
    T = _index_by_stem(t, subset, 'text')
    V = _index_by_stem(v, subset, 'video')

    keys = sorted(M.keys() & T.keys() & V.keys())

    if verbose:
        all_keys = M.keys() | T.keys() | V.keys()
        if len(all_keys) != len(keys):
            tv = T.keys() & V.keys()
            print(f"[WARNING] {subset}: {len(all_keys) - len(keys)} names not present in all "
                  f"three modalities -> dropped. "
                  f"motion w/o text+video={len(M.keys() - tv)}, "
                  f"text+video w/o motion={len(tv - M.keys())}, "
                  f"text/video disagree={len(T.keys() ^ V.keys())}")
            print(f"          e.g. motion-only: {sorted(M.keys() - tv)[:3]}")
            print(f"          e.g. text+video-only: {sorted(tv - M.keys())[:3]}")

    return [(M[k], T[k], V[k]) for k in keys]


def _legacy_split_chunks(subset, rng):
    """Original positional split, used only to keep split membership."""
    m, t, v = _list_subset(subset)
    n = min(len(m), len(t), len(v))
    legacy = list(zip(m[:n], t[:n], v[:n]))
    rng.shuffle(legacy)

    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return {'train': legacy[:train_end],
            'val': legacy[train_end:val_end],
            'test': legacy[val_end:]}


def build_split(subsets=None, seed=SPLIT_SEED, verbose=True):
    """Per-subset train/val/test split of name-matched triples."""
    if subsets is None:
        subsets = SUBSETS

    rng = random.Random(seed)
    result = {'train': [], 'val': [], 'test': []}

    if verbose:
        print(f"--- DATASET SPLIT (per-subset, name-matched, "
              f"{int(TRAIN_FRAC*100)}/{int(VAL_FRAC*100)}/{int(round((1-TRAIN_FRAC-VAL_FRAC)*100))}) ---")

    for subset in subsets:
        chunks = _legacy_split_chunks(subset, rng)
        matched = {stem(t): (m, t, v) for m, t, v in _gather_subset(subset, verbose)}

        counts = {}
        for split_name, chunk in chunks.items():
            kept = [matched[stem(t)] for _, t, _ in chunk if stem(t) in matched]
            result[split_name].extend(kept)
            counts[split_name] = len(kept)

        if verbose:
            total = sum(len(c) for c in chunks.values())
            print(f"  {subset}: {total} listed -> train={counts['train']}, "
                  f"val={counts['val']}, test={counts['test']} "
                  f"(dropped {total - sum(counts.values())})")

    if verbose:
        print(f"\nTraining Samples:   {len(result['train'])}")
        print(f"Validation Samples: {len(result['val'])}")
        print(f"Testing Samples:    {len(result['test'])}\n")

    return result


def legacy_train_motion_leaks(subsets=None, seed=SPLIT_SEED):
    """Val/test samples whose motion was in the legacy training split (pre-fix checkpoints only)."""
    if subsets is None:
        subsets = SUBSETS

    rng = random.Random(seed)  # same RNG sequence as build_split
    leaks = {'val': set(), 'test': set()}
    for subset in subsets:
        chunks = _legacy_split_chunks(subset, rng)
        old_train_motions = {stem(m) for m, _, _ in chunks['train']}
        for split_name in ('val', 'test'):
            for _, t, _ in chunks[split_name]:
                if stem(t) in old_train_motions:
                    leaks[split_name].add(stem(t))
    return leaks


def build_held_out(subsets=None, verbose=True):
    """All samples of the held-out subsets (not split)."""
    if subsets is None:
        subsets = HELD_OUT_SUBSETS

    all_samples = []
    if verbose:
        print(f"--- HELD-OUT SUBSETS (generalization test only, never trained on) ---")
    for subset in subsets:
        combined = _gather_subset(subset, verbose)
        all_samples.extend(combined)
        if verbose:
            print(f"  {subset}: {len(combined)} samples")
    return all_samples


def unzip_triples(triples):
    """[(m,t,v), ...] -> (list[m], list[t], list[v])"""
    if not triples:
        return [], [], []
    m, t, v = zip(*triples)
    return list(m), list(t), list(v)