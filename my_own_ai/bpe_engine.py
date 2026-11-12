"""Original deterministic BPE implementation using local merge updates.

Pair frequencies and their positions update only around each replacement,
avoiding a full corpus scan for every vocabulary merge.
"""
from collections import defaultdict
import heapq


def apply_ranked_merges(raw, ranks):
    """Canonical best-rank adjacent merges, with leftmost tie-breaking."""
    ids = list(raw)
    if len(ids) < 2 or not ranks:
        return ids
    previous = list(range(-1, len(ids)-1))
    following = list(range(1, len(ids))) + [-1]
    live = [True] * len(ids)
    heap = []

    def enqueue(i):
        if i == -1 or not live[i]:
            return
        j = following[i]
        if j != -1 and live[j]:
            rank = ranks.get((ids[i], ids[j]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))

    for i in range(len(ids)-1):
        enqueue(i)
    while heap:
        rank, i = heapq.heappop(heap)
        j = following[i]
        if (not live[i] or j == -1 or not live[j]
                or ranks.get((ids[i], ids[j])) != rank):
            continue
        p, k = previous[i], following[j]
        ids[i] = rank
        live[j] = False
        following[i] = k
        if k != -1:
            previous[k] = i
        enqueue(p)
        enqueue(i)
    result = []
    i = 0
    while i != -1:
        if live[i]:
            result.append(ids[i])
        i = following[i]
    return result


def train_byte_merges(documents, vocab_size, min_frequency, *, first_merge=262,
                      sample_bytes=2_000_000):
    """Learn byte BPE merges from a bounded representative source sample.

    A bounded, evenly allocated per-document sample keeps training feasible
    on CPU with RAM limits; all downstream language training may still see
    the complete corpus. It is a vocabulary sample, not a model-training holdout.
    """
    texts = [str(text) for text in documents if str(text).strip()]
    if not texts:
        raise ValueError("Cannot train tokenizer on an empty corpus")
    if sample_bytes < 1024:
        raise ValueError("Sample budget is too small")
    budget = max(1, sample_bytes // len(texts))
    chunks = [list(text.encode("utf-8")[:budget]) for text in texts]
    token, previous, following, live = [], [], [], []
    for chunk in chunks:
        offset = len(token)
        token.extend(chunk)
        previous.extend((offset + j - 1 if j else -1) for j in range(len(chunk)))
        following.extend((offset + j + 1 if j + 1 < len(chunk) else -1)
                         for j in range(len(chunk)))
        live.extend([True] * len(chunk))
    occurrences = defaultdict(set)
    for i, j in enumerate(following):
        if j != -1:
            occurrences[(token[i], token[j])].add(i)
    heap = [(-len(indices), pair) for pair, indices in occurrences.items()]
    heapq.heapify(heap)
    merges = []

    while heap and first_merge + len(merges) < vocab_size:
        best = None
        while heap:
            negative, pair = heapq.heappop(heap)
            if -negative == len(occurrences[pair]) and -negative >= min_frequency:
                best = pair
                break
        if best is None:
            break
        new_token = first_merge + len(merges)
        changed = set()

        def discard(i):
            if i == -1 or not live[i]:
                return
            j = following[i]
            if j != -1 and live[j]:
                pair = (token[i], token[j])
                occurrences[pair].discard(i)
                changed.add(pair)

        def add(i):
            if i == -1 or not live[i]:
                return
            j = following[i]
            if j != -1 and live[j]:
                pair = (token[i], token[j])
                occurrences[pair].add(i)
                changed.add(pair)

        for i in sorted(occurrences[best]):
            if not live[i]:
                continue
            j = following[i]
            if j == -1 or not live[j] or (token[i], token[j]) != best:
                continue
            p, k = previous[i], following[j]
            discard(p)
            discard(i)
            discard(j)
            token[i] = new_token
            live[j] = False
            following[i] = k
            if k != -1:
                previous[k] = i
            add(p)
            add(i)

        merges.append(best)
        for pair in changed:
            if occurrences[pair]:
                heapq.heappush(heap, (-len(occurrences[pair]), pair))
        # A removed pair is retained as an empty set so stale heap entries
        # can be checked without treating it as a valid candidate.
    return merges
