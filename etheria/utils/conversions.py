from typing import Sequence
import etheria as eth


def data_normalization(data: Sequence[float]) -> Sequence[float]:
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        return [0.0 for _ in data]
    norm = max_val - min_val
    return [(x - min_val) / norm for x in data]

def data_seq_normalization(data: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
    min_val = min(min(sublist) for sublist in data)
    max_val = max(max(sublist) for sublist in data)
    if max_val == min_val:
        return [[0.0 for _ in sublist] for sublist in data]
    norm = max_val - min_val
    return [[(x - min_val) / norm for x in sublist] for sublist in data]

def to_vector(data: Sequence[float]) -> eth.Vector:
    print("Converting to vector:", data)
    return eth.Vector(data)

def to_matrix(data: Sequence[Sequence[float]]) -> Sequence[eth.Vector]:
    return [eth.Vector(sublist) for sublist in data]