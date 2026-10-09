import numpy as np

from server.backend.models_ai.vector_index import FaissVectorIndex


class FakeFaissIndex:
    def __init__(self):
        self.vectors = []

    def add(self, vectors):
        self.vectors.append(vectors.copy())


def test_faiss_index_does_not_duplicate_vectors_in_python_memory(tmp_path, monkeypatch):
    index = FaissVectorIndex(dimension=3, index_dir=tmp_path)
    faiss_index = FakeFaissIndex()
    monkeypatch.setattr(index, "_has_faiss", True)
    monkeypatch.setattr(index, "_index", faiss_index)

    index.add(np.array([1, 0, 0], dtype=np.float32), "frame-1", "CAM-01", 1.0)

    assert len(faiss_index.vectors) == 1
    assert index._vectors == []


def test_numpy_fallback_keeps_vectors_required_for_search(tmp_path, monkeypatch):
    index = FaissVectorIndex(dimension=3, index_dir=tmp_path)
    monkeypatch.setattr(index, "_has_faiss", False)
    monkeypatch.setattr(index, "_index", None)

    index.add(np.array([1, 0, 0], dtype=np.float32), "frame-1", "CAM-01", 1.0)

    assert len(index._vectors) == 1
    assert np.array_equal(index._vectors[0], np.array([1, 0, 0], dtype=np.float32))
