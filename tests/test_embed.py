import json
import numpy as np
import pytest
from vemb.embed import (
    CACHE_VERSION,
    MODEL,
    cosine_similarity,
    guess_mime,
    load_cache,
    save_cache,
)


def test_cosine_identical():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_opposite():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_zero_vector():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_guess_mime_jpg():
    assert guess_mime("photo.jpg") == "image/jpeg"


def test_guess_mime_jpeg():
    assert guess_mime("photo.jpeg") == "image/jpeg"


def test_guess_mime_png():
    assert guess_mime("photo.png") == "image/png"


def test_guess_mime_mp3():
    assert guess_mime("clip.mp3") == "audio/mpeg"


def test_guess_mime_mp4():
    assert guess_mime("clip.mp4") == "video/mp4"


def test_guess_mime_pdf():
    assert guess_mime("doc.pdf") == "application/pdf"


def test_guess_mime_unsupported():
    with pytest.raises(SystemExit):
        guess_mime("data.csv")


def test_cache_empty_when_no_files(tmp_path):
    keys, matrix = load_cache(tmp_path, 3072)
    assert keys == {}
    assert matrix is None


def test_cache_roundtrip(tmp_path):
    keys = {"a:1:1": 0, "b:2:2": 1, "c:3:3": 2}
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    save_cache(tmp_path, 3, keys, matrix)
    loaded_keys, loaded_matrix = load_cache(tmp_path, 3)
    assert loaded_keys == keys
    assert loaded_matrix.dtype == np.float32
    assert loaded_matrix.shape == (3, 3)
    assert loaded_matrix.tolist() == matrix.tolist()


def test_cache_dim_mismatch_rejects(tmp_path):
    keys = {"x:1:1": 0}
    matrix = np.array([[1, 0, 0]], dtype=np.float32)
    save_cache(tmp_path, 3, keys, matrix)
    loaded_keys, loaded_matrix = load_cache(tmp_path, 1024)
    assert loaded_keys == {}
    assert loaded_matrix is None


def test_cache_model_mismatch_rejects(tmp_path):
    keys = {"x:1:1": 0}
    matrix = np.array([[1, 0, 0]], dtype=np.float32)
    save_cache(tmp_path, 3, keys, matrix)
    manifest_path = tmp_path / ".vemb" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"] = "different-model"
    manifest_path.write_text(json.dumps(manifest))
    loaded_keys, loaded_matrix = load_cache(tmp_path, 3)
    assert loaded_keys == {}
    assert loaded_matrix is None


def test_legacy_cache_migration(tmp_path):
    cache_dir = tmp_path / ".vemb"
    cache_dir.mkdir()
    legacy = {
        "version": 1,
        "model": MODEL,
        "dim": 2,
        "entries": {
            "a:1:1": {"file": "/x/a", "values": [3.0, 4.0]},
            "b:2:2": {"file": "/x/b", "values": [0.0, 1.0]},
        },
    }
    (cache_dir / "cache.json").write_text(json.dumps(legacy))
    keys, matrix = load_cache(tmp_path, 2)
    assert set(keys.keys()) == {"a:1:1", "b:2:2"}
    assert matrix.shape == (2, 2)
    row_a = matrix[keys["a:1:1"]]
    assert np.linalg.norm(row_a) == pytest.approx(1.0, abs=1e-5)
    assert row_a.tolist() == pytest.approx([0.6, 0.8], abs=1e-5)
    assert not (cache_dir / "cache.json").exists()
    assert (cache_dir / "manifest.json").exists()
    assert (cache_dir / "vectors.npy").exists()
