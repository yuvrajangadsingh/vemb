import numpy as np
import pytest
from vemb.embed import cosine_similarity, cosine_similarity_batch, guess_mime


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


def test_batch_matches_scalar_for_each_row():
    query = [1.0, 2.0, 3.0]
    matrix = [
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [-1.0, -2.0, -3.0],
        [0.0, 0.0, 0.0],
    ]
    scores = cosine_similarity_batch(query, matrix)
    expected = [cosine_similarity(query, row) for row in matrix]
    assert scores.tolist() == pytest.approx(expected, abs=1e-5)


def test_batch_zero_query():
    query = [0.0, 0.0, 0.0]
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    scores = cosine_similarity_batch(query, matrix)
    assert scores.tolist() == [0.0, 0.0]


def test_batch_shape_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity_batch([1.0, 2.0], [[1.0, 2.0, 3.0]])


def test_batch_numpy_input():
    q = np.array([1.0, 0.0], dtype=np.float32)
    m = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    scores = cosine_similarity_batch(q, m)
    assert scores.tolist() == pytest.approx([1.0, 0.0, -1.0], abs=1e-5)
