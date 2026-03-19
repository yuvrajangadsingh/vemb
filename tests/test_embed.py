import pytest
from vemb.embed import cosine_similarity, guess_mime


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
