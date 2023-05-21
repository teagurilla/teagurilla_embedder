from teagurilla_embedder import Embedder, __version__


def test_version():
    assert __version__ == "0.1.0"


def test_embed_dimensions():
    embedder = Embedder()
    vector = embedder.sentence_into_vector("Hello, world!")
    assert vector.shape == (1024,)
