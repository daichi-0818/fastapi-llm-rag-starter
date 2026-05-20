from app.services.embeddings import FakeEmbeddings


def test_fake_embeddings_dimensions() -> None:
    emb = FakeEmbeddings()
    vecs = emb.embed(["hello", "world"])
    assert len(vecs) == 2
    assert all(len(v) == emb.dimension for v in vecs)


def test_fake_embeddings_deterministic() -> None:
    emb = FakeEmbeddings()
    a = emb.embed(["same text"])[0]
    b = emb.embed(["same text"])[0]
    assert a == b


def test_fake_embeddings_distinguish_inputs() -> None:
    emb = FakeEmbeddings()
    a = emb.embed(["alpha"])[0]
    b = emb.embed(["beta"])[0]
    assert a != b
