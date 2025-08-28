from src.ingest.chunking import chunk_text

def test_chunking_basic():
    text = "Para1.\n\nPara2 is a bit longer.\n\nPara3."
    chunks = chunk_text(text, max_chars=20)
    assert len(chunks) >= 2

