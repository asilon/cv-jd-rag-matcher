from src.ingest.jd_parser import extract_requirements

def test_extract_requirements():
    jd = "- Must know Python\n- Experience with Docker\nNice to have: Rust"
    reqs = extract_requirements(jd)
    assert "Must know Python" in reqs
    assert "Experience with Docker" in reqs

