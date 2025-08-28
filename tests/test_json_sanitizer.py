from src.utils.json_sanitizer import extract_json

def test_extract_json():
    s = "prose...\n{ \"a\": 1, \"b\": [2,3,], }\nmore prose"
    j = extract_json(s)
    assert j["a"] == 1
    assert j["b"] == [2,3]

