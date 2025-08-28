from pdfminer.high_level import extract_text

def pdf_to_text(path: str) -> str:
    return extract_text(path)

