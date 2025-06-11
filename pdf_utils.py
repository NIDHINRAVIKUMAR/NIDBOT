from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    :param file_path: Path to the PDF file.
    :return: Extracted text from the PDF.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


