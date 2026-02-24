import pdfplumber
import os

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

if __name__ == "__main__":
    # Test extraction
    test_pdf = "../Documentation/Unleashing 10000 Word Generation From Long Context.pdf"
    if os.path.exists(test_pdf):
        content = extract_text_from_pdf(test_pdf)
        print(f"Extracted {len(content)} characters from test PDF.")
    else:
        print("Test PDF not found.")
