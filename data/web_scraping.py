import requests
from bs4 import BeautifulSoup
import PyPDF2
import re
from io import BytesIO

def extract_text_from_pdf(pdf_content, url):
    pdf_text = ""
    try:
        with BytesIO(pdf_content) as open_pdf_file:
            reader = PyPDF2.PdfReader(open_pdf_file)
            for page in reader.pages:
                pdf_text += page.extract_text() + " "
    except Exception as e:
        print(f"Failed to extract text from PDF {url}: {e}")
    return pdf_text

def parse_html(url):

    try:
        response = requests.get(url)
        response.raise_for_status()

        if 'application/pdf' in response.headers.get('Content-Type'):
            return extract_text_from_pdf(response.content, url)


        soup = BeautifulSoup(response.text, 'html.parser')

        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)

        return text

    except requests.exceptions.SSLError as ssl_error:
        print(f"SSL error while fetching {url}: {ssl_error}")
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")

    return ''