# OCR-NLP

This project combines Optical Character Recognition (OCR) with Natural Language Processing (NLP) to extract and analyze text from images using Python.

## Features

- Convert images to text using Tesseract OCR
- Clean and preprocess the extracted text
- Perform basic NLP tasks: 
  - Tokenization
  - Stopword removal
  - Lemmatization
  - Named Entity Recognition (NER)

## Tech Stack

- Python
- Tesseract OCR (`pytesseract`)
- `nltk`, `spacy`
- `opencv-python`

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/akshat172003/OCR-NLP.git
cd OCR-NLP
````

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR and ensure it is available in your system's PATH.

4. Run the scripts:

```bash
python ocr_script.py
python nlp_script.py
```

## Project Structure

* `images/` — Input image files
* `ocr_script.py` — Script for extracting text from images
* `nlp_script.py` — Script for processing and analyzing text
