import os
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
from fuzzywuzzy import fuzz
import cv2
import fitz  # PyMuPDF

# Setează calea către executabilul Tesseract dacă este necesar
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows

# Eliminarea zgomotului
def remove_noise(img_array):
    return cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

# Funcție pentru ajustarea contrastului
def enhance_contrast(img, factor=2.0):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

# Funcție pentru transformarea în tonuri de gri
def convert_to_grayscale(img_array):
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

# Funcție pentru binarizare
def binarize_image(img_array):
    _, img_bin = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_bin

# Funcție pentru mărirea imaginii
def upscale_image(img_array, scale=2):
    width = int(img_array.shape[1] * scale)
    height = int(img_array.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(img_array, dim, interpolation=cv2.INTER_LINEAR)

# Funcție pentru clasificarea documentelor cu similaritate text
def classify_document(text):
    classifications = {
        'contract': ['contract', 'parties', 'agreement', 'terms', 'părți', 'acord', 'termeni'],
        'NDA': ['non-disclosure', 'confidentiality', 'NDA', 'acord de confidențialitate', 'ne-divulgare'],
        'GDPR': ['data protection', 'GDPR', 'personal data', 'protecția datelor', 'date personale'],
        'extract': ['bank statement', 'account summary', 'balance', 'extras de cont', 'sumar de cont', 'sold'],
        'passport': ['passport', 'nationality', 'country of issuance', 'pașaport', 'naționalitate', 'țara de emitere'],
        'birth_certificate': ['birth certificate', 'date of birth', 'place of birth', 'certificat de naștere', 'data nașterii', 'locul nașterii'],
        'IP': ['intellectual property', 'trademark', 'patent', 'proprietate intelectuală', 'marcă înregistrată', 'brevet'],
    }

    highest_similarity = 0
    best_match = 'other'

    for category, keywords in classifications.items():
        for keyword in keywords:
            similarity = fuzz.partial_ratio(keyword.lower(), text.lower())
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = category

    return best_match if highest_similarity > 70 else 'other'  # Pragul de similaritate

# Funcție pentru procesarea unui PDF folosind OCR
def process_pdf_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    detected_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = enhance_contrast(img)
        img_array = np.array(img)
        img_array = remove_noise(img_array)
        img_array = upscale_image(img_array)
        text = pytesseract.image_to_string(img_array)
        detected_text += text + "\n"
    return detected_text

# Funcție pentru procesarea unei imagini
def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = enhance_contrast(img, factor=2.0)
    img_array = np.array(img)
    img_array = remove_noise(img_array)
    img_array = upscale_image(img_array)
    text = pytesseract.image_to_string(img_array)
    detected_text = text
    return detected_text

# Funcție pentru a procesa fișierul pe baza extensiei
def process_file(file_path):
    if file_path.lower().endswith('.pdf'):
        return process_pdf_with_ocr(file_path)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        return process_image(file_path)
    else:
        raise ValueError("Format fișier neacceptat. Acceptăm doar PDF-uri și imagini.")

# Interfața de utilizator cu Streamlit
def main():
    st.title("Document Processor")

    choice = st.selectbox("Selectați opțiunea:", ["Procesați un fișier", "Procesați un folder"])

    if choice == "Procesați un fișier":
        file = st.file_uploader("Încărcați un fișier (imagine sau PDF):", type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'])
        if file is not None:
            file_path = f"temp_{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.read())

            try:
                detected_text = process_file(file_path)
                st.subheader("Text detectat:")
                st.text(detected_text)
                document_type = classify_document(detected_text)
                st.subheader(f"Tipul documentului: {document_type}")
                txt_filename = f"{document_type}_{os.path.splitext(file.name)[0]}.txt"
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write(detected_text)
                st.success(f"Textul a fost salvat în fișierul '{txt_filename}'.")
            except ValueError as ve:
                st.error(f"Eroare: {ve}")
            except Exception as e:
                st.error(f"A apărut o eroare: {e}")
            finally:
                os.remove(file_path)

    elif choice == "Procesați un folder":
        folder_path = st.text_input("Introduceți calea către folder:")
        if folder_path and os.path.isdir(folder_path):
            process_folder(folder_path)
            st.success(f"Procesarea fișierelor din folderul '{folder_path}' a fost finalizată.")
        else:
            st.warning("Introduceți o cale validă către un folder.")

if __name__ == "__main__":
    main()
