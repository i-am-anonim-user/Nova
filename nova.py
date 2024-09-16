import os
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from fuzzywuzzy import fuzz
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained CRNN model
model = load_model('crnn_model.h5')

# Function for preprocessing image for CRNN model
def preprocess_image(img_array):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img_array = cv2.resize(img_array, (128, 32))  # Resize to model input size
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return img_array

# Dummy function to simulate text prediction from CRNN model
def predict_text_from_image(img_array):
    preprocessed_img = preprocess_image(img_array)
    predictions = model.predict(preprocessed_img)
    # Dummy implementation: replace with actual decoding logic
    detected_text = "Detected text"  # This should be replaced with actual decoding
    return detected_text

# Function for processing PDF using OCR with CRNN
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
        detected_text += predict_text_from_image(img_array) + "\n"
    return detected_text

# Function for processing image with CRNN
def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = enhance_contrast(img, factor=2.0)
    img_array = np.array(img)
    img_array = remove_noise(img_array)
    img_array = upscale_image(img_array)
    detected_text = predict_text_from_image(img_array)
    return detected_text

# Function for file processing based on extension
def process_file(file_path):
    if file_path.lower().endswith('.pdf'):
        if pdf_has_text(file_path):
            return extract_text_from_pdf(file_path)
        else:
            return process_pdf_with_ocr(file_path)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        return process_image(file_path)
    else:
        raise ValueError("Format fișier neacceptat. Acceptăm doar PDF-uri și imagini.")

# Streamlit UI
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
