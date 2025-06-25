import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import io


def preprocess_image(image, preprocessing_option):
    """
    Apply preprocessing to improve OCR accuracy
    """
    # Convert PIL image to OpenCV format
    img_array = np.array(image)

    # Handle different image formats and data types
    if img_array.dtype == bool:
        img_array = img_array.astype(np.uint8) * 255
    elif img_array.dtype == np.float32 or img_array.dtype == np.float64:
        img_array = (img_array * 255).astype(np.uint8)
    elif img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    # Convert to grayscale if it's a color image
    if len(img_array.shape) == 3 and img_array.shape[2] in [3, 4]:
        if img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        else:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    if preprocessing_option == "None":
        return img_array
    elif preprocessing_option == "Gaussian Blur":
        return cv2.GaussianBlur(img_array, (5, 5), 0)
    elif preprocessing_option == "Threshold":
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img_array
    elif preprocessing_option == "Morphological Operations":
        kernel = np.ones((2, 2), np.uint8)
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
        return img_array
    elif preprocessing_option == "Denoise + Threshold":
        img_array = cv2.fastNlMeansDenoising(img_array)
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img_array

    return img_array


def perform_ocr(image, config_options):
    """
    Perform OCR on the image using Tesseract
    """
    try:
        text = pytesseract.image_to_string(image, config=config_options)

        data = pytesseract.image_to_data(image, config=config_options, output_type=pytesseract.Output.DICT)

        return text, data
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return None, None

def draw_bounding_boxes(image, data):
    """
    Draw bounding boxes around detected text
    """
    if isinstance(image, np.ndarray):
        img_with_boxes = image.copy()
    else:
        img_with_boxes = np.array(image)

    if img_with_boxes.dtype == bool:
        img_with_boxes = img_with_boxes.astype(np.uint8) * 255
    elif img_with_boxes.dtype != np.uint8:
        img_with_boxes = img_with_boxes.astype(np.uint8)

    if len(img_with_boxes.shape) == 2:
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_GRAY2RGB)

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 30:  # Only show confident detections
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            img_with_boxes = cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_with_boxes

def main():
    st.set_page_config(page_title="OCR Prototype", layout="wide")

    st.title("🔍 Simple OCR Prototype")
    st.markdown("Upload an image to extract text using Tesseract OCR")

    st.sidebar.header("Configuration")

    preprocessing_option = st.sidebar.selectbox(
        "Image Preprocessing",
        ["None", "Gaussian Blur", "Threshold", "Morphological Operations", "Denoise + Threshold"]
    )

    st.sidebar.subheader("Tesseract Settings")
    page_segmentation = st.sidebar.selectbox(
        "Page Segmentation Mode (PSM)",
        [
            "3 - Fully automatic page segmentation (default)",
            "6 - Uniform block of text",
            "7 - Single text line",
            "8 - Single word",
            "11 - Sparse text",
            "13 - Raw line (treat as single text line)"
        ]
    )

    psm_value = page_segmentation.split(" - ")[0]

    oem = st.sidebar.selectbox(
        "OCR Engine Mode (OEM)",
        [
            "3 - Default (Legacy + LSTM)",
            "0 - Legacy engine only",
            "1 - Neural nets LSTM only",
            "2 - Legacy + LSTM engines"
        ]
    )

    oem_value = oem.split(" - ")[0]

    language = st.sidebar.selectbox(
        "Language",
        ["eng", "fra", "deu", "spa", "chi_sim", "jpn"]
    )

    config_options = f'--oem {oem_value} --psm {psm_value} -l {language}'

    show_bounding_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)

    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload an image containing text for OCR processing"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)

        processed_image = preprocess_image(image, preprocessing_option)

        with col2:
            st.subheader(f"Processed Image ({preprocessing_option})")
            st.image(processed_image, caption="Processed Image", use_container_width=True, clamp=True)

        with st.spinner("Performing OCR..."):
            extracted_text, ocr_data = perform_ocr(processed_image, config_options)

        if extracted_text is not None:
            # Display results
            st.subheader("📝 Extracted Text")
            if extracted_text.strip():
                st.text_area("OCR Result", extracted_text, height=200)

                if ocr_data:
                    confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        st.metric("Average Confidence", f"{avg_confidence:.1f}%")

                if show_bounding_boxes and ocr_data:
                    st.subheader("🎯 Text Detection")
                    img_with_boxes = draw_bounding_boxes(processed_image, ocr_data)
                    st.image(img_with_boxes, caption="Detected Text Regions", use_container_width=True)

                st.download_button(
                    label="📥 Download Extracted Text",
                    data=extracted_text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No text was detected in the image. Try adjusting the preprocessing options or image quality.")

        with st.expander("ℹ️ Processing Details"):
            st.write(f"**Preprocessing:** {preprocessing_option}")
            st.write(f"**Tesseract Config:** {config_options}")
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")

if __name__ == "__main__":
    main()
