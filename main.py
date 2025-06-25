import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    st.warning("EasyOCR not installed. Some features will be disabled.")

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not installed. Clustering features will be disabled.")


# Configure page
st.set_page_config(page_title="Advanced OCR Prototype", layout="wide")


def visualize_confidence_heatmap(image, data):
    """
    Create a confidence heatmap overlay
    """
    heatmap = np.zeros_like(image, dtype=np.float32)

    for i in range(len(data['text'])):
        conf = int(data['conf'][i])
        if conf > 0:
            x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            # Normalize confidence to 0-1
            conf_normalized = conf / 100.0
            heatmap[y:y+h, x:x+w] = conf_normalized

    # Create color-coded heatmap
    heatmap_colored = plt.cm.RdYlGn(heatmap)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

    return heatmap_colored


def advanced_preprocessing(image, options):
    """
    Apply advanced preprocessing techniques
    """
    img_array = np.array(image)

    # Handle data type conversion
    if img_array.dtype == bool:
        img_array = img_array.astype(np.uint8) * 255
    elif img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()

    processed = gray.copy()

    # Apply selected preprocessing steps
    if "Adaptive Threshold" in options:
        processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

    if "Contrast Enhancement" in options:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed = clahe.apply(processed)

    if "Deskewing" in options:
        processed = deskew_image(processed)

    if "Noise Removal" in options:
        # Remove small noise using morphological operations
        kernel = np.ones((2,2), np.uint8)
        try:
            # Try morphological opening (erosion followed by dilation)
            processed = cv2.erode(processed, kernel, iterations=1)
            processed = cv2.dilate(processed, kernel, iterations=1)
            # Then morphological closing (dilation followed by erosion)
            processed = cv2.dilate(processed, kernel, iterations=1)
            processed = cv2.erode(processed, kernel, iterations=1)
        except AttributeError:
            # Fallback to simple median blur if morphological ops fail
            processed = cv2.medianBlur(processed, 3)

    if "Edge Enhancement" in options:
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(processed, -1, kernel)

    if "Scale Up" in options:
        # Scale up by 2x for better OCR
        processed = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return processed


def deskew_image(image):
    """
    Automatically deskew image using Hough transform
    """
    # Find edges
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Find lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

    if lines is not None:
        angles = []
        for rho, theta in lines[:20]:  # Use first 20 lines
            angle = theta * 180 / np.pi
            if angle < 45:
                angles.append(angle)
            elif angle > 135:
                angles.append(angle - 180)

        if angles:
            median_angle = np.median(angles)

            # Rotate image to correct skew
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated

    return image


def smart_text_detection(image):
    """
    Use ML-based text detection to find text regions
    """
    if not EASYOCR_AVAILABLE:
        st.warning("EasyOCR not available. Install with: pip install easyocr")
        return [], [], []

    try:
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)

        # Detect text with bounding boxes
        results = reader.readtext(image, detail=1)

        # Extract bounding boxes and confidences
        boxes = []
        texts = []
        confidences = []

        for (bbox, text, conf) in results:
            # Convert bbox to standard format
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
            texts.append(text)
            confidences.append(conf)

        return boxes, texts, confidences
    except Exception as e:
        st.warning(f"EasyOCR error: {e}")
        return [], [], []


def cluster_text_regions(boxes, texts, confidences):
    """
    Use clustering to group nearby text regions (useful for tables/forms)
    """
    if not SKLEARN_AVAILABLE:
        st.warning("Clustering unavailable. Install with: pip install scikit-learn")
        return {}

    if not boxes:
        return {}

    # Convert boxes to center points for clustering
    centers = []
    for box in boxes:
        x, y, w, h = box
        centers.append([x + w/2, y + h/2])

    centers = np.array(centers)

    # Use DBSCAN clustering
    clustering = DBSCAN(eps=50, min_samples=2).fit(centers)

    # Group boxes by cluster
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            'box': boxes[i],
            'text': texts[i],
            'confidence': confidences[i]
        })

    return clusters


def enhance_low_confidence_regions(image, tesseract_data, threshold=60):
    """
    Apply additional processing to low-confidence regions
    """
    enhanced_results = []

    for i in range(len(tesseract_data['text'])):
        conf = int(tesseract_data['conf'][i])
        text = tesseract_data['text'][i].strip()

        if conf > 0 and conf < threshold and text:
            # Extract the region
            x, y, w, h = (tesseract_data['left'][i], tesseract_data['top'][i], 
                         tesseract_data['width'][i], tesseract_data['height'][i])

            # Add padding
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)

            region = image[y:y+h, x:x+w]

            if region.size > 0:
                # Apply aggressive enhancement to this region
                enhanced_region = enhance_region(region)

                # Re-run OCR on enhanced region
                enhanced_text = pytesseract.image_to_string(enhanced_region, config='--psm 8')

                enhanced_results.append({
                    'original_text': text,
                    'original_conf': conf,
                    'enhanced_text': enhanced_text.strip(),
                    'bbox': (x, y, w, h)
                })

    return enhanced_results


def enhance_region(region):
    """
    Apply intensive enhancement to a specific region
    """
    # Scale up
    enhanced = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Denoise
    enhanced = cv2.fastNlMeansDenoising(enhanced)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    enhanced = clahe.apply(enhanced)

    # Adaptive threshold
    enhanced = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)

    # Morphological operations
    kernel = np.ones((2,2), np.uint8)
    try:
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    except AttributeError:
        # Fallback if morphological operations aren't available
        enhanced = cv2.dilate(enhanced, kernel, iterations=1)
        enhanced = cv2.erode(enhanced, kernel, iterations=1)

    return enhanced


def draw_bounding_boxes(image, data, box_type="tesseract"):
    """
    Draw bounding boxes around detected text
    """
    # Ensure image is in the right format
    if isinstance(image, np.ndarray):
        img_with_boxes = image.copy()
    else:
        img_with_boxes = np.array(image)

    # Handle data type conversion
    if img_with_boxes.dtype == bool:
        img_with_boxes = img_with_boxes.astype(np.uint8) * 255
    elif img_with_boxes.dtype != np.uint8:
        img_with_boxes = img_with_boxes.astype(np.uint8)

    # Convert grayscale to RGB for colored bounding boxes
    if len(img_with_boxes.shape) == 2:
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_GRAY2RGB)

    if box_type == "tesseract":
        # Draw Tesseract bounding boxes
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:  # Only show confident detections
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                confidence = int(data['conf'][i])

                # Color code by confidence: Red (low) -> Yellow (medium) -> Green (high)
                if confidence < 50:
                    color = (255, 0, 0)  # Red
                elif confidence < 75:
                    color = (255, 255, 0)  # Yellow
                else:
                    color = (0, 255, 0)  # Green

                img_with_boxes = cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)

                # Add confidence text
                cv2.putText(img_with_boxes, f"{confidence}%", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    elif box_type == "easyocr":
        # Draw EasyOCR bounding boxes
        boxes, texts, confidences = data
        for i, (box, text, conf) in enumerate(zip(boxes, texts, confidences)):
            x, y, w, h = box
            confidence = int(conf * 100)  # EasyOCR confidence is 0-1

            # Color code by confidence
            if confidence < 50:
                color = (255, 0, 0)  # Red
            elif confidence < 75:
                color = (255, 255, 0)  # Yellow  
            else:
                color = (0, 255, 0)  # Green

            img_with_boxes = cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)

            # Add confidence and text preview
            preview_text = text[:10] + "..." if len(text) > 10 else text
            cv2.putText(img_with_boxes, f"{confidence}%: {preview_text}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img_with_boxes
    """
    Create a confidence heatmap overlay
    """
    heatmap = np.zeros_like(image, dtype=np.float32)

    for i in range(len(data['text'])):
        conf = int(data['conf'][i])
        if conf > 0:
            x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            # Normalize confidence to 0-1
            conf_normalized = conf / 100.0
            heatmap[y:y+h, x:x+w] = conf_normalized

    # Create color-coded heatmap
    heatmap_colored = plt.cm.RdYlGn(heatmap)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

    return heatmap_colored


def main():
    st.title("🚀 Advanced OCR with ML Enhancement")
    st.markdown("Leverage machine learning and advanced preprocessing for better OCR accuracy")

    # Sidebar configuration
    st.sidebar.header("Advanced Configuration")

    # OCR Engine Selection
    ocr_engine = st.sidebar.selectbox(
        "OCR Engine",
        ["Tesseract", "EasyOCR", "Hybrid (Both)"]
    )

    # Preprocessing options
    st.sidebar.subheader("Advanced Preprocessing")
    preprocessing_options = st.sidebar.multiselect(
        "Select preprocessing steps:",
        ["Adaptive Threshold", "Contrast Enhancement", "Deskewing", 
         "Noise Removal", "Edge Enhancement", "Scale Up"],
        default=["Contrast Enhancement", "Noise Removal"]
    )

    # Enhancement options
    st.sidebar.subheader("ML Enhancement")
    use_smart_detection = st.sidebar.checkbox("Smart Text Detection (EasyOCR)", value=True)
    enhance_low_conf = st.sidebar.checkbox("Enhance Low Confidence Regions", value=True)
    confidence_threshold = st.sidebar.slider("Enhancement Threshold", 0, 100, 60)

    # Visualization options
    show_confidence_heatmap = st.sidebar.checkbox("Show Confidence Heatmap", value=True)
    show_bounding_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
    show_clusters = st.sidebar.checkbox("Show Text Clusters", value=False)

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)

        # Apply advanced preprocessing
        with st.spinner("Applying advanced preprocessing..."):
            processed_image = advanced_preprocessing(image, preprocessing_options)

        with col2:
            st.subheader("🔧 Processed Image")
            st.image(processed_image, use_container_width=True, clamp=True)

        # Perform OCR with selected engine
        st.subheader("🔍 OCR Results")

        if ocr_engine in ["Tesseract", "Hybrid (Both)"]:
            with st.spinner("Running Tesseract OCR..."):
                tesseract_text = pytesseract.image_to_string(processed_image)
                tesseract_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)

            # Calculate average confidence
            confidences = [int(conf) for conf in tesseract_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            st.write("**Tesseract Results:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Confidence", f"{avg_confidence:.1f}%")
            with col2:
                st.metric("Words Detected", len([t for t in tesseract_data['text'] if t.strip()]))

            st.text_area("Extracted Text (Tesseract)", tesseract_text, height=150)

        if ocr_engine in ["EasyOCR", "Hybrid (Both)"] and use_smart_detection:
            with st.spinner("Running EasyOCR..."):
                easyocr_boxes, easyocr_texts, easyocr_confs = smart_text_detection(processed_image)

            if easyocr_texts:
                st.write("**EasyOCR Results:**")
                avg_easyocr_conf = sum(easyocr_confs) / len(easyocr_confs) if easyocr_confs else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Confidence", f"{avg_easyocr_conf:.1f}%")
                with col2:
                    st.metric("Text Regions", len(easyocr_texts))

                combined_text = " ".join(easyocr_texts)
                st.text_area("Extracted Text (EasyOCR)", combined_text, height=150)

        # Enhanced processing for low confidence regions
        if enhance_low_conf and ocr_engine in ["Tesseract", "Hybrid (Both)"]:
            with st.spinner("Enhancing low confidence regions..."):
                enhanced_results = enhance_low_confidence_regions(
                    processed_image, tesseract_data, confidence_threshold
                )

            if enhanced_results:
                st.subheader("🎯 Enhanced Low Confidence Regions")
                for i, result in enumerate(enhanced_results):
                    with st.expander(f"Region {i+1}: '{result['original_text']}' → '{result['enhanced_text']}'"):
                        st.write(f"**Original:** {result['original_text']} (confidence: {result['original_conf']}%)")
                        st.write(f"**Enhanced:** {result['enhanced_text']}")

        # Visualizations
        st.subheader("📊 Advanced Visualizations")

        # Bounding boxes visualization
        if show_bounding_boxes:
            viz_cols = st.columns(2)

            # Tesseract bounding boxes
            if 'tesseract_data' in locals():
                with viz_cols[0]:
                    st.write("**Tesseract Word Detection**")
                    tesseract_boxes_img = draw_bounding_boxes(processed_image, tesseract_data, "tesseract")
                    st.image(tesseract_boxes_img, caption="Color: Red=Low, Yellow=Medium, Green=High Confidence", use_container_width=True)

            # EasyOCR bounding boxes
            if use_smart_detection and 'easyocr_boxes' in locals() and easyocr_boxes:
                with viz_cols[1]:
                    st.write("**EasyOCR Text Detection**")
                    easyocr_data = (easyocr_boxes, easyocr_texts, easyocr_confs)
                    easyocr_boxes_img = draw_bounding_boxes(processed_image, easyocr_data, "easyocr")
                    st.image(easyocr_boxes_img, caption="EasyOCR Detected Regions", use_container_width=True)

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            if show_confidence_heatmap and 'tesseract_data' in locals():
                st.write("**Confidence Heatmap**")
                heatmap = visualize_confidence_heatmap(processed_image, tesseract_data)
                st.image(heatmap, caption="Red=Low Confidence, Green=High Confidence", use_container_width=True)

        with viz_col2:
            if show_clusters and use_smart_detection and 'easyocr_boxes' in locals():
                st.write("**Text Region Clusters**")
                if easyocr_boxes:
                    clusters = cluster_text_regions(easyocr_boxes, easyocr_texts, easyocr_confs)
                    st.write(f"Found {len(clusters)} text clusters")

                    # Visualize clusters
                    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                    ax.imshow(processed_image, cmap='gray')

                    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
                    for cluster_id, color in zip(clusters.keys(), colors):
                        if cluster_id != -1:  # -1 is noise in DBSCAN
                            for item in clusters[cluster_id]:
                                x, y, w, h = item['box']
                                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                                       edgecolor=color, facecolor='none')
                                ax.add_patch(rect)

                    ax.set_title("Text Region Clusters")
                    ax.axis('off')
                    st.pyplot(fig)

        # Performance summary
        st.subheader("📈 Performance Summary")

        summary_data = []
        if 'avg_confidence' in locals():
            summary_data.append(["Tesseract Avg Confidence", f"{avg_confidence:.1f}%"])
        if 'avg_easyocr_conf' in locals():
            summary_data.append(["EasyOCR Avg Confidence", f"{avg_easyocr_conf:.1f}%"])
        if 'enhanced_results' in locals():
            summary_data.append(["Enhanced Regions", len(enhanced_results)])

        if summary_data:
            import pandas as pd
            df = pd.DataFrame(summary_data, columns=["Metric", "Value"])
            st.table(df)


if __name__ == "__main__":
    main()
