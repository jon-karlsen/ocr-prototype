import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import json
import pandas as pd
from datetime import datetime

from data_models import DetectedLabelValuePair
from layout_learner import LayoutLearner
from label_extractor import DynamicLabelExtractor
from correction_interface import manual_correction_interface
from visualization import visualize_detected_pairs, create_analysis_summary


def main():
    st.set_page_config(page_title="Smart Document Extractor", layout="wide")
    st.title("🧠 Smart Document Extractor with Learning")
    st.markdown("Extract, correct, and learn from document layouts")

    # Initialize learner
    if 'learner' not in st.session_state:
        st.session_state.learner = LayoutLearner()

    # Document type selection
    document_type = st.selectbox(
        "Document Type",
        ["Invoice", "Receipt", "Form", "Contract", "Other"],
        help="Select document type to use specific learned patterns"
    )

    # Configuration
    st.sidebar.header("⚙️ Settings")
    confidence_threshold = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.4, 0.1)
    show_bounding_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)
    enable_learning = st.sidebar.checkbox("Enable ML Corrections", True)

    # Extraction method selection
    extraction_methods = st.sidebar.multiselect(
        "Extraction Methods",
        ["Pattern-based", "Spatial Analysis", "NLP (spaCy)", "Table Detection"],
        default=["Pattern-based", "Spatial Analysis", "Table Detection"]
    )

    # Model statistics
    st.sidebar.subheader("📊 Learning Statistics")
    corrections = st.session_state.learner.get_corrections(document_type)
    st.sidebar.metric("Total Corrections", len(corrections))

    if corrections:
        recent_corrections = [c for c in corrections if 
                            (datetime.now() - c.timestamp).days < 7]
        st.sidebar.metric("This Week", len(recent_corrections))

        # Show learning progress
        if len(corrections) >= 10:
            st.sidebar.success("✅ Model trained!")
        else:
            remaining = 10 - len(corrections)
            st.sidebar.info(f"📝 Need {remaining} more corrections to train")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a document image", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Source Document")
            st.image(image, use_container_width=True)

        # Process image
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Perform OCR and extraction
        with st.spinner("Analyzing document..."):
            ocr_text = pytesseract.image_to_string(gray)
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

            # Extract with learning
            learner = st.session_state.learner if enable_learning else None
            extractor = DynamicLabelExtractor(learner)
            detected_pairs = extractor.extract_dynamic_labels(ocr_text, ocr_data, document_type)

            # Filter by confidence
            filtered_pairs = [p for p in detected_pairs if p.confidence >= confidence_threshold]

        with col2:
            st.subheader("🎯 Detected Pairs")
            if show_bounding_boxes and any(p.label_bbox or p.value_bbox for p in filtered_pairs):
                visualized_img = visualize_detected_pairs(gray, filtered_pairs)
                st.image(visualized_img, caption="L=Label, V=Value", use_container_width=True)
            else:
                st.image(gray, use_container_width=True)

        # Display extracted pairs
        st.subheader("📋 Extracted Data")

        if filtered_pairs:
            # Show initial extraction results
            df_data = []
            for pair in filtered_pairs:
                df_data.append({
                    'Label': pair.label,
                    'Value': pair.value,
                    'Confidence': f"{pair.confidence:.1%}",
                    'Method': pair.pattern_type,
                    'Spatial Rel.': pair.spatial_relationship or '-'
                })

            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)

            # Manual correction interface
            st.markdown("---")
            corrected_pairs = manual_correction_interface(
                filtered_pairs, 
                st.session_state.learner, 
                document_type
            )

            # Display final results if corrections were made
            if corrected_pairs and corrected_pairs != filtered_pairs:
                st.subheader("✅ Final Extracted Data")

                final_df = pd.DataFrame([
                    {
                        'Label': p.label, 
                        'Value': p.value, 
                        'Confidence': f"{p.confidence:.1%}",
                        'Source': 'Manual' if 'manual' in p.pattern_type else 'Auto'
                    }
                    for p in corrected_pairs
                ])

                st.dataframe(final_df, use_container_width=True)

            # Export options
            st.subheader("📤 Export Extracted Data")

            export_pairs = corrected_pairs if 'corrected_pairs' in locals() and corrected_pairs else filtered_pairs

            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                # JSON export
                json_data = {pair.label: pair.value for pair in export_pairs}
                st.download_button(
                    "📥 Download JSON",
                    json.dumps(json_data, indent=2),
                    file_name=f"extracted_{document_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            with export_col2:
                # CSV export
                csv_data = pd.DataFrame([
                    {'Label': p.label, 'Value': p.value} 
                    for p in export_pairs
                ]).to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    file_name=f"extracted_{document_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with export_col3:
                # Key-value text export
                kv_text = "\n".join([f"{p.label}: {p.value}" for p in export_pairs])
                st.download_button(
                    "📥 Download TXT",
                    kv_text,
                    file_name=f"extracted_{document_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        else:
            st.info("No label-value pairs detected. Try adjusting the confidence threshold or uploading a different document.")

        # Analysis summary
        with st.expander("📊 Extraction Analysis"):
            analysis = create_analysis_summary(detected_pairs, filtered_pairs)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Detected", analysis['total_detected'])
            with col2:
                st.metric("Above Threshold", analysis['total_filtered'])
            with col3:
                st.metric("Avg Confidence", f"{analysis['confidence_stats']['avg']:.1%}")

            if analysis['method_counts']:
                st.write("**Extraction methods used:**")
                for method, count in analysis['method_counts'].items():
                    st.write(f"- {method}: {count} pairs")

        # Raw OCR text
        with st.expander("📄 Raw OCR Text"):
            st.text_area("Full extracted text", ocr_text, height=200)

    else:
        # Landing page content
        st.markdown("""
        ## 🚀 How to Use

        1. **Upload** a document image (PNG, JPG, TIFF, etc.)
        2. **Review** automatically extracted label-value pairs
        3. **Correct** any errors using the manual correction interface
        4. **Save** corrections to train the AI model
        5. **Export** clean, structured data

        ## 🧠 Learning Features

        - **Pattern Recognition**: Learns from your corrections
        - **Document Types**: Separate models for different document types
        - **Spatial Analysis**: Understands layout relationships
        - **Continuous Improvement**: Gets better with each correction

        ## 📋 Supported Document Types

        - **Invoices**: Vendor, amount, date, line items
        - **Receipts**: Store, total, items, tax
        - **Forms**: Any structured form with labels and fields
        - **Contracts**: Parties, dates, terms
        - **Custom**: Any document with label-value patterns
        """)


if __name__ == "__main__":
    main()
