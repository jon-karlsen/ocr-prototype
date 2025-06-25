import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List


from data_models import DetectedLabelValuePair, CorrectionRecord
from layout_learner import LayoutLearner


def manual_correction_interface(pairs: List[DetectedLabelValuePair], 
                              learner: LayoutLearner,
                              document_type: str) -> List[DetectedLabelValuePair]:
    """Interface for manual correction of extracted pairs"""

    st.subheader("✏️ Manual Correction Interface")

    corrected_pairs = []
    corrections_made = []

    # Convert to DataFrame for editing
    df_data = []
    for i, pair in enumerate(pairs):
        df_data.append({
            'ID': i,
            'Label': pair.label,
            'Value': pair.value,
            'Confidence': pair.confidence,
            'Action': 'Keep'
        })

    if df_data:
        df = pd.DataFrame(df_data)

        # Editable interface with better validation
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", disabled=True),
                "Label": st.column_config.TextColumn("Label", required=True),
                "Value": st.column_config.TextColumn("Value", required=True),
                "Confidence": st.column_config.NumberColumn("Confidence", disabled=True, format="%.2f"),
                "Action": st.column_config.SelectboxColumn(
                    "Action",
                    options=["Keep", "Modify", "Delete"],
                    default="Keep",
                    required=True
                )
            },
            num_rows="dynamic"
        )

        # Clean the dataframe to remove completely empty rows
        edited_df = edited_df.dropna(subset=['ID'], how='all')

        # Process corrections
        for _, row in edited_df.iterrows():
            # Handle NaN values safely
            try:
                pair_id = int(row['ID']) if pd.notna(row['ID']) else None
                if pair_id is None or pair_id >= len(pairs):
                    continue
                original_pair = pairs[pair_id]
            except (ValueError, TypeError):
                continue

            # Skip rows with missing essential data
            if pd.isna(row['Label']) or pd.isna(row['Value']) or pd.isna(row['Action']):
                continue

            if row['Action'] == 'Delete':
                if original_pair:
                    # Record deletion
                    correction = CorrectionRecord(
                        original_label=original_pair.label,
                        corrected_label="",
                        original_value=original_pair.value,
                        corrected_value="",
                        spatial_features=learner._extract_spatial_features(original_pair) if original_pair.label_bbox else {},
                        document_type=document_type,
                        timestamp=datetime.now(),
                        user_action='delete'
                    )
                    corrections_made.append(correction)
                continue

            elif row['Action'] == 'Modify':
                # Check if actually modified and data is valid
                if (str(row['Label']).strip() != str(original_pair.label).strip() or 
                    str(row['Value']).strip() != str(original_pair.value).strip()):

                    # Record modification
                    correction = CorrectionRecord(
                        original_label=original_pair.label,
                        corrected_label=str(row['Label']).strip(),
                        original_value=original_pair.value,
                        corrected_value=str(row['Value']).strip(),
                        spatial_features=learner._extract_spatial_features(original_pair) if original_pair.label_bbox else {},
                        document_type=document_type,
                        timestamp=datetime.now(),
                        user_action='modify'
                    )
                    corrections_made.append(correction)

                # Create corrected pair
                corrected_pair = DetectedLabelValuePair(
                    label=str(row['Label']).strip(),
                    value=str(row['Value']).strip(),
                    confidence=float(row['Confidence']) if pd.notna(row['Confidence']) else original_pair.confidence,
                    label_bbox=original_pair.label_bbox,
                    value_bbox=original_pair.value_bbox,
                    pattern_type=f"{original_pair.pattern_type}_manual_corrected",
                    spatial_relationship=original_pair.spatial_relationship
                )
                corrected_pairs.append(corrected_pair)

            else:  # Keep
                corrected_pairs.append(original_pair)

    # Add new pairs interface
    st.subheader("➕ Add New Label-Value Pairs")

    col1, col2 = st.columns(2)
    with col1:
        new_label = st.text_input("New Label")
    with col2:
        new_value = st.text_input("New Value")

    if st.button("Add New Pair") and new_label and new_value:
        new_pair = DetectedLabelValuePair(
            label=new_label,
            value=new_value,
            confidence=1.0,
            pattern_type="manual_added"
        )
        corrected_pairs.append(new_pair)

        # Record addition
        correction = CorrectionRecord(
            original_label="",
            corrected_label=new_label,
            original_value="",
            corrected_value=new_value,
            spatial_features={},
            document_type=document_type,
            timestamp=datetime.now(),
            user_action='add'
        )
        corrections_made.append(correction)

        st.success("Added new pair!")

    # Save corrections
    if corrections_made:
        if st.button("💾 Save Corrections & Learn"):
            for correction in corrections_made:
                learner.record_correction(correction)

            # Retrain model
            learner.train_from_corrections(document_type)

            st.success(f"Saved {len(corrections_made)} corrections!")
            st.rerun()

    return corrected_pairs
