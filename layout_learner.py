import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import streamlit as st


from data_models import DetectedLabelValuePair, CorrectionRecord


try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class LayoutLearner:
    """Learns from manual corrections to improve extraction"""

    def __init__(self, db_path: str = "corrections.db"):
        self.db_path = db_path
        self.setup_database()
        self.label_classifier = None
        self.spatial_patterns = {}
        self.document_templates = {}


    def setup_database(self):
        """Initialize SQLite database for storing corrections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_label TEXT,
                corrected_label TEXT,
                original_value TEXT,
                corrected_value TEXT,
                spatial_features TEXT,
                document_type TEXT,
                timestamp TEXT,
                user_action TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_layouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_type TEXT,
                layout_pattern TEXT,
                confidence REAL,
                usage_count INTEGER DEFAULT 1,
                last_updated TEXT
            )
        ''')

        conn.commit()
        conn.close()


    def record_correction(self, correction: CorrectionRecord):
        """Store a correction in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO corrections 
            (original_label, corrected_label, original_value, corrected_value, 
             spatial_features, document_type, timestamp, user_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            correction.original_label,
            correction.corrected_label,
            correction.original_value,
            correction.corrected_value,
            json.dumps(correction.spatial_features),
            correction.document_type,
            correction.timestamp.isoformat(),
            correction.user_action
        ))

        conn.commit()
        conn.close()


    def get_corrections(self, document_type: str = None) -> List[CorrectionRecord]:
        """Retrieve corrections from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if document_type:
            cursor.execute('SELECT * FROM corrections WHERE document_type = ?', (document_type,))
        else:
            cursor.execute('SELECT * FROM corrections')

        corrections = []
        for row in cursor.fetchall():
            corrections.append(CorrectionRecord(
                original_label=row[1],
                corrected_label=row[2],
                original_value=row[3],
                corrected_value=row[4],
                spatial_features=json.loads(row[5]),
                document_type=row[6],
                timestamp=datetime.fromisoformat(row[7]),
                user_action=row[8]
            ))

        conn.close()
        return corrections


    def train_from_corrections(self, document_type: str = None):
        """Train models from accumulated corrections"""
        if not SKLEARN_AVAILABLE:
            st.warning("scikit-learn not available. Install with: pip install scikit-learn")
            return

        corrections = self.get_corrections(document_type)

        if len(corrections) < 10:
            st.warning(f"Need at least 10 corrections to train. Currently have {len(corrections)}")
            return

        # Prepare training data
        features = []
        labels = []

        for correction in corrections:
            if correction.user_action != 'delete':
                # Extract features from spatial relationships
                feature_vector = [
                    correction.spatial_features.get('x_ratio', 0),
                    correction.spatial_features.get('y_ratio', 0),
                    correction.spatial_features.get('width_ratio', 0),
                    correction.spatial_features.get('height_ratio', 0),
                    correction.spatial_features.get('distance_to_label', 0),
                    len(correction.corrected_label),
                    len(correction.corrected_value)
                ]
                features.append(feature_vector)
                labels.append(correction.corrected_label)


        # Train classifier
        if len(set(labels)) > 1:
            self.label_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.label_classifier.fit(features, labels)

            # Save model
            model_path = f"models/label_classifier_{document_type or 'general'}.pkl"
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.label_classifier, model_path)

            st.success(f"Trained model with {len(corrections)} corrections!")


    def predict_label_corrections(self, pairs: List[DetectedLabelValuePair], 
                                document_type: str = None) -> List[DetectedLabelValuePair]:
        """Use trained model to suggest corrections"""
        if not SKLEARN_AVAILABLE:
            return pairs

        if not self.label_classifier:
            model_path = f"models/label_classifier_{document_type or 'general'}.pkl"
            if os.path.exists(model_path):
                self.label_classifier = joblib.load(model_path)
            else:
                return pairs

        corrected_pairs = []
        for pair in pairs:
            if pair.label_bbox and pair.value_bbox:
                # Extract features
                features = self._extract_spatial_features(pair)
                feature_vector = [
                    features.get('x_ratio', 0),
                    features.get('y_ratio', 0),
                    features.get('width_ratio', 0),
                    features.get('height_ratio', 0),
                    features.get('distance_to_label', 0),
                    len(pair.label),
                    len(pair.value)
                ]

                # Predict
                try:
                    predicted_label = self.label_classifier.predict([feature_vector])[0]
                    confidence_boost = 0.1 if predicted_label != pair.label else 0

                    corrected_pair = DetectedLabelValuePair(
                        label=predicted_label,
                        value=pair.value,
                        confidence=min(1.0, pair.confidence + confidence_boost),
                        label_bbox=pair.label_bbox,
                        value_bbox=pair.value_bbox,
                        pattern_type=f"{pair.pattern_type}_ml_corrected",
                        spatial_relationship=pair.spatial_relationship
                    )
                    corrected_pairs.append(corrected_pair)
                except:
                    corrected_pairs.append(pair)
            else:
                corrected_pairs.append(pair)

        return corrected_pairs


    def _extract_spatial_features(self, pair: DetectedLabelValuePair) -> Dict[str, float]:
        """Extract spatial features for ML"""
        if not (pair.label_bbox and pair.value_bbox):
            return {}

        lx, ly, lw, lh = pair.label_bbox
        vx, vy, vw, vh = pair.value_bbox

        return {
            'x_ratio': vx / (lx + 1),  # Relative X position
            'y_ratio': vy / (ly + 1),  # Relative Y position
            'width_ratio': vw / (lw + 1),  # Relative width
            'height_ratio': vh / (lh + 1),  # Relative height
            'distance_to_label': np.sqrt((vx - lx)**2 + (vy - ly)**2),
            'horizontal_gap': vx - (lx + lw),
            'vertical_gap': vy - (ly + lh)
        }
