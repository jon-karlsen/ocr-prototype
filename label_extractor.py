import re
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import streamlit as st


from data_models import DetectedLabelValuePair


try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class DynamicLabelExtractor:
    """Extracts labels and values dynamically from documents"""

    def __init__(self, learner=None):
        self.learner = learner

        # Common label patterns that indicate key-value relationships
        self.label_indicators = [
            r'([A-Za-z\s]+):\s*$',  # "Label:"
            r'([A-Za-z\s]+)\s*:\s*',  # "Label: value"
            r'([A-Za-z\s]+)\s*-\s*$',  # "Label -"
            r'([A-Za-z\s]+)\s*=\s*$',  # "Label ="
            r'([A-Za-z\s]+)\s*\|\s*$',  # "Label |"
        ]

        # Common value patterns
        self.value_patterns = [
            r'\$?[\d,]+\.?\d*',  # Currency/numbers
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Dates
            r'[A-Z0-9][A-Z0-9\-]+',  # IDs/codes
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',  # Emails
            r'\([0-9]{3}\)\s*[0-9]{3}-[0-9]{4}',  # Phone numbers
        ]

        # Load spaCy model if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                st.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None


    def extract_dynamic_labels(self, text: str, ocr_data: Dict, 
                             document_type: str = None) -> List[DetectedLabelValuePair]:
        """Main method to extract labels and values dynamically"""
        pairs = []

        # Method 1: Pattern-based extraction
        pattern_pairs = self._extract_using_patterns(text, ocr_data)
        pairs.extend(pattern_pairs)

        # Method 2: Spatial relationship analysis
        spatial_pairs = self._extract_using_spatial_analysis(ocr_data)
        pairs.extend(spatial_pairs)

        # Method 3: NLP-based extraction (if spaCy available)
        if self.nlp:
            nlp_pairs = self._extract_using_nlp(text, ocr_data)
            pairs.extend(nlp_pairs)

        # Method 4: Table structure detection
        table_pairs = self._extract_from_tables(text, ocr_data)
        pairs.extend(table_pairs)

        # Deduplicate and merge similar pairs
        pairs = self._deduplicate_pairs(pairs)

        # Apply ML corrections if learner is available
        if self.learner:
            pairs = self.learner.predict_label_corrections(pairs, document_type)

        return pairs


    def _extract_using_patterns(self, text: str, ocr_data: Dict) -> List[DetectedLabelValuePair]:
        """Extract using regex patterns for common label-value formats"""
        pairs = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Pattern 1: "Label: Value" on same line
            colon_match = re.match(r'^([A-Za-z\s]+?):\s*(.+)$', line)
            if colon_match:
                label = colon_match.group(1).strip()
                value = colon_match.group(2).strip()

                if self._is_valid_label_value_pair(label, value):
                    pairs.append(DetectedLabelValuePair(
                        label=label,
                        value=value,
                        confidence=0.8,
                        pattern_type="colon_same_line"
                    ))

            # Pattern 2: "Label:" followed by value on next line
            elif line.endswith(':'):
                label = line[:-1].strip()
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and self._is_valid_label_value_pair(label, next_line):
                        pairs.append(DetectedLabelValuePair(
                            label=label,
                            value=next_line,
                            confidence=0.7,
                            pattern_type="colon_next_line"
                        ))

            # Pattern 3: Key-value with equals sign
            equals_match = re.match(r'^([A-Za-z\s]+?)\s*=\s*(.+)$', line)
            if equals_match:
                label = equals_match.group(1).strip()
                value = equals_match.group(2).strip()

                if self._is_valid_label_value_pair(label, value):
                    pairs.append(DetectedLabelValuePair(
                        label=label,
                        value=value,
                        confidence=0.6,
                        pattern_type="equals_sign"
                    ))

        return pairs


    def _extract_using_spatial_analysis(self, ocr_data: Dict) -> List[DetectedLabelValuePair]:
        """Extract using spatial relationships between text elements"""
        pairs = []

        # Create list of text elements with positions
        elements = []
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if text and int(ocr_data['conf'][i]) > 30:
                elements.append({
                    'text': text,
                    'left': ocr_data['left'][i],
                    'top': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i],
                    'conf': ocr_data['conf'][i]
                })

        # Find spatial relationships
        for i, elem1 in enumerate(elements):
            # Look for elements that might be labels
            if self._looks_like_label(elem1['text']):

                # Find nearby elements that could be values
                for j, elem2 in enumerate(elements):
                    if i == j:
                        continue

                    relationship = self._analyze_spatial_relationship(elem1, elem2)

                    if relationship in ['right_adjacent', 'below_aligned']:
                        label = elem1['text'].rstrip(':').strip()
                        value = elem2['text'].strip()

                        if self._is_valid_label_value_pair(label, value):
                            confidence = 0.6 if relationship == 'right_adjacent' else 0.5

                            pairs.append(DetectedLabelValuePair(
                                label=label,
                                value=value,
                                confidence=confidence,
                                label_bbox=(elem1['left'], elem1['top'], elem1['width'], elem1['height']),
                                value_bbox=(elem2['left'], elem2['top'], elem2['width'], elem2['height']),
                                pattern_type="spatial_analysis",
                                spatial_relationship=relationship
                            ))

        return pairs


    def _extract_using_nlp(self, text: str, ocr_data: Dict) -> List[DetectedLabelValuePair]:
        """Extract using NLP techniques for semantic understanding"""
        pairs = []

        if not self.nlp:
            return pairs

        doc = self.nlp(text)

        # Extract named entities
        entities = {}
        for ent in doc.ents:
            entities[ent.label_] = ent.text

        # Map entities to semantic categories
        for ent_type, ent_text in entities.items():
            semantic_label = self._map_entity_to_semantic_label(ent_type, ent_text)
            if semantic_label:
                pairs.append(DetectedLabelValuePair(
                    label=semantic_label,
                    value=ent_text,
                    confidence=0.7,
                    pattern_type="nlp_entity"
                ))

        return pairs


    def _extract_from_tables(self, text: str, ocr_data: Dict) -> List[DetectedLabelValuePair]:
        """Extract from table-like structures"""
        pairs = []

        # Simple table detection based on alignment
        lines = text.split('\n')

        # Look for lines with multiple tab-separated or space-separated values
        for line in lines:
            # Check for tab-separated values
            if '\t' in line:
                parts = [p.strip() for p in line.split('\t') if p.strip()]
                if len(parts) == 2:
                    pairs.append(DetectedLabelValuePair(
                        label=parts[0],
                        value=parts[1],
                        confidence=0.6,
                        pattern_type="table_tab_separated"
                    ))

            # Check for items separated by multiple spaces (column alignment)
            elif re.search(r'\s{3,}', line):
                parts = re.split(r'\s{3,}', line.strip())
                if len(parts) == 2:
                    parts = [p.strip() for p in parts if p.strip()]
                    if len(parts) == 2:
                        pairs.append(DetectedLabelValuePair(
                            label=parts[0],
                            value=parts[1],
                            confidence=0.5,
                            pattern_type="table_space_separated"
                        ))

        return pairs


    def _is_valid_label_value_pair(self, label: str, value: str) -> bool:
        """Validate if a label-value pair is meaningful"""
        # Filter out noise
        if len(label) < 2 or len(value) < 1:
            return False

        # Label should be mostly alphabetic
        if not re.search(r'[A-Za-z]', label):
            return False

        # Value shouldn't be too long (likely paragraph text)
        if len(value) > 100:
            return False

        # Skip common noise patterns
        noise_patterns = [
            r'^\d+$',  # Just numbers as labels
            r'^[^A-Za-z]+$',  # No letters in label
            r'^(the|and|or|but|in|on|at|to|for|of|with|by)$'  # Common words
        ]

        for pattern in noise_patterns:
            if re.match(pattern, label.lower()):
                return False

        return True


    def _looks_like_label(self, text: str) -> bool:
        """Check if text looks like a label"""
        # Ends with colon
        if text.endswith(':'):
            return True

        # Contains label-like words
        label_words = ['name', 'number', 'date', 'amount', 'total', 'address', 'phone', 'email']
        for word in label_words:
            if word in text.lower():
                return True

        return False


    def _analyze_spatial_relationship(self, elem1: Dict, elem2: Dict) -> str:
        """Analyze spatial relationship between two text elements"""
        # Calculate positions
        e1_right = elem1['left'] + elem1['width']
        e1_bottom = elem1['top'] + elem1['height']

        # Horizontal alignment tolerance
        h_tolerance = 20
        v_tolerance = 10

        # Right adjacent (same line, elem2 is to the right)
        if (abs(elem1['top'] - elem2['top']) < v_tolerance and 
            elem2['left'] > e1_right and 
            elem2['left'] - e1_right < 50):
            return 'right_adjacent'

        # Below and aligned (elem2 is below elem1, horizontally aligned)
        if (elem2['top'] > e1_bottom and 
            elem2['top'] - e1_bottom < 30 and
            abs(elem1['left'] - elem2['left']) < h_tolerance):
            return 'below_aligned'

        # Below and indented
        if (elem2['top'] > e1_bottom and 
            elem2['top'] - e1_bottom < 30 and
            elem2['left'] > elem1['left'] + 20):
            return 'below_indented'

        return 'unrelated'


    def _map_entity_to_semantic_label(self, ent_type: str, ent_text: str) -> Optional[str]:
        """Map spaCy entity types to semantic labels"""
        mapping = {
            'MONEY': 'Amount',
            'DATE': 'Date',
            'ORG': 'Organization',
            'PERSON': 'Person Name',
            'GPE': 'Location',
            'CARDINAL': 'Number',
            'ORDINAL': 'Order Number'
        }
        return mapping.get(ent_type)


    def _deduplicate_pairs(self, pairs: List[DetectedLabelValuePair]) -> List[DetectedLabelValuePair]:
        """Remove duplicate pairs and merge similar ones"""
        # Group by similar labels
        grouped = defaultdict(list)

        for pair in pairs:
            # Normalize label for grouping
            normalized_label = re.sub(r'[^\w\s]', '', pair.label.lower()).strip()
            grouped[normalized_label].append(pair)

        # Keep the highest confidence pair from each group
        deduplicated = []
        for label_group in grouped.values():
            # Sort by confidence and take the best one
            best_pair = max(label_group, key=lambda p: p.confidence)
            deduplicated.append(best_pair)

        return sorted(deduplicated, key=lambda p: p.confidence, reverse=True)
