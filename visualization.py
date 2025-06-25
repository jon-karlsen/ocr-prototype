import cv2
import numpy as np
from typing import List


from data_models import DetectedLabelValuePair


def visualize_detected_pairs(image: np.ndarray, pairs: List[DetectedLabelValuePair]) -> np.ndarray:
    """Visualize detected label-value pairs on the image"""
    img_with_boxes = image.copy()
    if len(img_with_boxes.shape) == 2:
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_GRAY2RGB)

    colors = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (255, 0, 255), # Magenta
        (0, 255, 255), # Cyan
    ]

    for i, pair in enumerate(pairs[:6]):  # Limit to 6 colors
        color = colors[i % len(colors)]

        # Draw label box
        if pair.label_bbox:
            x, y, w, h = pair.label_bbox
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_with_boxes, "L", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw value box
        if pair.value_bbox:
            x, y, w, h = pair.value_bbox
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_with_boxes, "V", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img_with_boxes


def create_analysis_summary(detected_pairs: List[DetectedLabelValuePair], 
                          filtered_pairs: List[DetectedLabelValuePair]) -> dict:
    """Create analysis summary for the extraction process"""
    method_counts = {}
    for pair in detected_pairs:
        method_counts[pair.pattern_type] = method_counts.get(pair.pattern_type, 0) + 1

    confidence_stats = {
        'min': min([p.confidence for p in detected_pairs]) if detected_pairs else 0,
        'max': max([p.confidence for p in detected_pairs]) if detected_pairs else 0,
        'avg': sum([p.confidence for p in detected_pairs]) / len(detected_pairs) if detected_pairs else 0
    }

    return {
        'total_detected': len(detected_pairs),
        'total_filtered': len(filtered_pairs),
        'method_counts': method_counts,
        'confidence_stats': confidence_stats
    }
