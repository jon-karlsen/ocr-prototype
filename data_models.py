from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
from datetime import datetime


@dataclass
class DetectedLabelValuePair:
    """Represents a dynamically detected label-value pair"""
    label: str
    value: str
    confidence: float
    label_bbox: Optional[Tuple[int, int, int, int]] = None
    value_bbox: Optional[Tuple[int, int, int, int]] = None
    pattern_type: str = ""
    spatial_relationship: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class CorrectionRecord:
    """Records manual corrections for learning"""
    original_label: str
    corrected_label: str
    original_value: str
    corrected_value: str
    spatial_features: Dict[str, float]
    document_type: str
    timestamp: datetime
    user_action: str  # 'add', 'modify', 'delete'

    def to_dict(self):
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }
