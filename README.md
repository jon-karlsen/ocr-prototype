# 🧠 Smart Document Extractor with Learning

An intelligent OCR-based document processing system that extracts label-value pairs from any document and learns from manual corrections to improve over time.

## 🚀 Features

- **Dynamic Label Discovery**: Automatically finds label-value patterns without predefined templates
- **Multiple Extraction Methods**: Pattern-based, spatial analysis, NLP, and table detection
- **Manual Correction Interface**: Easy-to-use interface for correcting extraction errors
- **Machine Learning**: Learns from corrections and improves future extractions
- **Document Type Support**: Separate models for invoices, receipts, forms, contracts
- **Export Options**: JSON, CSV, and text formats
- **Visualization**: Bounding box visualization of detected elements

## 📁 Project Structure

```
smart-document-extractor/
├── main.py                    # Main Streamlit application
├── data_models.py            # Data classes and structures
├── label_extractor.py        # Core extraction logic
├── layout_learner.py         # Machine learning component
├── correction_interface.py   # Manual correction UI
├── visualization.py          # Visualization utilities
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## 🛠️ Installation

### 1. Clone or Download Files
Save all the Python files in the same directory.

### 2. Install Dependencies

**Basic Installation (Core Features):**
```bash
pip install streamlit pytesseract Pillow opencv-python pandas numpy
```

**Full Installation (All Features):**
```bash
pip install -r requirements.txt
```

**Optional NLP Features:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### 3. Install Tesseract OCR

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH or set `pytesseract.pytesseract.tesseract_cmd`

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

## 🏃‍♂️ Running the Application

```bash
streamlit run main.py
```

The application will open in your web browser at `http://localhost:8501`

## 📋 Usage Guide

### 1. Basic Extraction
1. Upload a document image (PNG, JPG, TIFF, etc.)
2. Select document type (Invoice, Receipt, Form, etc.)
3. Adjust confidence threshold if needed
4. Review automatically extracted label-value pairs

### 2. Manual Correction
1. Use the correction interface to:
   - Modify incorrect labels or values
   - Delete unwanted pairs
   - Add missing pairs
2. Click "Save Corrections & Learn" to train the AI

### 3. Export Data
- Choose from JSON, CSV, or TXT formats
- Data includes all corrected label-value pairs

### 4. Learning System
- After 10+ corrections, the system trains ML models
- Future documents of the same type will benefit from learned patterns
- Models are saved in the `models/` directory

## 🎯 Supported Document Types

- **Invoices**: Vendor info, amounts, dates, line items
- **Receipts**: Store names, totals, items, tax
- **Forms**: Any structured form with labels and fields  
- **Contracts**: Parties, dates, terms, clauses
- **Custom**: Any document with label-value patterns

## 🔧 Configuration

### Extraction Methods
- **Pattern-based**: Regex patterns for "Label: Value" formats
- **Spatial Analysis**: Uses OCR bounding boxes to find relationships
- **NLP (spaCy)**: Semantic understanding of entities
- **Table Detection**: Extracts from tab/space-separated data

### Settings
- **Confidence Threshold**: Filter low-confidence extractions
- **Bounding Boxes**: Visual feedback for spatial analysis
- **ML Corrections**: Enable/disable learning features

## 📊 Learning Analytics

The system tracks:
- Total corrections made
- Corrections per document type
- Model training status
- Extraction method performance

## 🔍 Troubleshooting

### Common Issues

**"Tesseract not found"**
- Install Tesseract OCR and ensure it's in your PATH
- On Windows, you may need to set the path manually

**"spaCy model not found"**
- Run: `python -m spacy download en_core_web_sm`
- Or disable NLP features in settings

**"scikit-learn not available"**
- Install with: `pip install scikit-learn joblib`
- Learning features will be disabled without it

**No bounding boxes showing**
- Enable "Spatial Analysis" extraction method
- Check that OCR is detecting text elements properly
- Try adjusting confidence threshold

### Performance Tips

- Use high-quality, clear document images
- Ensure good contrast between text and background
- Crop images to focus on relevant content
- Use appropriate document type classification

## 🧪 Development

### Adding New Extraction Methods

1. Add method to `DynamicLabelExtractor` class
2. Call from `extract_dynamic_labels()` method
3. Return list of `DetectedLabelValuePair` objects

### Extending Learning Features

1. Modify `LayoutLearner` class for new ML algorithms
2. Add features to `_extract_spatial_features()` method
3. Update training logic in `train_from_corrections()`

### Custom Document Types

1. Add new type to document type dropdown
2. Create type-specific extraction patterns
3. Train separate models for each type

## 📄 License

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional extraction patterns
- Better spatial analysis algorithms
- Advanced ML models
- Template matching features
- Multi-language support
- Active learning capabilities

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Test with different document types and settings
