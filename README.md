# Analytical System

A comprehensive analytical system for face and voice analysis, comparison, and recognition.

**Repository:** https://github.com/The-Open-Tech-Company/analytical-system  
**License:** Unlicense (Open Source)

## Features

### Face Analysis
- **Face Feature Extraction**: Extracts biometric characteristics using MediaPipe Face Mesh
- **Gender, Age, Race Detection**: Advanced detection using multiple methods (DeepFace, DNN, heuristic)
- **Face Comparison**: Detailed comparison of two faces with similarity percentages
- **Visualization**: Interactive visualization of facial features and comparison results
- **Face Database**: Store and search faces in SQLite database

### Voice Analysis
- **Voice Feature Extraction**: Extracts acoustic characteristics (pitch, MFCC, formants, etc.)
- **Gender Detection**: Determines speaker gender from voice characteristics
- **Language Detection**: Identifies spoken language using acoustic analysis and speech recognition
- **Accent Detection**: Detects accent patterns
- **Emotion Detection**: Identifies emotional state from voice
- **Voice Comparison**: Compares two voice samples for similarity

## Requirements

- Python 3.11+
- OpenCV (opencv-contrib-python)
- MediaPipe
- NumPy
- TensorFlow (for DeepFace)
- Librosa (for voice analysis)
- SpeechRecognition (optional, for language detection)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### GUI Application (Recommended)

```bash
python gui_app.py
```

### Command Line

```bash
python main.py <image1_path> <image2_path>
```

## Project Structure

- `gui_app.py` - Main GUI application
- `main.py` - Command-line interface
- `face_analyzer.py` - Face feature extraction
- `face_comparator.py` - Face comparison logic
- `face_visualizer.py` - Face visualization
- `voice_analyzer.py` - Voice feature extraction and analysis
- `voice_comparator.py` - Voice comparison logic
- `face_database.py` - Face database management

## License

This project is released into the public domain under the Unlicense. See LICENSE file for details.
