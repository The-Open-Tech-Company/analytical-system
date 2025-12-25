# Analytical System - Detailed Documentation

**Repository:** https://github.com/The-Open-Tech-Company/analytical-system  
**License:** Unlicense (Open Source)  
**Open Source:** Yes

## Overview

The Analytical System is a comprehensive biometric analysis platform designed for face and voice recognition, comparison, and analysis. It provides advanced features for extracting, comparing, and visualizing biometric characteristics.

## Face Analysis Module

### Face Feature Extraction

The system uses MediaPipe Face Mesh to extract 468 facial landmarks and analyzes the following features:

1. **Face Oval** - Outer contour of the face
2. **Head Shape** - Overall head contour
3. **Eyes** - Left and right eye contours
4. **Eyebrows** - Left and right eyebrow shapes
5. **Nose** - Bridge, tip, and contour
6. **Mouth** - Outer and inner contours, upper and lower lips
7. **Cheeks** - Left and right cheek areas
8. **Ears** - Left and right ear contours and details
9. **Chin** - Chin shape and contour
10. **Forehead** - Forehead area
11. **Hair** - Hairline and temple areas

### Gender Detection

Multiple detection methods with priority order:

1. **Advanced Gender Detector** (Ensemble) - Combines multiple models for highest accuracy
2. **DeepFace** - Uses ArcFace or VGG-Face models
3. **OpenCV DNN** - Caffe-based gender detection models
4. **Heuristic Method** - Geometric analysis of facial proportions

### Age Estimation

Methods:
- **DeepFace** - Neural network-based age estimation
- **OpenCV DNN** - Caffe-based age models
- **Heuristic** - Analysis of facial proportions and texture

### Race/Ethnicity Detection

Methods:
- **Race Analyzer** - DeepFace/InsightFace-based detection
- **Heuristic** - Analysis of facial geometry and skin tone

### Face Comparison

The comparison system:
- Normalizes face sizes and orientations
- Compares individual features with weighted scoring
- Uses critical features (eyes, nose, mouth) for higher weight
- Provides similarity percentages for each feature
- Calculates overall similarity score

**Critical Features** (60% weight):
- Left eye, Right eye
- Nose tip, Nose bridge
- Mouth outer contour
- Face oval
- Chin

**Other Features** (40% weight):
- Eyebrows, Cheeks, Ears, Forehead, etc.

### Face Database

SQLite-based database for storing and searching faces:
- Store face features with metadata (name, birth year, additional info)
- Search for similar faces
- Update and delete records
- Export/import functionality

## Voice Analysis Module

### Voice Feature Extraction

Extracted features:
- **Pitch (F0)** - Fundamental frequency using PYIN and PIPTrack
- **MFCC** - Mel-frequency cepstral coefficients (13 coefficients)
- **Spectral Centroid** - Center of mass of spectrum
- **Zero Crossing Rate** - Frequency of zero crossings
- **RMS Energy** - Root mean square energy
- **Formants** - F1 and F2 formant frequencies
- **Speech Rate** - Syllables per second

### Gender Detection

Uses pitch, formants, and spectral characteristics:
- **Male voices**: 80-180 Hz (median ~120 Hz)
- **Female voices**: 150-300 Hz (median ~220 Hz)
- Threshold: ~165 Hz for Russian language

### Language Detection

Two methods combined:
1. **Acoustic Analysis** - MFCC, formants, spectral features
2. **Speech Recognition** - Google Speech Recognition API

Supported languages:
- Russian (primary)
- English
- Spanish
- German
- French
- Italian
- Polish
- Ukrainian

### Accent Detection

Analyzes formant patterns and MFCC characteristics to detect:
- Native speaker (no accent)
- Russian accent
- English accent
- Other accents

### Emotion Detection

Detects emotions from voice characteristics:
- Joy/Excitement
- Sadness
- Fear/Anxiety
- Anger
- Neutral

### Voice Comparison

Compares voices using:
- Pitch similarity
- MFCC cosine similarity
- Formant comparison
- Spectral centroid comparison
- Energy and ZCR comparison

## Technical Details

### Face Analysis Pipeline

1. **Image Loading** - Multiple fallback methods (OpenCV, NumPy, PIL)
2. **Preprocessing** - CLAHE enhancement, brightness/contrast adjustment
3. **Face Detection** - MediaPipe Face Detection (low threshold for poor quality images)
4. **Landmark Extraction** - MediaPipe Face Mesh (468 points)
5. **Feature Extraction** - Interpolation to increase point density (30x)
6. **Normalization** - Size and orientation normalization
7. **Comparison** - Point-to-point and metric-based comparison

### Voice Analysis Pipeline

1. **Audio Loading** - Librosa/SoundFile with format conversion
2. **Noise Filtering** - Median filter for clicks, high-pass filter for squeaks
3. **Feature Extraction** - Pitch, MFCC, spectral features
4. **Analysis** - Gender, language, accent, emotion detection
5. **Comparison** - Feature vector comparison

### Performance Optimizations

- Caching of model initializations
- Efficient NumPy operations
- Parallel processing where possible
- Memory-efficient image processing
- Optimized database queries

## GUI Features

### Face Analysis Window
- Image loading and display
- Real-time analysis results
- Interactive visualization
- Database integration

### Face Comparison Window
- Side-by-side image display
- Detailed comparison results
- Visualization modes (overall/detailed)
- Image rotation controls
- Export functionality

### Voice Analysis Window
- Audio file loading
- Real-time recording
- Analysis results display
- Waveform visualization
- Export functionality

### Voice Comparison Window
- Dual audio loading
- Comparison results
- Feature-by-feature analysis
- Similarity scores

## Database Schema

### Faces Table
- `id` - Primary key
- `full_name` - Person's name
- `birth_year` - Birth year (optional)
- `additional_info` - Additional information (optional)
- `face_features` - Pickled face features (BLOB)
- `created_at` - Timestamp

## File Formats

### Supported Image Formats
- JPEG/JPG
- PNG
- BMP
- GIF
- WebP

### Supported Audio Formats
- WAV
- MP3
- FLAC
- OGG
- M4A

## Error Handling

The system includes comprehensive error handling:
- File validation
- Image format detection
- Audio format conversion
- Model initialization fallbacks
- Graceful degradation when optional modules unavailable

## Configuration

Settings can be adjusted in `settings_window.py`:
- Face detection thresholds
- Comparison sensitivity
- Visualization options
- Database settings

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- opencv-contrib-python
- mediapipe
- numpy
- tensorflow
- deepface
- librosa
- soundfile
- speechrecognition

## License

This project is released into the public domain under the Unlicense. You are free to use, modify, distribute, and sell this software for any purpose, commercial or non-commercial.

## Contributing

Contributions are welcome! Please ensure code follows existing style and includes appropriate error handling.

## Support

For issues, questions, or contributions, please visit:
https://github.com/The-Open-Tech-Company/analytical-system

