# PCB Defect Detection Using Digital Image Processing and Machine Learning

## Overview
This project implements an automated Printed Circuit Board (PCB) defect detection system using digital image processing and machine learning techniques. It compares a reference PCB image with a test image to identify potential manufacturing defects.

## Objectives
- Automate PCB defect inspection.
- Detect differences between reference and test images.
- Reduce noise and improve image quality through preprocessing.
- Identify defective regions using morphological operations and connected-component analysis.
- Extract image features and classify defects using a Support Vector Machine (SVM).

## Methodology
The system follows a modular image-processing pipeline:

1. **Image Acquisition:** Load reference and test PCB images.
2. **Preprocessing:** Convert images to grayscale and reduce noise.
3. **Illumination Correction:** Minimize intensity variations.
4. **Image Alignment:** Align the test image with the reference image.
5. **Difference Detection:** Identify discrepancies between the images.
6. **Morphological Processing:** Apply opening and closing operations to refine defect regions.
7. **Connected-Component Analysis:** Extract individual candidate defect regions.
8. **Feature Extraction:** Calculate numerical features from detected regions.
9. **SVM Classification:** Train and evaluate a machine-learning classifier using extracted features.

## Technologies Used
- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Digital Image Processing
- Mathematical Morphology
- Support Vector Machine (SVM)

## Applications
- PCB manufacturing quality control
- Automated visual inspection
- Electronic component production
- Industrial defect analysis

## Future Scope
- Evaluate performance on larger and more diverse PCB datasets.
- Improve robustness to variations in lighting, orientation, and image quality.
- Extend defect classification to additional defect categories.
- Explore deep-learning models for comparison with traditional image-processing and machine-learning methods.
- Develop a real-time inspection interface.

## Conclusion
The project demonstrates a modular approach to PCB defect detection by combining reference-image comparison, morphological processing, feature extraction, and machine learning. It provides a foundation for further research in automated electronic manufacturing inspection.

## Author
MITAKSHARA SRIVASTAVA 
ALOK KUMAR MUNNA LAL SINGH 
PRIYA BHADORIA
