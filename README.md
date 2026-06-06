Aluminum Porosity Classification AI
Machine learning project developed for the MIS 443 – Applied AI in Business course.

Overview
This project aims to automate the classification of aluminum block images according to their porosity levels. Traditionally, this inspection process is performed manually by quality control personnel, making it vulnerable to human error and inconsistency.
The proposed solution combines deep learning feature extraction with classical machine learning classification to improve inspection accuracy and operational efficiency in aluminum wheel manufacturing.

Team Members
Muhammed Özlü
Bahadır Göktürk
Emine Kılıç
Mithat Tuğrul Tek
Abdullah Canlı

Supervisor:

Prof. Dr. Kadir Hızıroğlu
Business Problem

Porosity defects directly affect the quality of aluminum alloy wheels used by automotive manufacturers. Manual inspection can lead to inconsistent decisions, defective products, increased scrap rates, and customer returns.
This project addresses these issues by providing an AI-powered visual inspection system capable of classifying aluminum samples into eight porosity categories.

Methodology
1. Data Preparation
Original dataset: 96 images
Data augmentation applied to balance class distribution
Final dataset size: 320 images
8 porosity classes
Image size: 300 × 300 pixels
2. Feature Extraction

EfficientNetB3 was used as a feature extractor with the final classification layers removed.

Output:
1,536-dimensional feature vectors
3. Feature Scaling
StandardScaler was applied to normalize extracted features before classification.
4. Classification
Random Forest classifier optimized using RandomizedSearchCV.

Hyperparameter optimization included:
Number of estimators
Maximum depth
Minimum samples split
Bootstrap strategy
Technologies Used
Python
TensorFlow
Keras
EfficientNetB3
Scikit-Learn
Random Forest
NumPy
Matplotlib
Seaborn
Joblib
Model Performance

Final Accuracy:
93.36%
Evaluation Metrics:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC/AUC Curves
Project Architecture

Image Input
↓
EfficientNetB3 Feature Extraction
↓
StandardScaler
↓
Random Forest Classifier
↓
Porosity Class Prediction

Deployment
The trained model and scaler were exported using Joblib for production deployment.

Potential industrial workflow:
Production line camera captures image.
AI model classifies porosity level.
n8n workflow receives prediction.
PLC receives signal.
Robotic arm redirects the product according to its class.
