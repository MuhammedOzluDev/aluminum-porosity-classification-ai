import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from itertools import cycle

# =============================================================================

# =============================================================================

# Full path from the screenshot:
DATA_DIR = r"C:\Users\ESTEKNO\OneDrive - BAKIRÇAY ÜNİVERSİTESİ\Masaüstü\Görüntü işleme VSCODE\the sets of data\RPT_Dataset_3+"

# Image size for EfficientNetB3
IMG_SIZE = (300, 300)

# Reduced to 3 since running on CPU. Set to 1 if the computer lags too much.
AUGMENTATION_FACTOR = 3

# =============================================================================
# FUNCTIONS
# =============================================================================

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

def load_and_augment_data_eff(data_dir, image_size, aug_factor):
    data_path = Path(data_dir)
    all_images = []
    all_labels = []
    class_names = []

    if not data_path.exists():
        print(f"ERROR: Path not found -> {data_dir}")
        print("Please check folder names.")
        return np.array([]), np.array([]), []

    print(f"Reading data from: {data_dir}")
    print(f"Each image will be augmented {aug_factor} times (CPU mode)...")
    
    # Read folders sequentially
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            class_name = class_dir.name
            class_names.append(class_name)
            class_idx = len(class_names) - 1
            
            files = list(class_dir.glob('*'))
            valid_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            print(f"  - Processing class '{class_name}': {len(valid_files)} images.")
            
            for file_path in valid_files:
                try:
                    img = load_img(str(file_path), target_size=image_size)
                    img_arr = img_to_array(img)
                    
                    # 1. Add original image
                    all_images.append(preprocess_input(img_arr.copy()))
                    all_labels.append(class_idx)
                    
                    # 2. Augmentation
                    if aug_factor > 0:
                        img_arr_reshaped = img_arr.reshape((1,) + img_arr.shape)
                        i = 0
                        for batch in datagen.flow(img_arr_reshaped, batch_size=1):
                            aug_img = batch[0]
                            all_images.append(preprocess_input(aug_img))
                            all_labels.append(class_idx)
                            i += 1
                            if i >= aug_factor:
                                break
                except Exception as e:
                    print(f"Error ({file_path.name}): {e}")

    return np.array(all_images), np.array(all_labels), class_names

# =============================================================================
# MAIN BLOCK (Required for Windows)
# =============================================================================
if __name__ == '__main__':
    
    # 1. Load Data
    X_all, y_all, class_names = load_and_augment_data_eff(DATA_DIR, IMG_SIZE, AUGMENTATION_FACTOR)

    if len(X_all) == 0:
        print("Data could not be loaded! Terminating program.")
        exit()

    print(f"\nTotal Processed Data Count: {len(X_all)}")

    # 2. Train/Test Split
    X_train_img, X_test_img, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # 3. Feature Extraction (EfficientNetB3)
    print("\nExtracting features with EfficientNetB3 (This will take time on CPU, please wait)...")
    
    # Reduced batch_size to avoid memory issues on CPU
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3), pooling='avg')
    
    features_train = base_model.predict(X_train_img, batch_size=4, verbose=1)
    features_test = base_model.predict(X_test_img, batch_size=4, verbose=1)

    print(f"Feature shape: {features_train.shape}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(features_train)
    X_test_scaled = scaler.transform(features_test)

    # 4. Random Forest Training and Optimization
    print("\nTraining Random Forest...")

    param_dist = {
        'n_estimators': [200, 400],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42)

    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=5,  # Reduced number of iterations (For speed)
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='accuracy'
    )

    random_search.fit(X_train_scaled, y_train)

    # 5. Results
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*60)
    print(f"ULTIMATE MODEL ACCURACY: {test_acc * 100:.2f}%")
    print("="*60)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # =============================================================================
    # 6. GRAPHICAL EVALUATION (CONTINUATION OF CODE)
    # =============================================================================
    print("\nGenerating plots...")
    
    # Necessary additional libraries (Already imported at the top, but re-importing just in case)
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
    from itertools import cycle

    # -------------------------------------------------------------------------
    # A) Confusion Matrix
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Classes')
    plt.xlabel('Predicted Classes')
    plt.show()

    # -------------------------------------------------------------------------
    # B) ROC / AUC Curve (For All Classes)
    # -------------------------------------------------------------------------
    # Get probabilities from Random Forest (Required for ROC)
    y_score = best_model.predict_proba(X_test_scaled)
    
    # Binarize labels (For One-vs-Rest logic)
    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    n_classes = y_test_bin.shape[1]

    # Calculate ROC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the graph
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve ({}) (area = {:.2f})'.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC / AUC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # -------------------------------------------------------------------------
    # C) All Metrics in Single Bar Chart (Acc, Prec, Recall, F1)
    # -------------------------------------------------------------------------
    # Calculate metrics (Using weighted average)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [test_acc, precision, recall, f1]
    }
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Metric', y='Value', data=metrics_data, palette='viridis')
    
    # Write values on top of bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', fontsize=12, padding=3)
        
    plt.ylim(0, 1.1) # Fix Y axis between 0 and 1.1
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.show()

    print("\nAll plots generated and displayed.")

    # Save Model
    print("Saving model...")
    joblib.dump(best_model, 'ultimate_effnet_rf_model_local.pkl')
    joblib.dump(scaler, 'effnet_scaler_local.pkl')
    print("Process completed!")