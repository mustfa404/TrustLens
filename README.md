# TrustLens
# Fraud Detection ML Dashboard

A desktop-based machine learning application for fraud detection using multiple ML models. Built with Python Tkinter and integrated with MLflow for experiment tracking, model logging, and performance visualization.

---

# Features

* Train multiple ML models:
  * Decision Tree
  * Random Forest
  * K-Nearest Neighbors (KNN)
  * Support Vector Machine (SVM)

* Full data preprocessing pipeline:
  * Feature engineering (age extraction from DOB)
  * Label encoding for categorical variables
  * Feature scaling (StandardScaler)
  * Class balancing using upsampling

* Model evaluation metrics:
  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * Confusion Matrix
  * Classification Report

* MLflow integration:
  * Experiment tracking
  * Parameter logging
  * Metric logging
  * Model saving

* Interactive GUI:
  * Load train/test CSV files
  * Select models to train
  * Real-time training logs
  * Progress bar
  * Performance charts
  * Confusion matrix visualization
  * Best model selection

---

# Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib mlflow
```

Note: tkinter is included with Python by default.

---

# How to Run

```bash
python fraud_detection_app.py
```

---

# Dataset Format

The dataset must include:

is_fraud (0 = Legit, 1 = Fraud)

Supported features:
* Transaction-related fields
* Categorical variables
* Date of birth (dob)
* Metadata fields

Automatically removed columns:
* Unnamed: 0, trans_num, first, last, street, cc_num, trans_date_trans_time

---

# MLflow Settings

Default tracking database:
sqlite:///fraud_mlflow.db

You can change:
* Experiment name
* Tracking URI

directly from the GUI.

---

# Outputs

The application provides:

* Model comparison dashboard
* Performance charts
* Confusion matrices
* Classification reports
* Best model selection (based on F1-score)
* MLflow experiment logs

---

# Workflow

1. Load training and test datasets
2. Preprocess data (cleaning, encoding, scaling)
3. Balance dataset (fraud upsampling)
4. Train selected models
5. Evaluate performance
6. Log results to MLflow
7. Visualize results in GUI

---

# Models Used

* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors
* Support Vector Machine (RBF kernel)

---

# Project Structure

fraud_detection_app.py

---

# Future Improvements

* Deep learning models
* Hyperparameter tuning (GridSearch / Optuna)
* REST API deployment (Flask/FastAPI)
* Real-time fraud prediction
* Feature importance visualization

---

# Technologies Used

* Python
* Tkinter
* Scikit-learn
* Matplotlib
* MLflow

---

# License

This project is open-source and free to use for educational and research purposes.
