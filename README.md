# ICU Patient Mortality Predictor & Physiological Analyzer (ICU-MPLA)

This repository contains the source code and documentation for the **Advanced Python Programming Final Project**.
The system is built using the **PhysioNet/CinC Challenge 2012** dataset, designed to predict the in-hospital mortality risk of patients admitted to the Intensive Care Unit (ICU) based on physiological time-series observations collected within the first 48 hours of admission.

---

## Key Features
1. **Object-Oriented Design (OOP)**: Clean, class-based modular software design including `PatientRecord` for parsing individual patient text profiles, `FeatureExtractor` for longitudinal aggregations, and `ICUPredictor` for model lifecycle management.
2. **Multiprocessing Parallelization**: Dramatic extraction speedup using Python's `concurrent.futures.ProcessPoolExecutor`. Aggregating 8,000 patient records drops from **169 seconds (sequential)** to **21 seconds (parallel)** on a 4-core CPU (a **~8x speedup**).
3. **Robust System Utilities**: Exposes a custom `@timer` decorator using `functools.wraps` for execution benchmarks and integrates a professional `logging` system (with levels debug, info, warning, error) rather than basic `print()` statements.
4. **Command-Line Interface (CLI)**: Supports configurable execution arguments via `argparse` (e.g. classifier type, concurrent worker processes, cross-validation folds, custom decision thresholds).
5. **Automated Unit Testing**: Includes unit tests written in Python's standard `unittest` framework to verify parser logic, decorator actions, and feature aggregate computations.
6. **Professional Streamlit Web Dashboard**: A unified dark-theme clinical platform designed to fit on a single screen without vertical scrolling, containing:
   * **Patient Longitudinal Explorer**: Displays a horizontal grid of vitals cards with auto-height layout, a Plotly Gauge risk estimator (with suffix and font constraint to prevent clipping), scrollable recent alerts, and longitudinal trend plots.
   * **Clinical Risk Simulator**: Clinically grouped 5x2 slider input controls with side-by-side risk predictions and clinical recommendations, updating in real-time without page resets.
   * **Model Performance & Cohort**: A 3-column layout visualizing ROC curves, feature importances, and dataset cohort metrics side-by-side.

---

## Directory Structure
```text
├── data/                    # Dataset directory
│   ├── set-a/               # Raw patient txt files for training (4,000 records)
│   ├── set-b/               # Raw patient txt files for testing (4,000 records)
│   ├── Outcomes-a.txt       # Training outcomes, SAPS-I, and SOFA scores
│   ├── Outcomes-b.txt       # Testing outcomes (labels)
│   ├── features_set_a.csv   # Cached training feature matrix
│   └── features_set_b.csv   # Cached testing feature matrix
├── src/                     # Core source modules
│   ├── data_downloader.py   # Dataset downloader and extractor
│   ├── data_loader.py       # Patient file parser (PatientRecord class)
│   ├── features.py          # Feature extraction and aggregation logic (FeatureExtractor class)
│   ├── model.py             # Validation framework and serialization (ICUPredictor class)
│   └── utils.py             # Logging setup and custom decorators (@timer)
├── tests/                   # Test suite directory
│   └── test_pipeline.py     # Automated unit tests
├── outputs/                 # Model artifacts and outputs
│   ├── best_model.joblib    # Serialized XGBoost model pipeline
│   ├── roc_curves.png       # ROC curve comparison plot
│   ├── feature_importance.png# Physiological feature importance rank plot
│   └── evaluation_summary.txt# Text summary of validation performance
├── app.py                   # Streamlit web application
├── train.py                 # Main training and evaluation script (CLI entrypoint)
├── requirements.txt         # Project dependencies
└── Final_Report.md          # Technical report draft (English)
```

---

## Quick Start Guide

### 1. Install Dependencies
Install all required Python packages using pip:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
Verify the pipeline's core parser and decorators function correctly:
```bash
python -m unittest tests/test_pipeline.py
```

### 3. Download Data & Train Models
Run the training CLI script with multiprocessing workers (e.g. 4 workers):
```bash
python train.py --model XGBoost --workers 4
```
*This script automatically checks for dataset existence, runs the parallelized feature extraction (~21 seconds), and trains/evaluates the ML classifiers.*

### 4. Launch Streamlit Dashboard
Once the training is complete, start the web interface:
```bash
streamlit run app.py
```
Your browser will open `http://localhost:8501` automatically.


---

## Model Evaluation Summary (Test Set B)
The system uses an optimized **XGBoost Classifier** as its core predictive engine:
* **Accuracy**: `87.30%`
* **Area Under ROC Curve (AUROC)**: `0.8612` (highly robust discrimination)
* **PhysioNet Challenge Score (min(Se, +P))**: `0.4137`
  * Sensitivity (Recall): `41.37%`
  * Positive Predictivity (Precision): `57.32%`
* **Confusion Matrix**:
  * True Negatives (TN): 3,257
  * False Positives (FP): 175
  * False Negatives (FN): 333
  * True Positives (TP): 235

*Note: In critical care scenarios, the decision threshold was optimized to 0.66 to prioritize precision and prevent clinical alert fatigue while maintaining strong sensitivity.*
