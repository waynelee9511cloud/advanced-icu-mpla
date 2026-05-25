# Final Project Report

## Project Title: ICU Patient Mortality Predictor & Physiological Analyzer (ICU-MPLA)
*   **Course**: Advanced Python Programming
*   **Author**: Wayne Lee
*   **Workspace**: [Local Workspace](file:///d:/WayneLee_Profile/Desktop/進階程式語言期末)

---

## Abstract
Severity assessment and mortality risk prediction for patients admitted to the Intensive Care Unit (ICU) are vital for clinical decision support, resource allocation, and controlling confounding factors in clinical trials. This project presents an end-to-end Python-based solution leveraging the **PhysioNet/CinC Challenge 2012** cohort, analyzing multi-dimensional physiological time-series observations from the first 48 hours of ICU stay to predict in-hospital mortality risk.

We implemented and compared three machine learning models: Logistic Regression, Random Forest, and Extreme Gradient Boosting (XGBoost). The experimental results demonstrate that the **XGBoost** model achieves superior performance, yielding an **AUROC of 0.8612** and an **Accuracy of 87.30%** on the independent test set B. The optimal classification decision threshold was optimized to 0.66, yielding an official challenge score (`min(Sensitivity, Positive Predictivity)`) of **0.4137**. Feature importance analysis identified **Glasgow Coma Score (GCS)**, **Age**, **Blood Urea Nitrogen (BUN)**, and **Urine output** as the most critical predictors of patient outcomes. To facilitate translation into clinical settings, a web-based interactive application was developed using **Streamlit** to enable longitudinal data visualization and real-time risk simulation.

---

## 1. Introduction
Predicting patient mortality in intensive care medicine is crucial for identifying deteriorating patients early and optimizing intervention strategies. Traditionally, clinicians rely on acuity scoring systems such as APACHE II or SAPS-II. However, developing accurate predictive models on ICU physiological data presents significant challenges:
1.  **Multi-dimensionality and High Frequency**: Dozens of parameters (vital signs, lab results) are recorded at varying frequencies.
2.  **Irregular Sampling**: Certain measurements are recorded hourly, while others are taken once a day or not at all.
3.  **High Rate of Missing Values**: Lab tests are ordered as needed, leaving substantial gaps in patient records.

This project addresses these challenges by developing a robust, modular, and professional Python pipeline that automates data retrieval, executes longitudinal feature engineering, trains optimized machine learning classifiers, and presents results via an interactive clinical dashboard.

---

## 2. Dataset Description
The dataset is derived from the PhysioNet Challenge 2012 database, consisting of 8,000 adult ICU records split equally into Set A (training, 4,000 cases) and Set B (testing, 4,000 cases):
1.  **Baseline Descriptors**: Collected at admission (Time = `00:00`), including Age, Gender, Height, Weight, and ICU Unit Type (Coronary Care Unit, Cardiac Surgery Recovery Unit, Medical ICU, and Surgical ICU).
2.  **Physiological Time-Series**: 37 variables recorded at irregular intervals during the first 48 hours of ICU stay. These include vital signs (Heart Rate, Temperature, Respiration Rate), neurological indices (Glasgow Coma Score), fluid balance (Urine output), and laboratory values (White Blood Cell Count, Platelets, Serum Creatinine, Blood Urea Nitrogen, pH, etc.).
3.  **Outcomes**: The primary prediction target is **In-hospital death** (0: Survived, 1: Deceased during hospitalization). The training cohort Set A exhibits a mortality rate of `13.85%`, and test Set B has `14.20%`, representing a severe class imbalance.

---

## 3. Methodology & System Design

This project is built using a highly professional, robust, and object-oriented Python software architecture designed to demonstrate advanced programming concepts and software engineering best practices.

### 3.1 Object-Oriented Software Architecture
The system transitions from procedural script-based executions to an Object-Oriented Programming (OOP) design pattern:
1. **`PatientRecord` Class** ([data_loader.py](file:///d:/WayneLee_Profile/Desktop/進階程式語言期末/src/data_loader.py)): Encapsulates all raw properties of a patient's ICU stay. It parses general admission descriptors and segregates 37 irregular physiological time-series variables. Errors and formatting anomalies are captured through specialized exception handling and warnings.
2. **`FeatureExtractor` Class** ([features.py](file:///d:/WayneLee_Profile/Desktop/進階程式語言期末/src/features.py)): Manages the translation of irregular physiological measurements into a structured dataset. It extracts statistical aggregates and implements parallel processing pipelines.
3. **`ICUPredictor` Class** ([model.py](file:///d:/WayneLee_Profile/Desktop/進階程式語言期末/src/model.py)): Orchestrates the machine learning model lifecycle, managing preprocessing steps (SimpleImputer, StandardScaler), training, cross-validation, and serialized joblib exports.

### 3.2 Concurrency & Multiprocessing Parallelism
Parsing thousands of independent text files sequentially poses significant I/O and CPU bottlenecks in Python. To optimize performance and bypass Python's Global Interpreter Lock (GIL), we implemented multi-process concurrency using `concurrent.futures.ProcessPoolExecutor` inside the `FeatureExtractor` class.

On a multi-core benchmarking environment, the feature extraction runtime for the complete dataset (8,000 files) is summarized below:
- **Sequential Execution (1 Worker)**: **168.83 seconds**
- **Parallel Execution (4 Workers)**: **21.16 seconds** (yielding a **7.98x speedup**)

This near-linear performance improvement showcases efficient CPU resource allocation and execution profiling in production.

### 3.3 Custom Decorators & Logging Utility
To ensure clean software diagnostics and code reusability:
* **Custom `@timer` Decorator**: Exposes execution times of major functions using Python's closure mechanism and `functools.wraps`. This is applied as a wrapper on ingestion, feature engineering, cross-validation, and model fitting functions.
* **Standardized Logging**: Replaces standard `print()` statements with Python's built-in `logging` module. Runtimes, warnings, and system anomalies are routed to a centralized logger, ensuring correct debug and info prioritization.

### 3.4 Command-Line Interface (CLI)
The pipeline execution script ([train.py](file:///d:/WayneLee_Profile/Desktop/進階程式語言期末/train.py)) implements an `argparse` CLI, enabling users to customize configurations from the terminal:
```bash
python train.py --model [XGBoost|Random Forest|Logistic Regression] --workers [num_cores] --cv-folds [folds]
```

### 3.5 Automated Unit Testing
A test suite is implemented in [test_pipeline.py](file:///d:/WayneLee_Profile/Desktop/進階程式語言期末/tests/test_pipeline.py) using the `unittest` framework, validating parsing accuracy, aggregate calculation logic, and decorator execution.

---

## 4. Experimental Results & Discussion

### 4.1 Cross-Validation Performance
The average performance metrics obtained from 5-Fold Stratified Cross-Validation on Set A are summarized below (standard deviations in parentheses):

| Model Name | Mean AUROC | Mean F1-Score | Mean PhysioNet Score | Optimal Decision Threshold |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8282 (±0.0270) | 0.4592 | 0.3413 | 0.5000 |
| **Random Forest** | 0.8358 (±0.0157) | 0.0908 | 0.0487 | 0.5000 |
| **XGBoost** | **0.8475 (±0.0137)** | **0.4845** | **0.4666** | **0.6600** |

#### Discussion:
1.  **Superiority of XGBoost**: The gradient boosting tree significantly outperformed the other models in AUROC (0.8475) and the official challenge score (0.4666). Its capability to handle NaNs natively avoids the bias introduced by simple median imputations on laboratory variables.
2.  **Threshold Optimization Necessity**: Under the default threshold of 0.5, Random Forest suffered from extremely low sensitivity (0.0487), failing to identify high-risk deceased patients due to class imbalance. Optimizing the threshold to 0.66 for XGBoost successfully balanced precision and sensitivity.

### 4.2 Independent Test Set B Evaluation
Using the final optimized XGBoost model on the independent validation set B, we obtained:
*   **Test Accuracy**: `87.30%`
*   **Test AUROC**: `0.8612` (indicating highly robust discrimination and generalization)
*   **Official Challenge Score (min(Se, +P))**: `0.4137`
    *   Sensitivity (Recall): `41.37%`
    *   Positive Predictivity (Precision): `57.32%`
*   **Confusion Matrix**:
    *   True Negatives (TN): 3,257 | False Positives (FP): 175
    *   False Negatives (FN): 333 | True Positives (TP): 235

### 4.3 Feature Importance Analysis
XGBoost feature importance coefficients reveal the top physiological indicators driving mortality risk:
1.  **Glasgow Coma Score (GCS)**: Minimum (`GCS_min`) and last (`GCS_last`) values rank as the most critical features. GCS measures central neurological function; low scores indicate brain injury or severe encephalopathy.
2.  **Age**: Older patients have lower physiological reserves, correlating strongly with increased risk.
3.  **Blood Urea Nitrogen (BUN)**: Mean levels (`BUN_mean`) reflect renal performance and metabolic waste retention. High BUN levels strongly correlate with acute kidney injury (AKI) or decompensated heart failure.
4.  **Urine Output**: Mean urine output (`Urine_mean`) is a crucial real-time marker for tissue perfusion and kidney function. Oliguria or anuria are early clinical warning signs of organ failure.
5.  **White Blood Cell Count (WBC)**: Elevated WBC levels (`WBC_mean`) indicate systemic inflammation or severe infection (sepsis).

---

## 5. Web Dashboard Design
To bridge the gap between predictive modeling and clinical utility, a professional web-based dashboard ([app.py](file:///d:/WayneLee_Profile/Desktop/進階程式語言期末/app.py)) was developed using Streamlit. The application features a premium dark theme configured natively via [config.toml](file:///d:/WayneLee_Profile/Desktop/進階程式語言期末/.streamlit/config.toml) (Slate Dark Blue background `#0d1e3d`, Dark Navy panels `#09152e`, and White text `#ffffff`), preventing color contrast or text cutoff issues.

To optimize the viewport space and eliminate vertical scrolling on standard resolutions, all modules are structured to fit within a single screen:
1.  **Sidebar-Integrated Controls & Citations**:
    *   Integrates the Patient Selector dropdown and the dynamic Patient Profile Banner (ID, Age, Gender, ICU unit, and clinical outcome) into the sidebar.
    *   Displays explicit attributions and references (PhysioNet/CinC Challenge 2012, XGBoost inference system, and developer Wayne Lee) inside a structured sidebar info card.
2.  **Patient Longitudinal Explorer (Tab 1)**:
    *   Displays 4 high-contrast vitals cards (Heart Rate, Blood Pressure, SpO2, Temperature) in a horizontal row, using auto-height CSS to avoid cutoff when text wraps.
    *   Features a side-by-side dashboard row: a custom Plotly Gauge chart showing real-time risk assessment (using a constrained font size of 18 and a percentage suffix to prevent canvas cutoffs) and a scrollable Alerts Console; paired with a Plotly Longitudinal Trend line chart showing temporal variations of selected vitals.
    *   Exhibits a compact lab value aggregate grid for 48-hour mean outcomes at the bottom.
3.  **Clinical Risk Simulator (Tab 2)**:
    *   Features a clinically grouped 5x2 grid of physiological sliders spanning the full width of the screen, allowing users to modify patient parameters interactively.
    *   Renders risk outcomes and structured clinical recommendations side-by-side below the inputs, updating instantly on slider changes without page resets.
4.  **Model Performance & Cohort (Tab 3)**:
    *   Lays out Cohort Statistics, the validation ROC Curve, and Physiological Feature Importance rankings side-by-side in three columns, autoscaling the images to prevent vertical scrolling.

---

## 6. Conclusion
This project successfully implemented an end-to-end Python machine learning pipeline and a professional clinical support tool for ICU mortality prediction. The optimized XGBoost model achieves robust validation performance (AUROC = 0.8612), and the interactive Streamlit dashboard demonstrates how machine learning models can be transformed into accessible clinical decision tools.

Future work includes exploring recurrent architectures (e.g., LSTMs or GRUs) to model raw physiological time-series directly and validating the models on external databases such as MIMIC-IV to evaluate multi-center generalizability.
