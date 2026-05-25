## slide_layout: title
- top_text: Course: Advanced Python Programming | Wayne Lee
- title: ICU Patient Mortality Predictor & Physiological Analyzer (ICU-MPLA)
- subtitle: ICU Mortality Prediction & Physiological Time-Series Analysis

## slide_layout: section
- title: 1. Introduction & Background

## slide_layout: content
- title: Clinical Background & Challenges
- body:
  - ICU mortality risk prediction is vital for clinical decision support, resource allocation, and patient stratification in clinical trials
  - Traditional acuity scoring systems (e.g., APACHE II, SAPS-II) are static, rule-based, and fail to capture complex temporal dynamics
  - Irregularly sampled physiological time-series data presents unique challenges:
    - High dimensionality and varying sampling frequencies (vital signs vs. lab values)
    - Significant proportion of missing values as lab tests are ordered only when clinically indicated
    - Severe outcome class imbalance (mortality rate ~14%)

## slide_layout: section
- title: 2. Software Architecture & Concurrency

## slide_layout: content
- title: Object-Oriented Software Design
- body:
  - The codebase is refactored from flat procedural scripts to an Object-Oriented design:
    - **`PatientRecord` Class**: Encapsulates clinical demographics and physiological時序 variables; handles file parsing and warnings cleanly
    - **`FeatureExtractor` Class**: Computes statistical aggregates and handles parallel dataset building
    - **`ICUPredictor` Class**: Manages classifier models, cross-validation, threshold optimization, and serialized joblib exports
  - **Decorators & Logging**: A custom `@timer` decorator profiles execution runtimes; the python `logging` library handles diagnostic outputs

## slide_layout: image_content
- title: Core Experimental Ingestion & Orchestration
- body:
  - Replaces scattered scripts with a single automated training/testing orchestrator (train.py)
  - Coordinates raw data downloading, feature extraction, cross-validation, and final testing
  - Executes independent out-of-sample evaluations automatically on Test Set B
- image: outputs/code_experiment_pipeline.png

## slide_layout: image_content
- title: System Profiling via Custom Decorators
- body:
  - Implements a reusable, non-intrusive @timer decorator closure to wrap pipeline functions
  - Uses time.perf_counter() to automatically track execution time of core methods
  - Standardizes logging levels (INFO, ERROR) to replace scattered print statements
  - Preserves original function names and docstrings using functools.wraps
- image: outputs/code_decorator_timer.png

## slide_layout: image_content
- title: Strong Encapsulation: PatientRecord Class
- body:
  - Bundles irregular, sparsely sampled physiological measurements and descriptors into a class
  - Constructor validates demographic indicators and handles missing values safely on load
  - Defines a custom string representation (__repr__) for clean terminal debugging
  - Decouples raw reading and format parsing from subsequent feature engineering steps
- image: outputs/code_oop_patient_record.png

## slide_layout: content
- title: Multiprocessing Performance Benchmark
- body:
  - Parsing thousands of independent patient records sequentially creates significant CPU and I/O bottlenecks in Python
  - Multiprocessing is implemented using `concurrent.futures.ProcessPoolExecutor` to run calculations in parallel on multiple cores
  - **Runtime Benchmarks (8,000 patient records)**:
    - Sequential Feature Extraction (1 Worker): **168.83 seconds**
    - Parallel Feature Extraction (4 Workers): **21.16 seconds**
    - Performance Gain: **7.98x Speedup** (near-linear multi-core efficiency)
  - Features an `argparse` CLI in `train.py` with a `--workers` argument for thread scaling

## slide_layout: image_content
- title: Concurrency: Parallel Feature Extraction
- body:
  - Parallelizes feature aggregation across independent records using a ProcessPoolExecutor
  - Implements a top-level parsing helper function to satisfy Windows pickling limits
  - Achieved a 7.98x speedup (processing 8,000 files in 21.16s vs 168.83s sequentially)
- image: outputs/code_concurrency_parallel.png

## slide_layout: section
- title: 3. Dataset & Feature Engineering

## slide_layout: content
- title: PhysioNet Challenge 2012 Cohort
- body:
  - **Study Cohort**: 8,000 adult ICU records split equally into Set A (development) and Set B (independent test)
  - **Baseline Descriptors**: Age, Gender, Height, Weight, and ICU unit type (CCU, CSRU, MICU, SICU) collected at admission
  - **Physiological Time-Series**: 37 variables monitored over the first 48 hours of ICU stay (e.g., HR, Temp, GCS, Urine, BUN, Creatinine, pH)
  - **Primary Outcome**: In-hospital death (Set A: 13.85% mortality rate; Set B: 14.20% mortality rate)

## slide_layout: content
- title: Longitudinal Feature Engineering
- body:
  - Irregular sequences are converted to fixed-size representations by extracting 5 statistical aggregates per time-series parameter:
    - **Mean**: Captures the overall physiological baseline level over 48 hours
    - **Minimum (Min)**: Highlights critical dips (e.g., severe hypotension or hypothermia)
    - **Maximum (Max)**: Highlights critical spikes (e.g., high fever or severe hypertension)
    - **Last Value**: Represents the patient's state at the end of the 48-hour window
    - **Count**: Captures observation frequency, acting as a clinical proxy for patient instability
  - Combined with baseline descriptors, this yields a structured matrix of **190 features** per patient

## slide_layout: section
- title: 4. Model Development & Evaluation

## slide_layout: two_columns
- title: Model Comparison (5-Fold Stratified CV)
- left_body:
  - **Logistic Regression (Linear Baseline)**:
    - Built with median imputation and standardization
    - Class weights balanced to handle outcome skewness
    - Mean AUROC: 0.8282 (±0.0270)
    - Mean PhysioNet Score: 0.3413
  - **Random Forest (Ensemble Baseline)**:
    - Built with median imputation and 200 decision trees
    - High class-imbalance sensitivity leads to poor default recall
    - Mean AUROC: 0.8358 (±0.0157)
    - Mean PhysioNet Score: 0.0487 (unoptimized threshold)
- right_body:
  - **XGBoost (Selected Classifier)**:
    - Natively handles NaNs during tree splitting without manual imputation bias
    - Balanced minority class scale weights (`scale_pos_weight`)
    - Optimized decision threshold via validation fold grid search
    - Mean AUROC: **0.8475 (±0.0137)**
    - Mean PhysioNet Score: **0.4666**
    - Optimal decision threshold: **0.6600**

## slide_layout: image_content
- title: Independent Test Set B Evaluation
- body:
  - Evaluated on the independent validation cohort Set B (4,000 patient records) to assess generalizability
  - Test Accuracy: **87.30%** | Test AUROC: **0.8612** (indicating robust discrimination)
  - PhysioNet Challenge Score: **0.4137** (Sensitivity: 41.37%, Precision: 57.32%)
  - Confusion Matrix: TN = 3,257, FP = 175, FN = 333, TP = 235
- image: figures/roc_curves.png

## slide_layout: image_content
- title: Key Physiological Predictors
- body:
  - **Glasgow Coma Score (GCS)**: Minimum (`GCS_min`) and last (`GCS_last`) values are the most critical predictors, reflecting neurological function
  - **Age**: Older patients have lower physiological reserve, correlating with high risk
  - **Blood Urea Nitrogen (BUN)**: Mean levels (`BUN_mean`) reflect renal function and fluid balance
  - **Urine Output**: Mean urine output (`Urine_mean`) is a crucial marker for tissue perfusion
  - **White Blood Cell (WBC)**: High counts (`WBC_mean`) indicate systemic inflammation or sepsis
- image: figures/feature_importance.png

## slide_layout: section
- title: 5. Interactive Clinical Dashboard

## slide_layout: image_content
- title: Streamlit Clinical Dashboard Platform
- body:
  - **Sidebar Controls & Citations**: Patient selectors and profile banners moved to the sidebar along with clear data source attributions
  - **Longitudinal Explorer**: High-contrast vitals cards with auto-height layout, Plotly Gauge risk scores with suffix and font constraint, and scrollable alert panels
  - **Risk Simulator**: Grouped 5x2 physiological sliders with side-by-side gauge predictions and clinical recommendations, updating in real-time on a single page
- image: figures/dashboard_mockup.png

## slide_layout: image_content
- title: Production Safeguards: Dynamic Feature Alignment
- body:
  - XGBoost models restrict the ordering and dimensions of features during live predictions
  - The simulator checks the trained model's feature_names_in_ dynamically at runtime
  - Aligns and formats input vectors to ensure robust predictions without column mismatch errors
  - Safeguards live clinical deployment against feature shifts or missing values
- image: outputs/code_model_alignment.png

## slide_layout: section
- title: 6. Conclusion & Future Work

## slide_layout: content
- title: Key Takeaways & Next Steps
- body:
  - **Conclusion**:
    - Developed an end-to-end Python pipeline from raw data ingestion to interactive web UI
    - XGBoost model outperformed linear and traditional ensemble approaches, achieving AUROC of 0.8612
    - The clinical risk simulator bridges the gap between predictive ML models and clinical utility
  - **Future Work**:
    - Explore deep learning recurrent architectures (e.g., LSTMs, GRUs, Transformers) to process raw time-series directly
    - Validate model generalizability using external databases (e.g., MIMIC-IV) and multi-center cohorts
