import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from src.utils import logger, timer
from src.features import FeatureExtractor
from src.model import ICUPredictor, calculate_physionet_score

@timer
def run_data_ingestion():
    """
    Downloads dataset files if they don't already exist.
    """
    data_dir = "data"
    set_a_dir = os.path.join(data_dir, "set-a")
    outcome_a_path = os.path.join(data_dir, "Outcomes-a.txt")
    
    if not (os.path.exists(set_a_dir) and os.path.exists(outcome_a_path)):
        logger.info("Raw dataset files not found. Running downloader...")
        import subprocess
        subprocess.run(["python", "src/data_downloader.py"], check=True)
    else:
        logger.info("Raw dataset files verified in workspace.")

@timer
def run_feature_extraction(extractor, set_dir, outcome_path, cache_path, num_workers):
    """
    Extracts features and caches them to disk.
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading cached feature matrix from {cache_path}...")
        df = pd.read_csv(cache_path)
    else:
        df = extractor.build_dataset(set_dir, outcome_path, num_workers=num_workers)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached extracted features to {cache_path}")
    return df

@timer
def run_model_validation(predictor, X, y, cv_folds):
    """
    Runs cross-validation.
    """
    return predictor.evaluate_cv(X, y, cv_folds=cv_folds)

@timer
def run_model_training(predictor, X, y, save_path):
    """
    Trains the final model pipeline and serializes it.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    predictor.fit(X, y)
    predictor.save(save_path)
    return predictor

def plot_roc_curve(pipeline_dict, X_train, y_train, X_test, y_test, save_path):
    """
    Plots the receiver operating characteristic curve for comparison.
    """
    from sklearn.metrics import roc_curve, auc
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model_name = pipeline_dict['model_name']
    model = pipeline_dict['model']
    
    if model_name in ["Logistic Regression", "Random Forest"]:
        imputer = pipeline_dict['imputer']
        scaler = pipeline_dict['scaler']
        X_tr = imputer.transform(X_train)
        if scaler:
            X_tr = scaler.transform(X_tr)
        X_te = imputer.transform(X_test)
        if scaler:
            X_te = scaler.transform(X_te)
            
        y_tr_prob = model.predict_proba(X_tr)[:, 1]
        y_te_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_tr_prob = model.predict_proba(X_train)[:, 1]
        y_te_prob = model.predict_proba(X_test)[:, 1]
        
    fpr_tr, tpr_tr, _ = roc_curve(y_train, y_tr_prob)
    fpr_te, tpr_te, _ = roc_curve(y_test, y_te_prob)
    
    auc_tr = auc(fpr_tr, tpr_tr)
    auc_te = auc(fpr_te, tpr_te)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_tr, tpr_tr, color='darkorange', lw=2, label=f'Train ROC (AUC = {auc_tr:.4f})')
    plt.plot(fpr_te, tpr_te, color='navy', lw=2, label=f'Test ROC (AUC = {auc_te:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC Curve plot saved to {save_path}")

def plot_feature_importance(pipeline_dict, feature_names, save_path, top_n=20):
    """
    Plots the top N features by classification importance.
    """
    model_name = pipeline_dict['model_name']
    model = pipeline_dict['model']
    
    if model_name == "Logistic Regression":
        importances = np.abs(model.coef_[0])
    elif model_name in ["Random Forest", "XGBoost"]:
        importances = model.feature_importances_
    else:
        return
        
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title(f'Top {top_n} Feature Importances - {model_name}')
    plt.xlabel('Relative Importance' if model_name != "Logistic Regression" else 'Absolute Coefficient')
    plt.ylabel('Feature Name')
    plt.grid(True, alpha=0.3, axis='x')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Advanced ICU Mortality Prediction Training & Validation Pipeline")
    parser.add_argument("--model", type=str, default="XGBoost",
                        choices=["Logistic Regression", "Random Forest", "XGBoost"],
                        help="Machine learning classifier to develop.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent worker processes for feature extraction.")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of cross-validation folds.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional fixed decision threshold. If omitted, search for the best on validation sets.")
    args = parser.parse_args()
    
    logger.info("=== Starting Advanced ICU Mortality Prediction Pipeline ===")
    logger.info(f"Selected Model: {args.model}")
    logger.info(f"Parallel Workers: {args.workers}")
    logger.info(f"CV Folds: {args.cv_folds}")
    
    # 1. Ingest Data
    run_data_ingestion()
    
    # 2. Extract features
    data_dir = "data"
    set_a_dir = os.path.join(data_dir, "set-a")
    set_b_dir = os.path.join(data_dir, "set-b")
    outcome_a_path = os.path.join(data_dir, "Outcomes-a.txt")
    outcome_b_path = os.path.join(data_dir, "Outcomes-b.txt")
    
    feature_extractor = FeatureExtractor()
    
    df_train = run_feature_extraction(
        feature_extractor, set_a_dir, outcome_a_path, 
        os.path.join(data_dir, "features_set_a.csv"), args.workers
    )
    df_test = run_feature_extraction(
        feature_extractor, set_b_dir, outcome_b_path, 
        os.path.join(data_dir, "features_set_b.csv"), args.workers
    )
    
    meta_cols = ['RecordID', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death']
    feature_cols = [c for c in df_train.columns if c not in meta_cols]
    
    X_train = df_train[feature_cols]
    y_train = df_train['In-hospital_death']
    X_test = df_test[feature_cols]
    y_test = df_test['In-hospital_death']
    
    logger.info(f"Features dimension: {len(feature_cols)}")
    logger.info(f"Train size: {X_train.shape[0]} patients | Test size: {X_test.shape[0]} patients")
    
    # 3. Model Evaluation (Cross-Validation)
    predictor = ICUPredictor(args.model)
    run_model_validation(predictor, X_train, y_train, args.cv_folds)
    
    # 4. Final Model Training & Export
    model_save_path = "outputs/best_model.joblib"
    run_model_training(predictor, X_train, y_train, model_save_path)
    
    # Override optimal threshold if custom threshold was passed
    if args.threshold is not None:
        logger.info(f"Overriding optimal threshold with custom: {args.threshold:.4f}")
        predictor.threshold = args.threshold
        # Re-save to overwrite threshold
        predictor.save(model_save_path)
        
    # 5. Evaluate on Test Set B
    pipeline_dict = predictor.export_pipeline_dict()
    model = pipeline_dict['model']
    
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= predictor.threshold).astype(int)
    
    acc = accuracy_score(y_test, y_test_pred)
    auc_val = roc_auc_score(y_test, y_test_prob)
    p_score, sens, prec = calculate_physionet_score(y_test, y_test_pred)
    
    logger.info(f"\n=== Test Set B Evaluation Result ({args.model}) ===")
    logger.info(f"Accuracy: {acc:.4%}")
    logger.info(f"AUROC: {auc_val:.4f}")
    logger.info(f"PhysioNet Score: {p_score:.4f} (Sensitivity: {sens:.4f}, Precision: {prec:.4f})")
    
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # 6. Save visualizations & report
    plot_roc_curve(pipeline_dict, X_train, y_train, X_test, y_test, "outputs/roc_curves.png")
    plot_feature_importance(pipeline_dict, feature_cols, "outputs/feature_importance.png", top_n=20)
    
    with open("outputs/evaluation_summary.txt", "w") as f:
        f.write("=== PhysioNet Challenge 2012 Model Evaluation ===\n")
        f.write(f"Best Model: {args.model}\n")
        f.write(f"Decision Threshold: {predictor.threshold:.4f}\n\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test AUROC: {auc_val:.4f}\n")
        f.write(f"Test Challenge Score: {p_score:.4f}\n")
        f.write(f"  - Sensitivity: {sens:.4f}\n")
        f.write(f"  - Precision: {prec:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TN: {cm[0,0]} | FP: {cm[0,1]}\n")
        f.write(f"FN: {cm[1,0]} | TP: {cm[1,1]}\n")
        
    logger.info("Training and evaluation pipeline completed successfully!")

if __name__ == "__main__":
    main()
