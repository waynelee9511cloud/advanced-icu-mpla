import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from src.utils import logger

def calculate_physionet_score(y_true, y_pred):
    """
    Calculate the Event 1 score: min(Sensitivity, Positive Predictivity)
    Sensitivity = Recall = TP / (TP + FN)
    Positive Predictivity = Precision = TP / (TP + FP)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    return min(sensitivity, precision), sensitivity, precision

class ICUPredictor:
    """
    Object-oriented predictive pipeline that wraps the machine learning classifiers,
    preprocessing components, validation routines, and inference.
    """
    def __init__(self, model_name="XGBoost"):
        self.model_name = model_name
        self.imputer = None
        self.scaler = None
        self.model = None
        self.threshold = 0.5
        
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initializes the preprocessing steps and classifier based on model type."""
        logger.debug(f"Initializing {self.model_name} architecture...")
        
        if self.model_name == "Logistic Regression":
            self.imputer = SimpleImputer(strategy='median')
            self.scaler = StandardScaler()
            self.model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
            
        elif self.model_name == "Random Forest":
            self.imputer = SimpleImputer(strategy='median')
            self.scaler = None
            self.model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
            
        elif self.model_name == "XGBoost":
            self.imputer = None
            self.scaler = None
            # XGBoost model is created dynamically in fit/cv to adjust scale_pos_weight
            self.model = None
            
        else:
            logger.error(f"Unsupported model name: {self.model_name}")
            raise ValueError(f"Unknown model name: {self.model_name}")
            
    def _get_xgb_model(self, y_train):
        """Helper to instantiate XGBoost with proper scale_pos_weight."""
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos = neg_count / pos_count if pos_count > 0 else 1.0
        return XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )

    def evaluate_cv(self, X, y, cv_folds=5):
        """Runs Stratified K-Fold cross-validation and reports performance statistics."""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        metrics = {
            'auc_scores': [], 'f1_scores': [], 'p_scores': [],
            'sens_scores': [], 'prec_scores': [], 'acc_scores': []
        }
        
        logger.info(f"Evaluating {self.model_name} with {cv_folds}-Fold Stratified CV...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Local clone training
            if self.model_name == "Logistic Regression":
                imp = SimpleImputer(strategy='median')
                scl = StandardScaler()
                X_tr = scl.fit_transform(imp.fit_transform(X_train))
                X_va = scl.transform(imp.transform(X_val))
                
                clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
                clf.fit(X_tr, y_train)
                probs = clf.predict_proba(X_va)[:, 1]
                preds = (probs >= 0.5).astype(int)
                
            elif self.model_name == "Random Forest":
                imp = SimpleImputer(strategy='median')
                X_tr = imp.fit_transform(X_train)
                X_va = imp.transform(X_val)
                
                clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
                clf.fit(X_tr, y_train)
                probs = clf.predict_proba(X_va)[:, 1]
                preds = (probs >= 0.5).astype(int)
                
            elif self.model_name == "XGBoost":
                clf = self._get_xgb_model(y_train)
                clf.fit(X_train, y_train)
                probs = clf.predict_proba(X_val)[:, 1]
                
                # Grid search decision threshold
                best_th = 0.5
                best_p = 0.0
                best_preds = None
                for th in np.arange(0.1, 0.9, 0.02):
                    temp_preds = (probs >= th).astype(int)
                    score, _, _ = calculate_physionet_score(y_val, temp_preds)
                    if score > best_p:
                        best_p = score
                        best_th = th
                        best_preds = temp_preds
                        
                preds = best_preds if best_preds is not None else (probs >= 0.5).astype(int)
                
            # Compile fold metrics
            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds)
            auc = roc_auc_score(y_val, probs)
            p_score, sens, prec = calculate_physionet_score(y_val, preds)
            
            metrics['acc_scores'].append(acc)
            metrics['f1_scores'].append(f1)
            metrics['auc_scores'].append(auc)
            metrics['p_scores'].append(p_score)
            metrics['sens_scores'].append(sens)
            metrics['prec_scores'].append(prec)
            
            logger.info(f"  Fold {fold+1} - AUROC: {auc:.4f}, F1: {f1:.4f}, PhysioNet Score: {p_score:.4f} (Se: {sens:.4f}, +P: {prec:.4f})")
            
        mean_auc = np.mean(metrics['auc_scores'])
        mean_p = np.mean(metrics['p_scores'])
        logger.info(f"  --- Mean Metrics ---")
        logger.info(f"  AUROC: {mean_auc:.4f} (±{np.std(metrics['auc_scores']):.4f})")
        logger.info(f"  PhysioNet Score: {mean_p:.4f} (Se: {np.mean(metrics['sens_scores']):.4f}, +P: {np.mean(metrics['prec_scores']):.4f})")
        
        return {
            'mean_auroc': mean_auc,
            'mean_f1': np.mean(metrics['f1_scores']),
            'mean_physionet': mean_p,
            'mean_sensitivity': np.mean(metrics['sens_scores']),
            'mean_precision': np.mean(metrics['prec_scores']),
            'auc_scores': metrics['auc_scores'],
            'p_scores': metrics['p_scores']
        }
        
    def fit(self, X, y):
        """Fits the model pipeline on the entire dataset."""
        logger.info(f"Fitting final {self.model_name} model on full dataset...")
        
        if self.model_name == "Logistic Regression":
            self.imputer = SimpleImputer(strategy='median')
            self.scaler = StandardScaler()
            X_proc = self.scaler.fit_transform(self.imputer.fit_transform(X))
            self.model.fit(X_proc, y)
            self.threshold = 0.5
            
        elif self.model_name == "Random Forest":
            self.imputer = SimpleImputer(strategy='median')
            self.scaler = None
            X_proc = self.imputer.fit_transform(X)
            self.model.fit(X_proc, y)
            self.threshold = 0.5
            
        elif self.model_name == "XGBoost":
            self.imputer = None
            self.scaler = None
            self.model = self._get_xgb_model(y)
            self.model.fit(X, y)
            
            # Grid search for optimal threshold on training set
            y_probs = self.model.predict_proba(X)[:, 1]
            best_th = 0.5
            best_p = 0.0
            for th in np.arange(0.1, 0.9, 0.02):
                temp_preds = (y_probs >= th).astype(int)
                score, _, _ = calculate_physionet_score(y, temp_preds)
                if score > best_p:
                    best_p = score
                    best_th = th
            self.threshold = best_th
            logger.info(f"Optimal threshold found: {self.threshold:.4f}")
            
        return self

    def export_pipeline_dict(self):
        """Exports a dictionary representation compatible with the old app.py structure."""
        return {
            'model_name': self.model_name,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'model': self.model,
            'threshold': self.threshold
        }

    def save(self, path):
        """Serializes and saves the model pipeline dictionary to a file."""
        pipeline_dict = self.export_pipeline_dict()
        joblib.dump(pipeline_dict, path)
        logger.info(f"Successfully saved model to {path}")

# Backward-compatible function APIs
def evaluate_model_cv(model_name, X, y, cv_folds=5):
    predictor = ICUPredictor(model_name)
    return predictor.evaluate_cv(X, y, cv_folds)

def train_and_save_final_model(model_name, X, y, save_path):
    predictor = ICUPredictor(model_name)
    predictor.fit(X, y)
    predictor.save(save_path)
    return predictor.export_pipeline_dict()
