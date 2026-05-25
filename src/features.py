import os
import pandas as pd
import numpy as np
import concurrent.futures
from src.data_loader import PatientRecord, TIME_SERIES_PARAMS
from src.utils import logger

class FeatureExtractor:
    """
    Class responsible for extracting physiological aggregates and building 
    structured training/validation datasets.
    """
    
    def extract_features(self, record: PatientRecord) -> dict:
        """
        Extracts aggregated features for a single patient record.
        """
        features = {
            'RecordID': record.record_id,
            'Age': record.descriptors['Age'],
            'Gender': record.descriptors['Gender'],
            'Height': record.descriptors['Height'],
            'ICUType': record.descriptors['ICUType'],
            'Weight_admit': record.descriptors['Weight']
        }
        
        for param in TIME_SERIES_PARAMS:
            observations = record.time_series[param]
            
            if not observations:
                features[f'{param}_mean'] = np.nan
                features[f'{param}_min'] = np.nan
                features[f'{param}_max'] = np.nan
                features[f'{param}_last'] = np.nan
                features[f'{param}_count'] = 0
                continue
                
            times, values = zip(*observations)
            values = np.array(values)
            
            features[f'{param}_mean'] = np.mean(values)
            features[f'{param}_min'] = np.min(values)
            features[f'{param}_max'] = np.max(values)
            features[f'{param}_last'] = values[-1]
            features[f'{param}_count'] = len(values)
            
        return features

    def build_dataset(self, data_dir: str, outcome_file_path: str = None, num_workers: int = 1) -> pd.DataFrame:
        """
        Builds the complete feature dataset for all patient records in a directory.
        Supports parallel feature extraction using a ProcessPoolExecutor.
        """
        if not os.path.exists(data_dir):
            logger.error(f"Directory {data_dir} does not exist.")
            raise ValueError(f"Directory {data_dir} does not exist.")
            
        txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        file_paths = [os.path.join(data_dir, f) for f in txt_files]
        total_files = len(file_paths)
        
        logger.info(f"Extracting features from {total_files} files in {data_dir} (workers={num_workers})...")
        
        patient_features_list = []
        
        if num_workers > 1:
            # Parallel extraction using ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit tasks
                future_to_path = {executor.submit(_parse_and_extract_helper, path): path for path in file_paths}
                
                completed = 0
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        features = future.result()
                        patient_features_list.append(features)
                    except Exception as exc:
                        logger.error(f"File {path} generated an exception: {exc}")
                        
                    completed += 1
                    if completed % 500 == 0 or completed == total_files:
                        logger.info(f"Processed {completed}/{total_files} patient records...")
        else:
            # Sequential extraction
            for i, file_path in enumerate(file_paths):
                try:
                    record = PatientRecord(file_path)
                    features = self.extract_features(record)
                    patient_features_list.append(features)
                except Exception as exc:
                    logger.error(f"File {file_path} generated an exception: {exc}")
                    
                if (i + 1) % 500 == 0 or (i + 1) == total_files:
                    logger.info(f"Processed {i + 1}/{total_files} patient records...")
                    
        df_features = pd.DataFrame(patient_features_list)
        
        # Merge outcomes/labels if provided
        if outcome_file_path and os.path.exists(outcome_file_path):
            logger.info(f"Merging outcome labels from {outcome_file_path}...")
            df_outcomes = pd.read_csv(outcome_file_path)
            df_features = pd.merge(df_features, df_outcomes, on='RecordID', how='inner')
            logger.info(f"Dataset shape after merging labels: {df_features.shape}")
            
        return df_features

# Top-level helper function for multiprocessing pickling on Windows
def _parse_and_extract_helper(file_path):
    record = PatientRecord(file_path)
    extractor = FeatureExtractor()
    return extractor.extract_features(record)

# Backward-compatible function API
def build_feature_dataset(data_dir, outcome_file_path=None):
    extractor = FeatureExtractor()
    return extractor.build_dataset(data_dir, outcome_file_path, num_workers=1)
