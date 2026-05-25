import os
import numpy as np
from src.utils import logger

# List of all 37 time-series parameters in PhysioNet 2012
TIME_SERIES_PARAMS = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine',
    'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
    'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH',
    'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TropI', 'TropT', 'Urine',
    'WBC', 'Weight'
]

# List of 6 general descriptors collected at admission (Time = 00:00)
GENERAL_DESCRIPTORS = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']

class PatientRecord:
    """
    Object-oriented representation of an ICU patient record.
    Encapsulates patient descriptors and raw physiological time-series data.
    """
    
    def __init__(self, file_path=None):
        self.record_id = None
        self.descriptors = {
            'RecordID': None,
            'Age': np.nan,
            'Gender': np.nan,
            'Height': np.nan,
            'ICUType': np.nan,
            'Weight': np.nan
        }
        self.time_series = {param: [] for param in TIME_SERIES_PARAMS}
        
        if file_path:
            self.load_from_file(file_path)
            
    def __repr__(self):
        return (f"<PatientRecord ID={self.record_id} "
                f"Age={self.descriptors.get('Age')} "
                f"Gender={self.descriptors.get('Gender')} "
                f"ICUType={self.descriptors.get('ICUType')}>")
                
    def _time_to_minutes(self, time_str):
        """Convert HH:MM string to elapsed minutes since admission."""
        try:
            parts = time_str.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        except ValueError as e:
            logger.warning(f"Failed to parse time string '{time_str}': {e}")
        return 0

    def load_from_file(self, file_path):
        """
        Parses the PhysioNet patient record file.
        """
        if not os.path.exists(file_path):
            logger.error(f"Patient file not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.debug(f"Parsing patient file: {file_path}")
        
        with open(file_path, 'r') as f:
            # Check for header
            first_line = f.readline().strip()
            if not first_line.startswith("Time"):
                # If first line is data, process it
                f.seek(0)
                
            for line_num, line in enumerate(f, start=2):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) != 3:
                    logger.debug(f"Invalid format at {file_path}:{line_num}: '{line}'")
                    continue
                    
                time_str, param, val_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
                
                try:
                    val = float(val_str)
                except ValueError:
                    logger.debug(f"Non-numeric value '{val_str}' at {file_path}:{line_num}")
                    continue
                    
                # In PhysioNet 2012, -1 signifies missing data in some cases
                if val == -1:
                    val = np.nan
                    
                time_min = self._time_to_minutes(time_str)
                
                # Baseline descriptors are recorded at Time = 00:00
                if time_min == 0 and param in GENERAL_DESCRIPTORS:
                    if param == 'RecordID':
                        self.descriptors['RecordID'] = int(val)
                        self.record_id = int(val)
                    else:
                        self.descriptors[param] = val
                elif param in TIME_SERIES_PARAMS:
                    if not np.isnan(val):
                        self.time_series[param].append((time_min, val))
                        
        # Fallback to infer RecordID from filename if missing from the file body
        if self.record_id is None:
            try:
                filename = os.path.basename(file_path)
                self.record_id = int(filename.split('.')[0])
                self.descriptors['RecordID'] = self.record_id
            except ValueError:
                logger.warning(f"Could not infer RecordID from filename: {file_path}")
                
        logger.debug(f"Successfully loaded PatientRecord {self.record_id}")
        return self

def parse_patient_file(file_path):
    """
    Backward-compatible function API that returns (descriptors, time_series).
    """
    record = PatientRecord(file_path)
    return record.descriptors, record.time_series
