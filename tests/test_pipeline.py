import unittest
import tempfile
import os
import numpy as np
from src.data_loader import PatientRecord
from src.features import FeatureExtractor
from src.utils import timer, logger

class TestPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create a mock patient record file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        self.temp_file.write("Time,Parameter,Value\n")
        self.temp_file.write("00:00,RecordID,132539\n")
        self.temp_file.write("00:00,Age,65\n")
        self.temp_file.write("00:00,Gender,1\n")
        self.temp_file.write("00:10,HR,80\n")
        self.temp_file.write("01:20,HR,95\n")
        self.temp_file.write("02:30,Temp,37.5\n")
        self.temp_file.write("03:40,GCS,15\n")
        self.temp_file.close()
        
    def tearDown(self):
        os.remove(self.temp_file.name)

    def test_patient_record_parsing(self):
        """Test that PatientRecord loads and parses descriptors and time series correctly."""
        record = PatientRecord(self.temp_file.name)
        
        self.assertEqual(record.record_id, 132539)
        self.assertEqual(record.descriptors['Age'], 65.0)
        self.assertEqual(record.descriptors['Gender'], 1.0)
        
        # Test time series values
        hr_series = record.time_series['HR']
        self.assertEqual(len(hr_series), 2)
        # Check elapsed minutes calculation: "00:10" -> 10, "01:20" -> 80
        self.assertEqual(hr_series[0], (10, 80.0))
        self.assertEqual(hr_series[1], (80, 95.0))
        
        temp_series = record.time_series['Temp']
        self.assertEqual(len(temp_series), 1)
        self.assertEqual(temp_series[0], (150, 37.5)) # "02:30" -> 150 min

    def test_timer_decorator(self):
        """Test that the @timer decorator preserves function execution and runs without errors."""
        @timer
        def dummy_function(x):
            return x * 2
            
        result = dummy_function(10)
        self.assertEqual(result, 20)
        self.assertEqual(dummy_function.__name__, "dummy_function")

    def test_feature_extractor(self):
        """Test that FeatureExtractor correctly aggregates statistical metrics."""
        record = PatientRecord(self.temp_file.name)
        extractor = FeatureExtractor()
        features = extractor.extract_features(record)
        
        # Verify RecordID and metadata
        self.assertEqual(features['RecordID'], 132539)
        self.assertEqual(features['Age'], 65.0)
        self.assertEqual(features['Gender'], 1.0)
        
        # Verify HR statistics: values are [80.0, 95.0]
        self.assertEqual(features['HR_min'], 80.0)
        self.assertEqual(features['HR_max'], 95.0)
        self.assertEqual(features['HR_mean'], 87.5)
        self.assertEqual(features['HR_last'], 95.0)
        self.assertEqual(features['HR_count'], 2)
        
        # Verify GCS statistics: single value [15.0]
        self.assertEqual(features['GCS_min'], 15.0)
        self.assertEqual(features['GCS_max'], 15.0)
        self.assertEqual(features['GCS_mean'], 15.0)
        self.assertEqual(features['GCS_last'], 15.0)
        self.assertEqual(features['GCS_count'], 1)
        
        # Verify parameter with no values (should be NaN)
        self.assertTrue(np.isnan(features['Albumin_mean']))
        self.assertEqual(features['Albumin_count'], 0)

if __name__ == '__main__':
    unittest.main()
