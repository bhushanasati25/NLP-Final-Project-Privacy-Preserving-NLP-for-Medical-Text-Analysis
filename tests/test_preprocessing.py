import unittest
import pandas as pd
import numpy as np
from src.preprocessing.data_processor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.preprocessor = DataPreprocessor()
        self.sample_text = "Patient shows symptoms of COVID-19: fever (39.5°C) and cough!"
        self.sample_df = pd.DataFrame({
            'text': [
                "Patient has fever and cough",
                "Normal checkup, no symptoms",
                "COVID-19 test results: positive"
            ]
        })

    def test_clean_text(self):
        """Test text cleaning functionality"""
        cleaned_text = self.preprocessor.clean_text(self.sample_text)
        
        # Check lowercase conversion
        self.assertFalse(any(c.isupper() for c in cleaned_text))
        
        # Check special character removal
        self.assertFalse(any(not c.isalnum() and not c.isspace() for c in cleaned_text))
        
        # Check number removal
        self.assertFalse(any(c.isdigit() for c in cleaned_text))
        
        # Check stopword removal
        self.assertFalse(' of ' in cleaned_text)
        self.assertFalse(' and ' in cleaned_text)

    def test_vectorization(self):
        """Test TF-IDF vectorization"""
        _, features = self.preprocessor.process_data(self.sample_df)
        
        # Check feature matrix dimensions
        self.assertEqual(features.shape[0], len(self.sample_df))
        self.assertLessEqual(features.shape[1], 10000)  # max_features constraint
        
        # Check if features are normalized
        feature_norms = np.linalg.norm(features.toarray(), axis=1)
        np.testing.assert_array_almost_equal(feature_norms, np.ones_like(feature_norms))

    def test_empty_text(self):
        """Test handling of empty text"""
        empty_text = ""
        cleaned_text = self.preprocessor.clean_text(empty_text)
        self.assertEqual(cleaned_text, "")

    def test_special_characters(self):
        """Test handling of special characters"""
        special_text = "!@#$%^&*()_+"
        cleaned_text = self.preprocessor.clean_text(special_text)
        self.assertEqual(cleaned_text, "")

    def test_data_processing(self):
        """Test complete data processing pipeline"""
        processed_df, features = self.preprocessor.process_data(self.sample_df)
        
        # Check if DataFrame was processed correctly
        self.assertIn('cleaned_text', processed_df.columns)
        self.assertEqual(len(processed_df), len(self.sample_df))
        
        # Check feature matrix
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], len(processed_df))

    def test_numeric_values(self):
        """Test handling of numeric values"""
        numeric_text = "Temperature is 39.5 degrees"
        cleaned_text = self.preprocessor.clean_text(numeric_text)
        self.assertNotIn("39.5", cleaned_text)
        self.assertIn("temperature", cleaned_text)
        self.assertIn("degrees", cleaned_text)

if __name__ == '__main__':
    unittest.main()
