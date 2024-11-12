import unittest
from src.preprocessing.data_processor import DataPreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
    
    def test_clean_text(self):
        text = "This is a TEST123 text!"
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "test text")

# tests/test_federated.py
import unittest
import torch
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer

class TestFederated(unittest.TestCase):
    def test_model_aggregation(self):
        # Add test cases for federated learning
        pass

# tests/test_model.py
import unittest
import torch
from src.models.text_classifier import SimpleTextClassifier

class TestModel(unittest.TestCase):
    def test_forward_pass(self):
        model = SimpleTextClassifier(input_dim=100)
        x = torch.randn(1, 100)
        output = model(x)
        self.assertEqual(output.shape[1], 5)