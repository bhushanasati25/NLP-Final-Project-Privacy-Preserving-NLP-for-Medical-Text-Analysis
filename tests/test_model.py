import unittest
import torch
import numpy as np
from src.models.text_classifier import SimpleTextClassifier

class TestTextClassifier(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.input_dim = 100
        self.batch_size = 32
        self.hidden_dim = 64
        self.output_dim = 5
        
        # Initialize model
        self.model = SimpleTextClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        
        # Create sample input
        self.sample_input = torch.randn(self.batch_size, self.input_dim)

    def test_model_initialization(self):
        """Test model initialization"""
        # Check layer dimensions
        self.assertEqual(self.model.layer1.in_features, self.input_dim)
        self.assertEqual(self.model.layer1.out_features, self.hidden_dim)
        self.assertEqual(self.model.layer2.in_features, self.hidden_dim)
        self.assertEqual(self.model.layer2.out_features, self.hidden_dim // 2)
        self.assertEqual(self.model.layer3.out_features, self.output_dim)

    def test_forward_pass(self):
        """Test forward pass"""
        # Run forward pass
        output = self.model(self.sample_input)
        
        # Check output dimensions
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.output_dim)
        
        # Check output values
        self.assertTrue(torch.isfinite(output).all())

    def test_dropout(self):
        """Test dropout behavior"""
        self.model.train()  # Set to training mode
        output1 = self.model(self.sample_input)
        output2 = self.model(self.sample_input)
        
        # Outputs should be different during training due to dropout
        self.assertFalse(torch.equal(output1, output2))
        
        self.model.eval()  # Set to evaluation mode
        output1 = self.model(self.sample_input)
        output2 = self.model(self.sample_input)
        
        # Outputs should be same during evaluation
        self.assertTrue(torch.equal(output1, output2))

    def test_gradients(self):
        """Test gradient computation"""
        self.model.train()
        output = self.model(self.sample_input)
        loss = output.sum()
        loss.backward()
        
        # Check if gradients were computed
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())

    def test_different_batch_sizes(self):
        """Test model with different batch sizes"""
        batch_sizes = [1, 16, 64, 128]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, self.input_dim)
            output = self.model(input_tensor)
            
            self.assertEqual(output.shape[0], batch_size)
            self.assertEqual(output.shape[1], self.output_dim)

    def test_model_parameters(self):
        """Test model parameters"""
        # Count number of parameters
        total_params = sum(
            p.numel() for p in self.model.parameters()
        )
        
        # Calculate expected number of parameters
        expected_params = (
            self.input_dim * self.hidden_dim +  # layer1 weights
            self.hidden_dim +  # layer1 bias
            self.hidden_dim * (self.hidden_dim // 2) +  # layer2 weights
            (self.hidden_dim // 2) +  # layer2 bias
            (self.hidden_dim // 2) * self.output_dim +  # layer3 weights
            self.output_dim  # layer3 bias
        )
        
        self.assertEqual(total_params, expected_params)

    def test_output_range(self):
        """Test output range before activation"""
        self.model.eval()
        output = self.model(self.sample_input)
        
        # Check if outputs are in reasonable range
        self.assertTrue(output.abs().max() < 100)

    def test_model_gpu(self):
        """Test model on GPU if available"""
        if torch.cuda.is_available():
            model = self.model.cuda()
            input_tensor = self.sample_input.cuda()
            
            output = model(input_tensor)
            
            self.assertTrue(output.is_cuda)
            self.assertEqual(output.shape[1], self.output_dim)

if __name__ == '__main__':
    unittest.main()