import unittest
import torch
import numpy as np
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.models.text_classifier import SimpleTextClassifier
from torch.utils.data import TensorDataset

class TestFederatedLearning(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create sample data
        self.input_dim = 100
        self.num_samples = 50
        self.num_clients = 3
        
        # Initialize model
        self.model = SimpleTextClassifier(
            input_dim=self.input_dim,
            hidden_dim=64,
            output_dim=5
        )
        
        # Create sample datasets
        self.client_datasets = self.create_sample_datasets()
        
        # Initialize server
        self.server = FederatedServer(self.model, self.num_clients)

    def create_sample_datasets(self):
        """Create sample datasets for testing"""
        datasets = []
        for _ in range(self.num_clients):
            features = torch.randn(self.num_samples, self.input_dim)
            labels = torch.randint(0, 5, (self.num_samples,))
            datasets.append(TensorDataset(features, labels))
        return datasets

    def test_client_training(self):
        """Test client-side training"""
        client = FederatedClient(
            model=self.model,
            dataset=self.client_datasets[0],
            client_id=0
        )
        
        # Train client model
        initial_params = {k: v.clone() for k, v in self.model.state_dict().items()}
        client.train(epochs=2, batch_size=16)
        final_params = client.model.state_dict()
        
        # Check if parameters were updated
        for key in initial_params:
            self.assertFalse(torch.equal(initial_params[key], final_params[key]))

    def test_model_aggregation(self):
        """Test server-side model aggregation"""
        # Train multiple clients
        client_models = []
        for i in range(self.num_clients):
            client = FederatedClient(
                model=SimpleTextClassifier(self.input_dim),
                dataset=self.client_datasets[i],
                client_id=i
            )
            client.train(epochs=1)
            client_models.append(client.model.state_dict())
        
        # Aggregate models
        initial_params = {k: v.clone() for k, v in self.server.global_model.state_dict().items()}
        self.server.aggregate_models(client_models)
        final_params = self.server.global_model.state_dict()
        
        # Check if parameters were updated
        for key in initial_params:
            self.assertFalse(torch.equal(initial_params[key], final_params[key]))

    def test_federated_round(self):
        """Test complete federated learning round"""
        # Initialize clients
        clients = [
            FederatedClient(
                model=SimpleTextClassifier(self.input_dim),
                dataset=dataset,
                client_id=i
            )
            for i, dataset in enumerate(self.client_datasets)
        ]
        
        # Run one federated round
        client_updates = []
        for client in clients:
            client.train(epochs=1)
            client_updates.append(client.model.state_dict())
        
        # Aggregate updates
        self.server.aggregate_models(client_updates)
        
        # Check if global model was updated
        self.assertIsNotNone(self.server.global_model.state_dict())

    def test_model_distribution(self):
        """Test model distribution to clients"""
        global_model = self.server.global_model
        client = FederatedClient(
            model=SimpleTextClassifier(self.input_dim),
            dataset=self.client_datasets[0],
            client_id=0
        )
        
        # Check if client model matches global model
        for key in global_model.state_dict():
            self.assertTrue(
                torch.equal(
                    global_model.state_dict()[key],
                    client.model.state_dict()[key]
                )
            )

    def test_privacy_preservation(self):
        """Test basic privacy preservation"""
        client = FederatedClient(
            model=self.model,
            dataset=self.client_datasets[0],
            client_id=0
        )
        
        # Train client model
        client.train(epochs=1)
        
        # Check if raw data is not accessible
        self.assertFalse(hasattr(client, 'raw_data'))

if __name__ == '__main__':
    unittest.main()