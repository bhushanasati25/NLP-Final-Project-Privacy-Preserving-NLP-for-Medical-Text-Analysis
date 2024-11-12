import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model, test_data, label_encoder):
        self.model = model
        self.test_data = test_data
        self.label_encoder = label_encoder
    
    def evaluate(self):
        """Evaluate model performance"""
        predictions = []
        true_labels = []
        
        for features, labels in self.test_data:
            outputs = self.model(features)
            preds = outputs.argmax(dim=1)
            predictions.extend(preds.numpy())
            true_labels.extend(labels.numpy())
        
        # Calculate metrics
        report = classification_report(
            true_labels, 
            predictions,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        return report, conf_matrix
    
    def plot_confusion_matrix(self, conf_matrix):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()