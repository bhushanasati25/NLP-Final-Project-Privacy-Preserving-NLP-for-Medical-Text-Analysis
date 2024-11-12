from flask import Flask, request, jsonify
import torch
import numpy as np

class ModelDeployment:
    def __init__(self, model, vectorizer, label_encoder):
        self.app = Flask(__name__)
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.get_json()
                text = data['text']
                
                # Preprocess input
                features = self.vectorizer.transform([text])
                feature_tensor = torch.FloatTensor(features.toarray())
                
                # Make prediction
                with torch.no_grad():
                    outputs = self.model(feature_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    prediction = torch.argmax(probabilities, dim=1)
                
                return jsonify({
                    'prediction': self.label_encoder.inverse_transform(prediction.numpy())[0],
                    'confidence': float(probabilities.max())
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port)