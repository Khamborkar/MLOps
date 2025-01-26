from flask import Flask, request, jsonify
import subprocess
import joblib
import sys
import importlib.util
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Dynamically load model.py
model_path = 'src/model.py'
spec = importlib.util.spec_from_file_location("model", model_path)
model_module = importlib.util.module_from_spec(spec)
sys.modules["model"] = model_module
spec.loader.exec_module(model_module)

# Extract functions from the loaded module
load_model_and_tokenizer = getattr(model_module, 'load_model_and_tokenizer', None)
predict_sentiment = getattr(model_module, 'predict_sentiment', None)

if not load_model_and_tokenizer or not predict_sentiment:
    raise ImportError("Failed to load required functions from the model module.")

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer once at the start of the application
try:
    model, tokenizer = load_model_and_tokenizer()
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model and tokenizer: {e}")
    model, tokenizer = None, None
    raise RuntimeError("Failed to load model or tokenizer. Check the artifacts.")

# Define a root route
@app.route('/')
def health_check():
    return jsonify({'status': 'App is running', 'routes': ['/predict']}), 200

# Define a route to predict sentiment from text
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not tokenizer:
        return jsonify({'error': 'Model or tokenizer not loaded properly'}), 500

    # Get the input text from the request
    input_text = request.json.get('text')
    if not input_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Make a prediction
        prediction = predict_sentiment(model, tokenizer, input_text)

        # Convert prediction to a user-friendly format
        sentiment = 'positive' if prediction[0][1] > max(prediction[0][0], prediction[0][2]) else \
                    'negative' if prediction[0][0] > max(prediction[0][1], prediction[0][2]) else 'neutral'

        # Return the result as JSON
        return jsonify({'sentiment': sentiment, 'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

