from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import time
import os

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


HF_TOKEN = os.environ.get('HF_TOKEN')

if not HF_TOKEN:
    logger.error("HF_TOKEN environment variable not set!")
    raise ValueError("HF_TOKEN must be set as environment variable")
# Use Inference API instead of Space
MODEL_NAME = "Anurag3703/bert-spam-classifier"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"


headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def query_model(text, max_retries=5):
    """Query the model with retry logic for loading"""
    for attempt in range(max_retries):
        response = requests.post(API_URL, headers=headers, json={"inputs": text})

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            # Model is loading
            wait_time = 10 * (attempt + 1)  # Exponential backoff
            logger.info(f"Model loading... waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            raise Exception(f"API returned {response.status_code}")

    raise Exception("Model failed to load after multiple attempts")


@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        logger.info(f"Classifying: '{text[:50]}...'")

        result = query_model(text)

        # Parse result format: [[{"label": "LABEL_0", "score": 0.xx}, {...}]]
        if isinstance(result, list) and len(result) > 0:
            predictions = result[0]

            spam_score = next(
                (p['score'] for p in predictions if 'LABEL_1' in p['label'] or 'spam' in p['label'].lower()), 0)
            ham_score = next(
                (p['score'] for p in predictions if 'LABEL_0' in p['label'] or 'ham' in p['label'].lower()), 0)

            label = "spam" if spam_score > ham_score else "ham"

            return jsonify({
                'text': text,
                'label': label,
                'confidence': round(max(spam_score, ham_score), 4),
                'probabilities': {
                    'spam': round(spam_score, 4),
                    'ham': round(ham_score, 4)
                }
            })
        else:
            return jsonify({'error': 'Unexpected response format'}), 500

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': MODEL_NAME})


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'BERT Spam Classifier API',
        'version': '1.0',
        'model': MODEL_NAME,
        'endpoints': {
            '/': 'GET - API documentation',
            '/health': 'GET - Health check',
            '/classify': 'POST - Classify text'
        }
    })


if __name__ == '__main__':
    import os

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)