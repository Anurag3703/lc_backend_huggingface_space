from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client
import logging

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your Hugging Face Space endpoint
SPACE_NAME = "Anurag3703/bert-spam-classifier-demo"

# Initialize Gradio client (lightweight!)
logger.info(f"Initializing connection to Space: {SPACE_NAME}")
client = Client(SPACE_NAME)
logger.info("✅ Connected to Space successfully!")


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify text as spam or ham using your HF Space

    Request body:
    {
        "text": "Your message here"
    }

    Response:
    {
        "text": "Your message here",
        "label": "spam" or "ham",
        "confidence": 0.99,
        "probabilities": {
            "spam": 0.99,
            "ham": 0.01
        }
    }
    """
    try:
        # Get input text
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        logger.info(f"Classifying text: '{text[:50]}...'")

        # Call your Space's API
        result = client.predict(text, api_name="/predict")

        # Parse the result
        spam_confidence = next(
            (item['confidence'] for item in result['confidences'] if item['label'] == 'Spam'),
            0
        )
        ham_confidence = next(
            (item['confidence'] for item in result['confidences'] if item['label'] == 'Not Spam (Ham)'),
            0
        )

        # Determine final label
        label = "spam" if spam_confidence > ham_confidence else "ham"
        confidence = max(spam_confidence, ham_confidence)

        response = {
            'text': text,
            'label': label,
            'confidence': round(confidence, 4),
            'probabilities': {
                'spam': round(spam_confidence, 4),
                'ham': round(ham_confidence, 4)
            }
        }

        logger.info(f"Result: {label} ({confidence:.2%} confidence)")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/classify/batch', methods=['POST'])
def classify_batch():
    """
    Classify multiple texts at once

    Request body:
    {
        "texts": ["message 1", "message 2", ...]
    }
    """
    try:
        data = request.get_json()
        texts = data.get('texts', [])

        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Please provide a list of texts'}), 400

        results = []
        for text in texts:
            if text.strip():
                result = client.predict(text, api_name="/predict")

                spam_conf = next((item['confidence'] for item in result['confidences']
                                  if item['label'] == 'Spam'), 0)
                ham_conf = next((item['confidence'] for item in result['confidences']
                                 if item['label'] == 'Not Spam (Ham)'), 0)

                results.append({
                    'text': text,
                    'label': "spam" if spam_conf > ham_conf else "ham",
                    'confidence': round(max(spam_conf, ham_conf), 4)
                })

        return jsonify({'results': results})

    except Exception as e:
        logger.error(f"Error during batch classification: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'space': SPACE_NAME,
        'endpoint': f"https://{SPACE_NAME.replace('/', '-')}.hf.space"
    })


@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'name': 'BERT Spam Classifier API',
        'version': '1.0',
        'space': SPACE_NAME,
        'endpoints': {
            '/': 'GET - API documentation',
            '/health': 'GET - Health check',
            '/classify': 'POST - Classify single text',
            '/classify/batch': 'POST - Classify multiple texts'
        },
        'example_request': {
            'url': '/classify',
            'method': 'POST',
            'body': {
                'text': 'Win a free iPhone now!'
            }
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)