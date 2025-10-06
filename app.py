from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client
import logging
import time

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

# Global client variable (lazy loaded)
_client = None


def get_client(max_retries=3, retry_delay=5):
    """Lazy load the Gradio client with retry logic"""
    global _client

    if _client is not None:
        return _client

    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing connection to Space (attempt {attempt + 1}/{max_retries}): {SPACE_NAME}")
            _client = Client(SPACE_NAME)
            logger.info("✅ Connected to Space successfully!")
            return _client
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect after {max_retries} attempts")
                raise Exception(
                    f"Could not connect to Space after {max_retries} attempts. The Space might be sleeping or unavailable.")

    return _client


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

        # Get client with retry logic
        client = get_client()

        # Call Space API with retry
        max_predict_retries = 3
        for attempt in range(max_predict_retries):
            try:
                result = client.predict(text, api_name="/predict")
                break
            except Exception as e:
                if attempt < max_predict_retries - 1:
                    logger.warning(f"Prediction attempt {attempt + 1} failed: {str(e)}, retrying...")
                    time.sleep(3)
                else:
                    raise Exception(f"Prediction failed after {max_predict_retries} attempts: {str(e)}")

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
        return jsonify({
            'error': str(e),
            'message': 'Classification failed. The Space might be sleeping. Please try again in 30 seconds.'
        }), 500


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

        # Get client with retry logic
        client = get_client()

        results = []
        for text in texts:
            if text.strip():
                try:
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
                except Exception as e:
                    logger.warning(f"Failed to classify text '{text[:30]}...': {str(e)}")
                    results.append({
                        'text': text,
                        'error': 'Classification failed'
                    })

        return jsonify({'results': results})

    except Exception as e:
        logger.error(f"Error during batch classification: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Batch classification failed. Please try again.'
        }), 500


@app.route('/warmup', methods=['GET'])
def warmup():
    """Warmup endpoint to initialize the Space connection"""
    try:
        logger.info("Warming up Space connection...")
        client = get_client()

        # Test with a simple prediction
        test_result = client.predict("test", api_name="/predict")

        return jsonify({
            'status': 'success',
            'message': 'Space connection initialized and warmed up',
            'space': SPACE_NAME
        })
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")
        return jsonify({
            'status': 'failed',
            'error': str(e),
            'message': 'Could not connect to Space. It might be sleeping.'
        }), 503


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        client_status = "connected" if _client is not None else "not_initialized"

        return jsonify({
            'status': 'healthy',
            'space': SPACE_NAME,
            'endpoint': f"https://{SPACE_NAME.replace('/', '-')}.hf.space",
            'client_status': client_status
        })
    except Exception as e:
        return jsonify({
            'status': 'healthy',
            'space': SPACE_NAME,
            'note': 'Client will initialize on first request'
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
            '/warmup': 'GET - Warmup Space connection',
            '/classify': 'POST - Classify single text',
            '/classify/batch': 'POST - Classify multiple texts'
        },
        'example_request': {
            'url': '/classify',
            'method': 'POST',
            'body': {
                'text': 'Win a free iPhone now!'
            }
        },
        'note': 'First request may take 20-30 seconds if Space is sleeping. Use /warmup to initialize.'
    })


if __name__ == '__main__':
    import os

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)