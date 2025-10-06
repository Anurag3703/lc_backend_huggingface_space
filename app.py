from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client
import logging
import time
import json

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
_client_init_time = None


def get_client(max_retries=3, retry_delay=10):
    """Lazy load the Gradio client with improved retry logic"""
    global _client, _client_init_time

    # Return cached client if it exists and was created recently (within 5 minutes)
    if _client is not None:
        if _client_init_time and (time.time() - _client_init_time) < 300:
            return _client
        else:
            logger.info("Client cache expired, reinitializing...")
            _client = None

    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to Space (attempt {attempt + 1}/{max_retries}): {SPACE_NAME}")

            # Try to initialize client with timeout
            _client = Client(SPACE_NAME, verbose=False)

            # Test the connection with a simple prediction
            logger.info("Testing connection with sample text...")
            test_result = _client.predict("test", api_name="/predict")

            _client_init_time = time.time()
            logger.info("✅ Connected and verified Space successfully!")
            return _client

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Space might still be waking up. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(
                    "Space returned invalid response. It might still be starting up. "
                    "Please wait 30-60 seconds and try again."
                )
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(
                    f"Could not connect to Space after {max_retries} attempts. "
                    "The Space might be sleeping or unavailable. Please try /warmup first."
                )

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
        try:
            client = get_client()
        except Exception as e:
            return jsonify({
                'error': str(e),
                'suggestion': 'Try hitting the /warmup endpoint first to wake up the Space',
                'status': 'space_unavailable'
            }), 503

        # Call Space API with retry
        max_predict_retries = 3
        result = None

        for attempt in range(max_predict_retries):
            try:
                logger.info(f"Prediction attempt {attempt + 1}/{max_predict_retries}")
                result = client.predict(text, api_name="/predict")
                logger.info(f"Raw result: {result}")
                break
            except Exception as e:
                logger.warning(f"Prediction attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_predict_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise Exception(f"Prediction failed after {max_predict_retries} attempts: {str(e)}")

        # Parse the result - handle different possible formats
        try:
            # Format 1: Dictionary with 'confidences' key
            if isinstance(result, dict) and 'confidences' in result:
                spam_confidence = next(
                    (item['confidence'] for item in result['confidences'] if 'Spam' in item['label']),
                    0
                )
                ham_confidence = next(
                    (item['confidence'] for item in result['confidences'] if 'Ham' in item['label']),
                    0
                )
            # Format 2: Dictionary with 'label' key (direct output)
            elif isinstance(result, dict) and 'label' in result:
                if result['label'] == 'Spam':
                    spam_confidence = result.get('confidence', 0.5)
                    ham_confidence = 1 - spam_confidence
                else:
                    ham_confidence = result.get('confidence', 0.5)
                    spam_confidence = 1 - ham_confidence
            # Format 3: Tuple or list format
            elif isinstance(result, (tuple, list)):
                # Assume first element is label, second is confidence dict
                spam_confidence = result[1].get('Spam', 0) if len(result) > 1 else 0
                ham_confidence = result[1].get('Ham', 0) if len(result) > 1 else 0
            else:
                logger.error(f"Unexpected result format: {result}")
                return jsonify({
                    'error': 'Unexpected response format from model',
                    'raw_result': str(result)
                }), 500

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
            logger.error(f"Error parsing result: {str(e)}")
            return jsonify({
                'error': 'Failed to parse model output',
                'details': str(e),
                'raw_result': str(result)
            }), 500

    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Classification failed. The Space might be sleeping. Please try /warmup first.'
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
        try:
            client = get_client()
        except Exception as e:
            return jsonify({
                'error': str(e),
                'suggestion': 'Try hitting the /warmup endpoint first'
            }), 503

        results = []
        for text in texts:
            if text.strip():
                try:
                    result = client.predict(text, api_name="/predict")

                    spam_conf = next((item['confidence'] for item in result['confidences']
                                      if 'Spam' in item['label']), 0)
                    ham_conf = next((item['confidence'] for item in result['confidences']
                                     if 'Ham' in item['label']), 0)

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
        logger.info("⏳ Warming up Space connection (this may take 30-60 seconds)...")
        start_time = time.time()

        # Force reinitialize client
        global _client
        _client = None

        client = get_client(max_retries=5, retry_delay=15)

        elapsed = time.time() - start_time

        return jsonify({
            'status': 'success',
            'message': 'Space connection initialized and warmed up',
            'space': SPACE_NAME,
            'warmup_time_seconds': round(elapsed, 2)
        })
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")
        return jsonify({
            'status': 'failed',
            'error': str(e),
            'message': 'Could not connect to Space. It might still be starting up. Please wait 60 seconds and try again.',
            'space': SPACE_NAME
        }), 503


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    client_status = "connected" if _client is not None else "not_initialized"

    return jsonify({
        'status': 'healthy',
        'space': SPACE_NAME,
        'space_url': f"https://{SPACE_NAME.replace('/', '-')}.hf.space",
        'client_status': client_status,
        'note': 'Use /warmup to initialize Space connection if not connected'
    })


@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'name': 'BERT Spam Classifier API',
        'version': '1.1',
        'space': SPACE_NAME,
        'endpoints': {
            '/': 'GET - API documentation',
            '/health': 'GET - Health check',
            '/warmup': 'GET - Warmup Space connection (recommended before first use)',
            '/classify': 'POST - Classify single text',
            '/classify/batch': 'POST - Classify multiple texts'
        },
        'usage': {
            'step_1': 'First, hit /warmup to wake up the Space (may take 30-60 seconds)',
            'step_2': 'Then use /classify to classify your text'
        },
        'example_request': {
            'url': '/classify',
            'method': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'body': {
                'text': 'Win a free iPhone now!'
            }
        }
    })


if __name__ == '__main__':
    import os

    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting server on port {port}")
    logger.info(f"📡 Using Space: {SPACE_NAME}")
    logger.info("💡 Tip: Hit /warmup endpoint first to wake up the Space!")

    app.run(debug=False, host='0.0.0.0', port=port)