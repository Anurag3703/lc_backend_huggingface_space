from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client
import logging
import time

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

SPACE_NAME = "Anurag3703/bert-spam-classifier-demo"

# Global client - lazy loaded
_client = None
_client_init_time = None


def get_client(max_retries=3):
    """Lazy load Gradio client"""
    global _client, _client_init_time

    # Return cached client if recent (within 10 minutes)
    if _client is not None:
        if _client_init_time and (time.time() - _client_init_time) < 600:
            return _client
        else:
            logger.info("Client cache expired, reinitializing...")
            _client = None

    # Initialize with retry
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to Space (attempt {attempt + 1}/{max_retries})")

            _client = Client(SPACE_NAME, verbose=False)

            # Test with sample prediction
            logger.info("Testing connection...")
            _client.predict("test", api_name="/predict")

            _client_init_time = time.time()
            logger.info("✅ Connected successfully!")
            return _client

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise Exception(
                    f"Could not connect after {max_retries} attempts. "
                    "Space might be sleeping. Try /warmup first."
                )

    return _client


@app.route('/classify', methods=['POST'])
def classify():
    """Classify text as spam or ham"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        logger.info(f"Classifying: '{text[:50]}...'")

        # Get client
        try:
            client = get_client()
        except Exception as e:
            return jsonify({
                'error': 'Space connection failed',
                'details': str(e),
                'suggestion': 'Call /warmup first to wake the Space'
            }), 503

        # Predict with retry
        max_retries = 3
        result = None

        for attempt in range(max_retries):
            try:
                result = client.predict(text, api_name="/predict")
                logger.info(f"Prediction result: {result}")
                break
            except Exception as e:
                logger.warning(f"Prediction attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise Exception(f"Prediction failed: {str(e)}")

        # Parse result
        spam_conf = next(
            (item['confidence'] for item in result['confidences'] if item['label'] == 'Spam'),
            0
        )
        ham_conf = next(
            (item['confidence'] for item in result['confidences'] if item['label'] == 'Not Spam (Ham)'),
            0
        )

        label = "spam" if spam_conf > ham_conf else "ham"
        confidence = max(spam_conf, ham_conf)

        response = {
            'text': text,
            'label': label,
            'confidence': round(confidence, 4),
            'probabilities': {
                'spam': round(spam_conf, 4),
                'ham': round(ham_conf, 4)
            }
        }

        logger.info(f"✅ {label} ({confidence:.2%})")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Classification failed. Try /warmup first.'
        }), 500


@app.route('/warmup', methods=['GET'])
def warmup():
    """Wake up and test the Space"""
    try:
        logger.info("🔥 Warming up Space (30-60 seconds)...")
        start = time.time()

        # Force reinit
        global _client
        _client = None

        # Get client (this initializes and tests)
        client = get_client(max_retries=5)

        # Additional test prediction
        test_result = client.predict("This is a test", api_name="/predict")

        elapsed = time.time() - start

        return jsonify({
            'status': 'success',
            'message': 'Space warmed up successfully',
            'space': SPACE_NAME,
            'warmup_time': round(elapsed, 2)
        })

    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")
        return jsonify({
            'status': 'failed',
            'error': str(e),
            'message': 'Could not connect to Space'
        }), 503


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    client_status = "connected" if _client is not None else "not_initialized"

    return jsonify({
        'status': 'healthy',
        'space': SPACE_NAME,
        'client_status': client_status
    })


@app.route('/', methods=['GET'])
def home():
    """API docs"""
    return jsonify({
        'name': 'BERT Spam Classifier API',
        'version': '1.3',
        'space': SPACE_NAME,
        'endpoints': {
            '/': 'GET - Docs',
            '/health': 'GET - Health check',
            '/warmup': 'GET - Wake Space (CALL FIRST!)',
            '/classify': 'POST - Classify text'
        },
        'usage': {
            'step_1': 'GET /warmup (wait 30-60s)',
            'step_2': 'POST /classify {"text": "your message"}'
        }
    })


if __name__ == '__main__':
    import os

    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting on port {port}")
    logger.info(f"📡 Space: {SPACE_NAME}")
    logger.info("💡 Client will lazy load on first request")
    app.run(debug=False, host='0.0.0.0', port=port)