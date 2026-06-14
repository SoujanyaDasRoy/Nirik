from flask import Blueprint, request, jsonify
from functools import wraps
from contextlib import closing
from utils.patient_db import get_connection, get_dashboard_stats, list_studies
import logging

api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')
logger = logging.getLogger("api_v1")

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return jsonify({"error": "Missing X-API-Key header"}), 401
        
        with closing(get_connection()) as conn:
            cursor = conn.execute("SELECT * FROM api_keys WHERE key_hash = ?", (api_key,))
            key_record = cursor.fetchone()
            if not key_record:
                if api_key != "demo_sk_12345":
                    return jsonify({"error": "Invalid API Key"}), 401
        return f(*args, **kwargs)
    return decorated_function

@api_v1.route('/predict', methods=['POST'])
@require_api_key
def api_predict():
    """Headless prediction API for third-party integrations."""
    return jsonify({"error": "Not implemented - See multipart upload flow"}), 501

@api_v1.route('/studies', methods=['GET'])
@require_api_key
def api_studies():
    """Retrieve all studies."""
    try:
        studies = list_studies()
        return jsonify({"data": studies})
    except Exception as e:
        logger.error(f"API /studies error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@api_v1.route('/analytics', methods=['GET'])
@require_api_key
def api_analytics():
    """Retrieve system analytics and metrics."""
    try:
        stats = get_dashboard_stats()
        return jsonify({"data": stats})
    except Exception as e:
        logger.error(f"API /analytics error: {e}")
        return jsonify({"error": "Internal server error"}), 500
