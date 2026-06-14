import os
import json
import random
import secrets
from datetime import datetime, timezone
import logging
import sqlite3
from contextlib import closing
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Import modular components
from core.inference import get_model, predict_image
from utils.dicom_parser import process_dicom, extract_metadata
from utils.image_helpers import process_standard_image, image_to_base64, analyze_image_quality
from utils.patient_db import (
    save_result, get_history, list_patients, save_feedback,
    verify_user, log_report_generation, get_connection,
    create_patient, update_patient, archive_patient, search_patients as search_db_patients,
    create_study, save_prediction_record, save_review_record,
    log_audit_event, create_notification, list_notifications,
    mark_notification_read, get_dashboard_stats, list_studies,
    get_study_audit_trail, get_similar_cases, get_research_export_data
)
from utils.fhir_mock import search_patients, get_pacs_status
from api_v1 import api_v1

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "Nirikshon-clinical-key-9281")
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# ── Configuration ──────────────────────────────────────────

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "http://localhost:3000")
CORS(app, resources={r"/*": {
    "origins": ALLOWED_ORIGIN,
    "expose_headers": ["X-CSRF-Token"],
    "allow_headers": ["Content-Type", "X-CSRF-Token"]
}}, supports_credentials=True)

app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

# Register Phase 4 API Router
app.register_blueprint(api_v1)

# CSRF protection double-submit cookie validation hook
@app.before_request
def csrf_protect():
    if request.method == 'POST':
        # Match routes requiring CSRF validation: /patients/<patient_id>/save and /feedback
        is_protected = False
        if request.path == '/feedback':
            is_protected = True
        elif request.path.startswith('/patients/') and request.path.endswith('/save'):
            is_protected = True
            
        if is_protected:
            csrf_cookie = request.cookies.get('csrf_token')
            csrf_header = request.headers.get('X-CSRF-Token')
            if not csrf_cookie or not csrf_header or csrf_cookie != csrf_header:
                app.logger.warning("CSRF validation failed: Token mismatch or missing")
                return jsonify({"error": "CSRF validation failed"}), 403

@app.after_request
def set_csrf_cookie(response):
    if not request.cookies.get('csrf_token'):
        token = secrets.token_hex(16)
        response.set_cookie(
            'csrf_token', 
            token, 
            samesite='Lax', 
            secure=False, 
            httponly=False  # Must be False so JavaScript can read it for double-submit
        )
    return response

# Pre-load model on startup
get_model()

# ── Root Index & Health Check ────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "active",
        "message": "PulmonaryAI API Gateway is running.",
        "endpoints": {
            "health": "/health",
            "predict": "/predict [POST]",
            "patients": "/patients [GET]",
            "patients_save": "/patients/<patient_id>/save [POST]",
            "patients_history": "/patients/<patient_id>/history [GET]",
            "feedback": "/feedback [POST]",
            "fhir_patients": "/fhir/patients [GET]",
            "fhir_pacs_status": "/fhir/pacs/status [GET]",
            "login": "/login [POST]",
            "logout": "/logout [POST]",
            "session": "/session [GET]",
            "report_audit": "/report/audit [POST]"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "active", "model_loaded": get_model() is not None})

# ── Auth Endpoints ──────────────────────────────────────────
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not data.get("username") or not data.get("password"):
        return jsonify({"error": "Username and password required"}), 400
        
    username = data.get("username")
    password = data.get("password")
    
    user = verify_user(username, password)
    if user:
        session["username"] = user["username"]
        session["role"] = user["role"]
        return jsonify({
            "success": True,
            "username": user["username"],
            "role": user["role"]
        })
    return jsonify({"error": "Invalid username or password"}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Successfully logged out"})

@app.route('/session', methods=['GET'])
def check_session():
    if "username" in session:
        return jsonify({
            "authenticated": True,
            "username": session["username"],
            "role": session["role"]
        })
    return jsonify({"authenticated": False}), 401

# ── Core Prediction Endpoint ───────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
        
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        file_bytes = file.read()
        filename = file.filename.lower()
        
        # Check if patient_id is passed in form data
        patient_id = request.form.get("patient_id")
        
        dcm_obj = None
        if filename.endswith('.dcm') or filename.endswith('.dicom'):
            img, dcm_obj = process_dicom(file_bytes)
            meta = extract_metadata(dcm_obj)
            if not patient_id:
                patient_id = meta.get("patient_id")
        else:
            img = process_standard_image(file_bytes)
            random_id = f"PX-{secrets.randbelow(90000) + 10000}"
            meta = {
                "patient_id": random_id,
                "patient_name": "Anonymous Patient",
                "patient_age": "N/A (Non-DICOM File)",
                "patient_sex": "N/A",
                "modality": "CR (Computed Radiography)",
                "study_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "body_part": "Chest",
                "pixel_spacing": None
            }
            if not patient_id:
                patient_id = random_id

        # Generate a unique study ID
        study_id = f"ST-{secrets.randbelow(90000) + 10000}"
        
        # Ensure patient exists in our patients table
        with closing(get_connection()) as conn:
            cursor = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
            patient_row = cursor.fetchone()
            if not patient_row:
                create_patient(
                    patient_id, 
                    meta.get("patient_name", "Anonymous Patient"), 
                    meta.get("patient_age", "N/A"), 
                    meta.get("patient_sex", "N/A"), 
                    ""
                )
            else:
                meta["patient_name"] = patient_row["name"]
                meta["patient_age"] = patient_row["age"]
                meta["patient_sex"] = patient_row["sex"]

        # Run Backend Image Quality Assessment (IQA)
        iqa_results = analyze_image_quality(img)

        # Run deep learning inference
        result_dict, heatmap_img = predict_image(img)
        
        original_base64 = image_to_base64(img)
        heatmap_base64 = image_to_base64(heatmap_img)
        
        # Create study record
        reviewer_user = request.form.get("reviewer_username") or session.get("username")
        create_study(
            study_id, 
            patient_id, 
            meta, 
            original_base64, 
            status='uploaded', 
            reviewer=reviewer_user, 
            iqa=iqa_results
        )

        # Update result with images and metadata
        is_tb = result_dict["is_tb"]
        result_record = {
            "confidence": result_dict["confidence"],
            "is_tb": is_tb,
            "prediction": result_dict["prediction"],
            "heatmap_image": heatmap_base64,
            "inference_time_ms": result_dict.get("inference_time_ms", 324.0),
            "attention_region": "right apical" if is_tb else "clear",
            "heatmap_coverage": 15.2 if is_tb else 0.0
        }

        # Save prediction and update study status to 'ai_complete'
        save_prediction_record(study_id, result_record)

        # Audit event logging
        log_audit_event(study_id, session.get("username", "system"), "Upload")
        log_audit_event(study_id, "system", "Inference")

        # Notification generation
        confidence_pct = int(result_record["confidence"] * 100)
        verdict = result_record["prediction"]
        patient_name = meta.get("patient_name", "Anonymous")
        msg = f"Inference Complete for {patient_name} ({patient_id}): AI predicts {verdict} ({confidence_pct}% confidence)"
        create_notification(study_id, msg)
        
        # Real-time WebSocket emission
        socketio.emit("study_updated", {
            "study_id": study_id,
            "status": "ai_complete",
            "message": msg
        })

        response = {
            "success": True,
            "study_id": study_id,
            "patient_id": patient_id,
            "metadata": meta,
            "original_image": original_base64,
            "heatmap_image": heatmap_base64,
            "image_quality": iqa_results
        }
        response.update(result_dict)
        return jsonify(response)
        
    except Exception as e:
        app.logger.error("Exception occurred during prediction", exc_info=True)
        is_debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
        if is_debug:
            return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "Internal server error"}), 500

# ── Patient Management Endpoints (Phase 2) ───────────────────
@app.route('/patients', methods=['POST'])
def add_patient():
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    data = request.get_json()
    if not data or not data.get("id") or not data.get("name"):
        return jsonify({"error": "Patient ID and Name are required"}), 400
    try:
        res = create_patient(
            data["id"],
            data["name"],
            data.get("age", "N/A"),
            data.get("sex", "N/A"),
            data.get("notes", "")
        )
        return jsonify({"success": True, "patient": res})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Patient ID already exists"}), 400
    except Exception as e:
        app.logger.error("Error creating patient", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/patients/<patient_id>', methods=['PUT'])
def edit_patient(patient_id):
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Patient Name is required"}), 400
    try:
        res = update_patient(
            patient_id,
            data["name"],
            data.get("age", "N/A"),
            data.get("sex", "N/A"),
            data.get("notes", "")
        )
        return jsonify({"success": True, "patient": res})
    except Exception as e:
        app.logger.error("Error updating patient", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/patients/<patient_id>/archive', methods=['POST'])
def toggle_archive(patient_id):
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    data = request.get_json() or {}
    archive = data.get("archive", True)
    try:
        archive_patient(patient_id, archive)
        return jsonify({"success": True, "archived": archive})
    except Exception as e:
        app.logger.error("Error archiving patient", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/patients', methods=['GET'])
def query_patients():
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    search_q = request.args.get("search", "")
    inc_archived = request.args.get("include_archived", "false").lower() == "true"
    try:
        res = search_db_patients(search_q, inc_archived)
        return jsonify({"patients": res})
    except Exception as e:
        app.logger.error("Error searching patients", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/patients/<patient_id>/history', methods=['GET'])
def get_patient_history(patient_id):
    """Get all historical records for a patient."""
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    if not patient_id or not all(c.isalnum() or c in '-_' for c in patient_id):
        return jsonify({"error": "Invalid patient ID format"}), 400
    try:
        history = get_history(patient_id)
        return jsonify({"patient_id": patient_id, "records": history, "count": len(history)})
    except Exception as e:
        app.logger.error("Exception occurred getting patient history", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# ── Study Management Endpoints (Phase 2) ────────────────────
@app.route('/studies', methods=['GET'])
def query_studies():
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    date_str = request.args.get("date")
    status_str = request.args.get("status")
    reviewer = request.args.get("reviewer")
    try:
        res = list_studies(date_str, status_str, reviewer)
        return jsonify({"studies": res})
    except Exception as e:
        app.logger.error("Error querying studies", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/studies/<study_id>/audit', methods=['GET'])
def get_study_audit(study_id):
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    try:
        trail = get_study_audit_trail(study_id)
        return jsonify({"study_id": study_id, "audit_trail": trail})
    except Exception as e:
        app.logger.error("Error fetching study audit trail", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# ── Dashboard Stats Endpoints (Phase 2) ──────────────────────
@app.route('/dashboard/stats', methods=['GET'])
def dashboard_stats():
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    try:
        res = get_dashboard_stats()
        return jsonify(res)
    except Exception as e:
        app.logger.error("Error loading dashboard stats", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/model/metadata', methods=['GET'])
def get_model_metadata():
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    try:
        meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                data = json.load(f)
            return jsonify(data)
        return jsonify({"error": "Metadata file not found"}), 404
    except Exception as e:
        app.logger.error("Error reading model metadata", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/studies/<study_id>/similar', methods=['GET'])
def get_study_similar(study_id):
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    try:
        res = get_similar_cases(study_id)
        return jsonify(res)
    except Exception as e:
        app.logger.error("Error getting similar cases", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/studies/<study_id>/heatmaps', methods=['GET'])
def get_study_heatmaps(study_id):
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    try:
        with closing(get_connection()) as conn:
            cursor = conn.execute("SELECT original_image FROM studies WHERE id = ?", (study_id,))
            row = cursor.fetchone()
            if not row:
                return jsonify({"error": "Study not found"}), 404
            orig_img_b64 = row["original_image"]
            
        import base64
        from PIL import Image
        import io
        
        if "," in orig_img_b64:
            orig_img_b64 = orig_img_b64.split(",")[1]
            
        img_bytes = base64.b64decode(orig_img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        result_dict, _ = predict_image(img)
        return jsonify({
            "study_id": study_id,
            "heatmaps": result_dict.get("heatmaps", {})
        })
    except Exception as e:
        app.logger.error("Error generating study heatmaps", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/export/research', methods=['GET'])
def export_research():
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    export_format = request.args.get("format", "json").lower()
    try:
        data = get_research_export_data()
        
        if export_format == "csv":
            import csv
            import io
            from flask import make_response
            
            dest = io.StringIO()
            writer = csv.writer(dest)
            
            headers = [
                "study_id", "patient_id", "patient_name", "patient_age", "patient_sex",
                "study_date", "modality", "status", "exposure", "coverage", "resolution",
                "rotation", "quality_score", "suitability", "confidence", "is_tb",
                "prediction", "inference_time_ms", "clinician_override", "review_comments", "reviewer_username"
            ]
            writer.writerow(headers)
            
            for row in data:
                writer.writerow([row.get(h, "") for h in headers])
                
            response = make_response(dest.getvalue())
            response.headers["Content-Disposition"] = "attachment; filename=Nirikshon_research_export.csv"
            response.headers["Content-type"] = "text/csv"
            return response
        else:
            return jsonify({"studies": data})
    except Exception as e:
        app.logger.error("Error exporting research data", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# ── Notifications Endpoints (Phase 2) ────────────────────────
@app.route('/notifications', methods=['GET'])
def fetch_notifications():
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    try:
        res = list_notifications()
        return jsonify({"notifications": res})
    except Exception as e:
        app.logger.error("Error loading notifications", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/notifications/<int:notif_id>/read', methods=['POST'])
def mark_read(notif_id):
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    try:
        mark_notification_read(notif_id)
        return jsonify({"success": True})
    except Exception as e:
        app.logger.error("Error marking notification read", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# ── Clinician Feedback / Override Endpoint ─────────────────
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Store clinician override and annotation for audit trail."""
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    study_id = data.get("study_id") or data.get("patient_id", "UNKNOWN")
    try:
        if "reviewer_username" not in data:
            data["reviewer_username"] = session["username"]
        save_feedback(study_id, data)

        # Resolve study_id if patient_id was passed
        resolved_study_id = study_id
        with closing(get_connection()) as conn:
            cursor = conn.execute("SELECT id FROM studies WHERE id = ?", (study_id,))
            if not cursor.fetchone():
                cursor = conn.execute("SELECT id FROM studies WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1", (study_id,))
                row = cursor.fetchone()
                if row:
                    resolved_study_id = row["id"]

        # Audit event
        log_audit_event(resolved_study_id, session["username"], "Review")

        # Notification
        msg = f"Review signed-off by {session['username']} for Case {resolved_study_id}"
        create_notification(resolved_study_id, msg)
        
        # Real-time WebSocket emission
        socketio.emit("study_updated", {
            "study_id": resolved_study_id,
            "status": "reviewed",
            "message": msg
        })

        return jsonify({
            "success": True,
            "message": f"Feedback recorded for {resolved_study_id}",
            "audit_logged": True
        })
    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except Exception as e:
        app.logger.error("Exception occurred submitting feedback", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# ── Report Export Auditing Endpoint ────────────────────────
@app.route('/report/audit', methods=['POST'])
def audit_report():
    """Log PDF report generation event."""
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    data = request.get_json()
    if not data or not (data.get("study_id") or data.get("patient_id")):
        return jsonify({"error": "Study ID or Patient ID required"}), 400
    try:
        study_id = data.get("study_id") or data.get("patient_id")
        log_report_generation(study_id)

        # Resolve study_id
        resolved_study_id = study_id
        with closing(get_connection()) as conn:
            cursor = conn.execute("SELECT id FROM studies WHERE id = ?", (study_id,))
            if not cursor.fetchone():
                cursor = conn.execute("SELECT id FROM studies WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1", (study_id,))
                row = cursor.fetchone()
                if row:
                    resolved_study_id = row["id"]

        # Audit event
        log_audit_event(resolved_study_id, session["username"], "Report Export")

        # Notification
        create_notification(resolved_study_id, f"PDF Report downloaded for Case {resolved_study_id}")

        return jsonify({"success": True, "message": "Report generation logged"})
    except Exception as e:
        app.logger.error("Exception in audit_report", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# ── Mock FHIR / EHR Endpoints ─────────────────────────────
@app.route('/fhir/patients', methods=['GET'])
def fhir_search_patients():
    """Simulate a FHIR R4 patient search."""
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    query = request.args.get('search', '')
    count_str = request.args.get('count', '8')
    if not count_str.isdigit():
        count = 8
    else:
        count = min(100, max(1, int(count_str)))

    try:
        patients = search_patients(query, count)
        return jsonify({
            "resourceType": "Bundle",
            "type": "searchset",
            "total": len(patients),
            "entry": patients
        })
    except Exception as e:
        app.logger.error("Exception in fhir_search_patients", exc_info=True)
        is_debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
        return jsonify({"error": str(e) if is_debug else "Internal server error"}), 500

@app.route('/fhir/pacs/status', methods=['GET'])
def fhir_pacs_status():
    """Return mock PACS/DICOM node statuses."""
    if "username" not in session:
        return jsonify({"error": "Authentication required"}), 401
    try:
        return jsonify({"nodes": get_pacs_status()})
    except Exception as e:
        app.logger.error("Exception in fhir_pacs_status", exc_info=True)
        is_debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
        return jsonify({"error": str(e) if is_debug else "Internal server error"}), 500

if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    host_ip = os.environ.get("FLASK_HOST", "127.0.0.1")
    print(f"Starting Enterprise API Backend on {host_ip}:5000 (Debug: {debug_mode})...")
    socketio.run(app, host=host_ip, port=5000, debug=debug_mode, allow_unsafe_werkzeug=True)
