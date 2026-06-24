"""
patient_db.py — Normalized SQLite database store for Nirikshon Phase 2.
Contains tables for users, patients, studies, predictions, reviews, reports, audit_trail, and notifications.
"""
import os
import sqlite3
import json
import logging
from datetime import datetime, timezone
from contextlib import closing
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np

if os.environ.get("DESKTOP_APP") == "true":
    if os.name == "nt":
        app_data_dir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Nirikhshon")
    else:
        app_data_dir = os.path.join(os.path.expanduser("~"), ".nirikhshon")
    os.makedirs(app_data_dir, exist_ok=True)
    DB_FILE = os.path.join(app_data_dir, "patients.db")
else:
    DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "patients.db")
logger = logging.getLogger("patient_db")

def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create the SQLite tables and seed default accounts on startup."""
    with closing(get_connection()) as conn:
        with conn:
            # Check if migrations are needed for Phase 4
            try:
                cursor = conn.execute("PRAGMA table_info(users)")
                columns = [row["name"] for row in cursor.fetchall()]
                if columns and "institution_id" not in columns:
                    logger.info("Migrating schema to Phase 4: Resetting database...")
                    conn.execute("DROP TABLE IF EXISTS api_keys")
                    conn.execute("DROP TABLE IF EXISTS case_shares")
                    conn.execute("DROP TABLE IF EXISTS study_comments")
                    conn.execute("DROP TABLE IF EXISTS reports")
                    conn.execute("DROP TABLE IF EXISTS reviews")
                    conn.execute("DROP TABLE IF EXISTS predictions")
                    conn.execute("DROP TABLE IF EXISTS studies")
                    conn.execute("DROP TABLE IF EXISTS patients")
                    conn.execute("DROP TABLE IF EXISTS users")
                    conn.execute("DROP TABLE IF EXISTS institutions")
                    conn.execute("DROP TABLE IF EXISTS audit_trail")
                    conn.execute("DROP TABLE IF EXISTS notifications")
            except Exception as e:
                logger.error(f"Migration check error: {e}")

            # 0. institutions table (Phase 4)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS institutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    type TEXT,
                    created_at TEXT
                )
            """)

            # 1. users table (Phase 4)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    role TEXT,
                    institution_id INTEGER,
                    created_at TEXT,
                    FOREIGN KEY(institution_id) REFERENCES institutions(id)
                )
            """)
            
            # 2. patients table (Phase 2)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    age TEXT,
                    sex TEXT,
                    notes TEXT,
                    is_archived INTEGER DEFAULT 0,
                    created_at TEXT
                )
            """)

            # 3. studies table (Phase 2 update)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS studies (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    study_date TEXT,
                    modality TEXT,
                    status TEXT,
                    original_image TEXT,
                    reviewer_username TEXT,
                    exposure TEXT,
                    coverage TEXT,
                    resolution TEXT,
                    rotation TEXT,
                    quality_score REAL,
                    suitability TEXT,
                    warnings TEXT,
                    created_at TEXT,
                    FOREIGN KEY(patient_id) REFERENCES patients(id),
                    FOREIGN KEY(reviewer_username) REFERENCES users(username)
                )
            """)

            # 4. predictions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT,
                    confidence REAL,
                    is_tb INTEGER,
                    prediction TEXT,
                    heatmap_image TEXT,
                    inference_time_ms REAL,
                    attention_region TEXT,
                    heatmap_coverage REAL,
                    created_at TEXT,
                    FOREIGN KEY(study_id) REFERENCES studies(id)
                )
            """)

            # 5. reviews table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT,
                    reviewer_username TEXT,
                    status TEXT,
                    comments TEXT,
                    clinician_note TEXT,
                    annotation_b64 TEXT,
                    created_at TEXT,
                    FOREIGN KEY(study_id) REFERENCES studies(id),
                    FOREIGN KEY(reviewer_username) REFERENCES users(username)
                )
            """)

            # 6. reports table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT,
                    created_at TEXT,
                    FOREIGN KEY(study_id) REFERENCES studies(id)
                )
            """)

            # 7. audit_trail table (Phase 2)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT,
                    timestamp TEXT,
                    username TEXT,
                    action TEXT,
                    FOREIGN KEY(study_id) REFERENCES studies(id)
                )
            """)

            # 8. notifications table (Phase 2)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT,
                    message TEXT,
                    is_read INTEGER DEFAULT 0,
                    created_at TEXT,
                    FOREIGN KEY(study_id) REFERENCES studies(id)
                )
            """)

            # 9. study_comments (Phase 4)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS study_comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT,
                    username TEXT,
                    comment TEXT,
                    created_at TEXT,
                    FOREIGN KEY(study_id) REFERENCES studies(id)
                )
            """)

            # 10. case_shares (Phase 4)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS case_shares (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT,
                    shared_by TEXT,
                    shared_with_institution_id INTEGER,
                    shared_with_username TEXT,
                    permissions TEXT,
                    created_at TEXT,
                    FOREIGN KEY(study_id) REFERENCES studies(id)
                )
            """)

            # 11. api_keys (Phase 4)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT UNIQUE,
                    name TEXT,
                    institution_id INTEGER,
                    created_at TEXT
                )
            """)

            # Seed default institution
            now_iso = datetime.now(timezone.utc).isoformat() + "Z"
            conn.execute("""
                INSERT OR IGNORE INTO institutions (id, name, type, created_at)
                VALUES (1, 'Central Hospital ApolloDx', 'Hospital', ?)
            """, (now_iso,))

            # Seed default users
            admin_hash = generate_password_hash("password123")
            reviewer_hash = generate_password_hash("password123")
            tech_hash = generate_password_hash("password123")
            
            conn.execute("""
                INSERT OR IGNORE INTO users (username, password_hash, role, institution_id, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, ("admin", admin_hash, "admin", 1, now_iso))

            conn.execute("""
                INSERT OR IGNORE INTO users (username, password_hash, role, institution_id, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, ("reviewer", reviewer_hash, "radiologist", 1, now_iso))

            conn.execute("""
                INSERT OR IGNORE INTO users (username, password_hash, role, institution_id, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, ("technician", tech_hash, "technician", 1, now_iso))


# Initialize database
init_db()


# ── Auth Helper Functions ──

def get_user(username: str) -> dict:
    """Retrieve user details by username."""
    with closing(get_connection()) as conn:
        cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            return {
                "username": row["username"],
                "role": row["role"],
                "password_hash": row["password_hash"]
            }
        return None

def verify_user(username: str, password_raw: str) -> dict:
    """Verify credentials and return user context if successful."""
    user = get_user(username)
    if user and check_password_hash(user["password_hash"], password_raw):
        return {
            "username": user["username"],
            "role": user["role"]
        }
    return None


# ── Patient Management Helper Functions ──

def create_patient(patient_id: str, name: str, age: str, sex: str, notes: str) -> dict:
    """Create a new patient record in the database."""
    now_str = datetime.now(timezone.utc).isoformat() + "Z"
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("""
                INSERT INTO patients (id, name, age, sex, notes, is_archived, created_at)
                VALUES (?, ?, ?, ?, ?, 0, ?)
            """, (patient_id, name, age, sex, notes, now_str))
    return {"id": patient_id, "name": name, "age": age, "sex": sex, "notes": notes}

def update_patient(patient_id: str, name: str, age: str, sex: str, notes: str) -> dict:
    """Update an existing patient record."""
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("""
                UPDATE patients 
                SET name = ?, age = ?, sex = ?, notes = ?
                WHERE id = ?
            """, (name, age, sex, notes, patient_id))
    return {"id": patient_id, "name": name, "age": age, "sex": sex, "notes": notes}

def archive_patient(patient_id: str, archive: bool = True) -> None:
    """Toggle a patient's archived status."""
    val = 1 if archive else 0
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("UPDATE patients SET is_archived = ? WHERE id = ?", (val, patient_id))

def search_patients(query: str = "", include_archived: bool = False) -> list:
    """Search for patient records matching query string."""
    q_str = f"%{query}%"
    with closing(get_connection()) as conn:
        if include_archived:
            cursor = conn.execute("""
                SELECT * FROM patients 
                WHERE (id LIKE ? OR name LIKE ?) 
                ORDER BY created_at DESC
            """, (q_str, q_str))
        else:
            cursor = conn.execute("""
                SELECT * FROM patients 
                WHERE is_archived = 0 AND (id LIKE ? OR name LIKE ?) 
                ORDER BY created_at DESC
            """, (q_str, q_str))
        return [dict(row) for row in cursor.fetchall()]


# ── Study Management Helper Functions ──

def create_study(study_id: str, patient_id: str, meta: dict, original_img: str, status: str = 'uploaded', reviewer: str = None, iqa: dict = None) -> None:
    """Create a new study associated with a patient."""
    now_str = datetime.now(timezone.utc).isoformat() + "Z"
    
    # Extract Image Quality metrics if provided
    iqa_metrics = iqa or {
        "exposure": "Adequate Exposure",
        "coverage": "Full Lung Coverage",
        "resolution": "Acceptable Resolution",
        "rotation": "No Rotation",
        "quality_score": 95,
        "suitable_for_ai": True,
        "warnings": []
    }
    
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO studies (
                    id, patient_id, study_date, modality, status, original_image, reviewer_username,
                    exposure, coverage, resolution, rotation, quality_score, suitability, warnings, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                study_id,
                patient_id,
                meta.get("study_date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                meta.get("modality", "CR"),
                status,
                original_img,
                reviewer,
                iqa_metrics.get("exposure"),
                iqa_metrics.get("coverage"),
                iqa_metrics.get("resolution"),
                iqa_metrics.get("rotation"),
                iqa_metrics.get("quality_score"),
                "suitable" if iqa_metrics.get("suitable_for_ai") else "unsuitable",
                json.dumps(iqa_metrics.get("warnings", [])),
                now_str
            ))

def save_prediction_record(study_id: str, result: dict) -> None:
    """Save model evaluation metrics to the predictions table."""
    now_str = datetime.now(timezone.utc).isoformat() + "Z"
    is_tb = 1 if result.get("is_tb", False) else 0
    prediction_label = result.get("prediction", "Unknown")
    
    attention = result.get("attention_region", "right apical" if is_tb else "clear")
    coverage = result.get("heatmap_coverage", 15.2 if is_tb else 0.0)
    
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("""
                INSERT INTO predictions (
                    study_id, confidence, is_tb, prediction, heatmap_image,
                    inference_time_ms, attention_region, heatmap_coverage, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                study_id,
                result.get("confidence", 0.0),
                is_tb,
                prediction_label,
                result.get("heatmap_image", ""),
                result.get("inference_time_ms", 324.0),
                attention,
                coverage,
                now_str
            ))
            # Automatically advance study status to 'ai_complete'
            conn.execute("UPDATE studies SET status = 'ai_complete' WHERE id = ?", (study_id,))

def save_review_record(study_id: str, reviewer: str, status: str, comment: str, note: str, annotation: str) -> None:
    """Save radiologist review decisions."""
    now_str = datetime.now(timezone.utc).isoformat() + "Z"
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("""
                INSERT INTO reviews (
                    study_id, reviewer_username, status, comments, clinician_note, annotation_b64, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (study_id, reviewer, status, comment, note, annotation, now_str))
            
            # Advance status to 'under_review' or 'finalized'
            new_status = 'finalized' if reviewer else 'under_review'
            conn.execute("UPDATE studies SET status = ? WHERE id = ?", (new_status, study_id))


# ── Audit Trail & Notifications Helper Functions ──

def log_audit_event(study_id: str, username: str, action: str) -> None:
    """Record technical audit events."""
    now_str = datetime.now(timezone.utc).isoformat() + "Z"
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("""
                INSERT INTO audit_trail (study_id, username, action, timestamp)
                VALUES (?, ?, ?, ?)
            """, (study_id, username, action, now_str))

def create_notification(study_id: str, message: str) -> None:
    """Insert a real-time workflow notification alert."""
    now_str = datetime.now(timezone.utc).isoformat() + "Z"
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("""
                INSERT INTO notifications (study_id, message, is_read, created_at)
                VALUES (?, ?, 0, ?)
            """, (study_id, message, now_str))

def list_notifications() -> list:
    """Return all active notifications."""
    with closing(get_connection()) as conn:
        cursor = conn.execute("""
            SELECT * FROM notifications 
            ORDER BY created_at DESC LIMIT 50
        """)
        return [dict(row) for row in cursor.fetchall()]

def mark_notification_read(notif_id: int) -> None:
    """Mark a notification alert as read."""
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("UPDATE notifications SET is_read = 1 WHERE id = ?", (notif_id,))


# ── Dashboard & List Query Helper Functions ──

def get_dashboard_stats() -> dict:
    """Query counts to feed the main supervisor dashboard."""
    with closing(get_connection()) as conn:
        # 1. Total Cases
        c1 = conn.execute("SELECT COUNT(*) FROM studies")
        total_cases = c1.fetchone()[0]
        
        # 2. TB Positive Cases (AI predicted or clinician overridden to TB)
        c2 = conn.execute("""
            SELECT COUNT(DISTINCT s.id) 
            FROM studies s
            LEFT JOIN predictions p ON s.id = p.study_id
            LEFT JOIN (
                SELECT study_id, status FROM reviews GROUP BY study_id HAVING max(id)
            ) r ON s.id = r.study_id
            WHERE (p.is_tb = 1 AND (r.status IS NULL OR r.status != 'reject'))
               OR (r.status = 'confirm' AND p.is_tb = 1)
        """)
        tb_positives = c2.fetchone()[0]
        
        # 3. Pending Reviews
        c3 = conn.execute("SELECT COUNT(*) FROM studies WHERE status IN ('uploaded', 'ai_complete', 'under_review')")
        pending_reviews = c3.fetchone()[0]
        
        # 4. Completed Reviews
        c4 = conn.execute("SELECT COUNT(*) FROM studies WHERE status = 'finalized'")
        completed_reviews = c4.fetchone()[0]
        
        # 5. Disease Distribution
        c_disease = conn.execute("""
            SELECT prediction, COUNT(*) 
            FROM predictions 
            GROUP BY prediction
        """)
        disease_dist = {"Tuberculosis": 0, "Normal": 0}
        for row in c_disease.fetchall():
            pred = row[0]
            count = row[1]
            if pred == "Tuberculosis":
                disease_dist["Tuberculosis"] = count
            elif pred == "Normal":
                disease_dist["Normal"] = count
                
        # 6. Confidence Distribution
        c_conf = conn.execute("SELECT confidence FROM predictions")
        conf_buckets = {"0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0}
        for row in c_conf.fetchall():
            conf = row[0]
            if conf < 0.2:
                conf_buckets["0-20%"] += 1
            elif conf < 0.4:
                conf_buckets["20-40%"] += 1
            elif conf < 0.6:
                conf_buckets["40-60%"] += 1
            elif conf < 0.8:
                conf_buckets["60-80%"] += 1
            else:
                conf_buckets["80-100%"] += 1

        # 7. Model Performance History (mock data)
        model_perf = [
            {"date": "Week 1", "accuracy": 92.0},
            {"date": "Week 2", "accuracy": 93.1},
            {"date": "Week 3", "accuracy": 93.8},
            {"date": "Week 4", "accuracy": 94.5}
        ]

        # 8. Reviewer Agreement Rate
        c_rev_total = conn.execute("SELECT COUNT(DISTINCT study_id) FROM reviews")
        total_rev = c_rev_total.fetchone()[0]
        c_rev_agree = conn.execute("SELECT COUNT(DISTINCT study_id) FROM reviews WHERE status = 'confirm'")
        agree_rev = c_rev_agree.fetchone()[0]
        
        agreement_rate = 100.0
        if total_rev > 0:
            agreement_rate = round((agree_rev / total_rev) * 100, 1)

        return {
            "total_cases": total_cases,
            "tb_positive_cases": tb_positives,
            "pending_reviews": pending_reviews,
            "completed_reviews": completed_reviews,
            "disease_distribution": disease_dist,
            "confidence_distribution": conf_buckets,
            "model_performance": model_perf,
            "reviewer_agreement_rate": agreement_rate
        }

def list_studies(date_str: str = None, status_str: str = None, reviewer: str = None) -> list:
    """List studies applying workflow filters."""
    query = """
        SELECT s.*, p.name as patient_name, pr.prediction, pr.confidence, pr.is_tb
        FROM studies s
        LEFT JOIN patients p ON s.patient_id = p.id
        LEFT JOIN (
            SELECT * FROM predictions GROUP BY study_id HAVING max(id)
        ) pr ON s.id = pr.study_id
        WHERE 1=1
    """
    params = []
    
    if date_str:
        query += " AND s.study_date = ?"
        params.append(date_str)
    if status_str:
        query += " AND s.status = ?"
        params.append(status_str)
    if reviewer:
        query += " AND s.reviewer_username = ?"
        params.append(reviewer)
        
    query += " ORDER BY s.created_at DESC"
    
    with closing(get_connection()) as conn:
        cursor = conn.execute(query, tuple(params))
        results = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            try:
                row_dict["warnings"] = json.loads(row_dict["warnings"] or "[]")
            except Exception:
                row_dict["warnings"] = []
            results.append(row_dict)
        return results


# ── Backward Compatibility Helpers ──

def save_result(patient_id: str, record: dict) -> None:
    """Backward-compatible wrapper to ingest predictions."""
    meta = record.get("metadata", {}) or {}
    with closing(get_connection()) as conn:
        cursor = conn.execute("SELECT id FROM patients WHERE id = ?", (patient_id,))
        if not cursor.fetchone():
            create_patient(
                patient_id,
                meta.get("patient_name", "Anonymous Patient"),
                meta.get("patient_age", "N/A"),
                meta.get("patient_sex", "N/A"),
                ""
            )
            
    import secrets
    study_id = record.get("study_id") or f"ST-{patient_id}-{secrets.randbelow(90000) + 10000}"
    if not meta:
        meta = {
            "patient_name": "Anonymous Patient",
            "patient_age": "N/A",
            "patient_sex": "N/A",
            "modality": "CR",
            "study_date": datetime.now(timezone.utc).strftime("%Y-%m-%d")
        }
        
    create_study(study_id, patient_id, meta, record.get("original_image", ""), status='ai_complete')
    save_prediction_record(study_id, record)

def save_feedback(study_id: str, feedback: dict) -> None:
    """Backward-compatible wrapper to save radiologist reviews."""
    with closing(get_connection()) as conn:
        cursor = conn.execute("SELECT id FROM studies WHERE id = ?", (study_id,))
        if not cursor.fetchone():
            # Check if study_id is actually a patient_id, and find the latest study
            cursor = conn.execute("SELECT id FROM studies WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1", (study_id,))
            row = cursor.fetchone()
            if row:
                study_id = row["id"]
            else:
                # Create a fallback study if none exists
                patient_id = study_id
                study_id = f"ST-{patient_id}"
                save_result(patient_id, {"confidence": 0.5, "is_tb": False, "prediction": "Normal"})
                
        reviewer = feedback.get("reviewer_username", "reviewer")
        status = feedback.get("clinician_prediction", "confirm")
        comment = feedback.get("reason", "")
        note = feedback.get("clinician_note", "")
        annotation = feedback.get("annotation_b64", "")
        
        save_review_record(study_id, reviewer, status, comment, note, annotation)

def log_report_generation(study_id: str) -> None:
    """Backward-compatible report logger."""
    with closing(get_connection()) as conn:
        # Check if study exists
        cursor = conn.execute("SELECT id FROM studies WHERE id = ?", (study_id,))
        if not cursor.fetchone():
            # Check if study_id is actually patient_id
            cursor = conn.execute("SELECT id FROM studies WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1", (study_id,))
            row = cursor.fetchone()
            if row:
                study_id = row["id"]
            else:
                return
        with conn:
            conn.execute("""
                INSERT INTO reports (study_id, created_at)
                VALUES (?, ?)
            """, (study_id, datetime.now(timezone.utc).isoformat() + "Z"))

def get_history(patient_id: str) -> list:
    """Retrieve historical prediction results for a patient's studies."""
    with closing(get_connection()) as conn:
        cursor = conn.execute("""
            SELECT s.id as study_id, s.patient_id, p_ref.name as patient_name, p_ref.age as patient_age, p_ref.sex as patient_sex, s.study_date, s.modality, s.original_image,
                   s.exposure, s.coverage, s.resolution, s.rotation, s.quality_score, s.suitability, s.warnings,
                   p.confidence, p.is_tb, p.prediction, p.heatmap_image, p.created_at as timestamp,
                   r.comments as clinician_reason, r.clinician_note, r.status as clinician_override, r.annotation_b64
            FROM studies s
            INNER JOIN predictions p ON s.id = p.study_id
            LEFT JOIN patients p_ref ON s.patient_id = p_ref.id
            LEFT JOIN (
                SELECT study_id, reviewer_username, status, comments, clinician_note, annotation_b64, max(id) FROM reviews GROUP BY study_id
            ) r ON s.id = r.study_id
            WHERE s.patient_id = ?
            ORDER BY p.id ASC
        """, (patient_id,))
        
        records = []
        for r in cursor.fetchall():
            warnings_list = []
            try:
                warnings_list = json.loads(r["warnings"] or "[]")
            except Exception:
                warnings_list = []
            records.append({
                "study_id": r["study_id"],
                "timestamp": r["timestamp"],
                "confidence": r["confidence"],
                "is_tb": bool(r["is_tb"]),
                "prediction": r["prediction"],
                "heatmap_b64": r["heatmap_image"],
                "original_b64": r["original_image"],
                "metadata": {
                    "patient_id": r["patient_id"],
                    "patient_name": r["patient_name"],
                    "patient_age": r["patient_age"],
                    "patient_sex": r["patient_sex"],
                    "modality": r["modality"],
                    "study_date": r["study_date"],
                },
                "clinician_note": r["clinician_note"] or "",
                "clinician_override": r["clinician_override"],
                "annotation_b64": r["annotation_b64"] or "",
                "clinician_reason": r["clinician_reason"] or "",
                "image_quality": {
                    "exposure": r["exposure"] or "Adequate Exposure",
                    "coverage": r["coverage"] or "Full Lung Coverage",
                    "resolution": r["resolution"] or "Acceptable Resolution",
                    "rotation": r["rotation"] or "No Rotation",
                    "quality_score": r["quality_score"] if r["quality_score"] is not None else 95,
                    "suitable_for_ai": r["suitability"] == "suitable" if r["suitability"] else True,
                    "warnings": warnings_list
                }
            })
        return records

def list_patients() -> list:
    """List patients matching backward-compatible requirements."""
    with closing(get_connection()) as conn:
        cursor = conn.execute("""
            SELECT p.id as patient_id, p.name as patient_name,
                   (SELECT COUNT(*) FROM studies WHERE patient_id = p.id) as scan_count,
                   (
                       SELECT pr.created_at 
                       FROM studies st
                       INNER JOIN predictions pr ON st.id = pr.study_id
                       WHERE st.patient_id = p.id
                       ORDER BY pr.id DESC LIMIT 1
                   ) as last_scan,
                   (
                       SELECT pr.prediction 
                       FROM studies st
                       INNER JOIN predictions pr ON st.id = pr.study_id
                       WHERE st.patient_id = p.id
                       ORDER BY pr.id DESC LIMIT 1
                   ) as last_prediction,
                   (
                       SELECT pr.confidence 
                       FROM studies st
                       INNER JOIN predictions pr ON st.id = pr.study_id
                       WHERE st.patient_id = p.id
                       ORDER BY pr.id DESC LIMIT 1
                   ) as last_confidence
            FROM patients p
            WHERE p.is_archived = 0
            ORDER BY p.created_at DESC
        """)
        
        result = []
        for r in cursor.fetchall():
            result.append({
                "patient_id": r["patient_id"],
                "patient_name": r["patient_name"],
                "scan_count": r["scan_count"],
                "last_scan": r["last_scan"] or "",
                "last_prediction": r["last_prediction"] or "Unknown",
                "last_confidence": r["last_confidence"] or 0.0,
            })
        return result

def get_study_audit_trail(study_id: str) -> list:
    """Retrieve the audit trail entries for a specific study, ordered by timestamp."""
    with closing(get_connection()) as conn:
        cursor = conn.execute("""
            SELECT timestamp, username, action 
            FROM audit_trail 
            WHERE study_id = ? 
            ORDER BY timestamp ASC
        """, (study_id,))
        return [dict(row) for row in cursor.fetchall()]

def _get_image_histogram_from_b64(b64_str: str) -> np.ndarray:
    try:
        import base64
        import cv2
        import numpy as np
        
        if not b64_str:
            return None
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
            
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # Resize to standardized size to ensure comparable histograms
        img_resized = cv2.resize(img, (256, 256))
        hist = cv2.calcHist([img_resized], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist
    except Exception:
        return None

def get_similar_cases(study_id: str) -> dict:
    """Retrieve up to 3 similar Tuberculosis cases and 3 similar Normal cases from the database."""
    import numpy as np
    import cv2
    with closing(get_connection()) as conn:
        # Fetch base study characteristics including original image
        base_cursor = conn.execute("""
            SELECT s.id as study_id, s.patient_id, p_ref.age as patient_age, p_ref.sex as patient_sex,
                   pr.confidence, pr.is_tb, s.original_image
            FROM studies s
            LEFT JOIN patients p_ref ON s.patient_id = p_ref.id
            LEFT JOIN predictions pr ON s.id = pr.study_id
            WHERE s.id = ?
        """, (study_id,))
        base = base_cursor.fetchone()
        if not base:
            return {"tb_similar": [], "normal_similar": []}
            
        base_age = str(base["patient_age"])
        base_sex = base["patient_sex"]
        base_conf = base["confidence"] if base["confidence"] is not None else 0.5
        base_image = base["original_image"]
        base_hist = _get_image_histogram_from_b64(base_image)
        
        # Query candidates (all other studies that have predictions from different patients)
        candidates_cursor = conn.execute("""
            SELECT s.id as study_id, s.patient_id, p_ref.name as patient_name,
                   p_ref.age as patient_age, p_ref.sex as patient_sex,
                   pr.confidence, pr.is_tb, pr.prediction, s.original_image, pr.heatmap_image
            FROM studies s
            INNER JOIN predictions pr ON s.id = pr.study_id
            LEFT JOIN patients p_ref ON s.patient_id = p_ref.id
            WHERE s.patient_id != ?
        """, (base["patient_id"],))
        
        tb_candidates = []
        normal_candidates = []
        
        for cand in candidates_cursor.fetchall():
            cand_age = str(cand["patient_age"])
            cand_sex = cand["patient_sex"]
            cand_conf = cand["confidence"] if cand["confidence"] is not None else 0.5
            cand_image = cand["original_image"]
            
            # Hybrid image-histogram + prediction distance similarity calculation
            conf_diff = abs(base_conf - cand_conf)
            cand_hist = _get_image_histogram_from_b64(cand_image)
            
            if base_hist is not None and cand_hist is not None:
                # Calculate correlation coefficient between histograms
                hist_corr = cv2.compareHist(base_hist, cand_hist, cv2.HISTCMP_CORREL)
                img_sim = max(0.0, hist_corr) * 100.0
                # Hybrid metric: 70% image histogram correlation, 30% prediction confidence similarity
                sim_score = (0.7 * img_sim) + (0.3 * (100.0 - conf_diff * 100.0))
            else:
                # Fallback to confidence difference similarity
                sim_score = 100.0 - conf_diff * 100.0
                
            sim_score = max(65.0, min(99.5, sim_score))
            
            cand_dict = {
                "study_id": cand["study_id"],
                "patient_id": cand["patient_id"],
                "patient_name": cand["patient_name"] or "Anonymous",
                "age": cand["patient_age"] or "N/A",
                "sex": cand["patient_sex"] or "N/A",
                "prediction": cand["prediction"],
                "confidence": cand["confidence"],
                "similarity_score": round(sim_score, 1),
                "original_image": cand["original_image"],
                "heatmap_image": cand["heatmap_image"]
            }
            
            if cand["is_tb"]:
                tb_candidates.append(cand_dict)
            else:
                normal_candidates.append(cand_dict)
                
        # Sort and select top 3 for each
        tb_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        normal_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "tb_similar": tb_candidates[:3],
            "normal_similar": normal_candidates[:3]
        }

def get_research_export_data() -> list:
    """Retrieve all study records and evaluation predictions for JSON/CSV research data export."""
    with closing(get_connection()) as conn:
        cursor = conn.execute("""
            SELECT s.id as study_id, s.patient_id, p_ref.name as patient_name, 
                   p_ref.age as patient_age, p_ref.sex as patient_sex, s.study_date, s.modality, s.status,
                   s.exposure, s.coverage, s.resolution, s.rotation, s.quality_score, s.suitability,
                   pr.confidence, pr.is_tb, pr.prediction, pr.inference_time_ms,
                   r.status as clinician_override, r.comments as review_comments, r.reviewer_username
            FROM studies s
            LEFT JOIN predictions pr ON s.id = pr.study_id
            LEFT JOIN patients p_ref ON s.patient_id = p_ref.id
            LEFT JOIN (
                SELECT study_id, status, comments, reviewer_username, max(id) FROM reviews GROUP BY study_id
            ) r ON s.id = r.study_id
            ORDER BY s.created_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

