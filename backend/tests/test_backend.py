import os
import sys
import pytest
import io
from PIL import Image

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from utils.dicom_parser import process_dicom, extract_metadata
from utils.image_helpers import process_standard_image, image_to_base64
from utils.patient_db import save_result, get_history
from utils.fhir_mock import search_patients, get_pacs_status

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_dicom_parser_malformed():
    with pytest.raises(Exception):
        process_dicom(b"malformed DICOM bytes")

def test_image_helpers_malformed():
    with pytest.raises(Exception):
        process_standard_image(b"malformed image bytes")

def test_image_helpers_base64():
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    b64_str = image_to_base64(img)
    assert b64_str.startswith("data:image/png;base64,")

def test_fhir_mock():
    patients = search_patients("Arjun", count=3)
    assert len(patients) == 3
    assert all("Patient" == p["resourceType"] for p in patients)
    
    pacs = get_pacs_status()
    assert len(pacs) > 0
    assert "name" in pacs[0]
    assert "status" in pacs[0]

def test_patient_db():
    test_id = "PX-TEST-12345"
    record = {
        "confidence": 0.85,
        "is_tb": True,
        "prediction": "Tuberculosis",
        "heatmap_image": "heatmap_bytes",
        "original_image": "original_bytes",
        "metadata": {
            "patient_name": "Test Patient",
            "patient_age": "45"
        }
    }
    save_result(test_id, record)
    history = get_history(test_id)
    assert len(history) >= 1
    assert history[-1]["confidence"] == 0.85
    assert history[-1]["is_tb"] is True
    assert history[-1]["prediction"] == "Tuberculosis"
    assert history[-1]["metadata"]["patient_name"] == "Test Patient"

def test_predict_invalid_file(client):
    # Authenticate client session
    with client.session_transaction() as sess:
        sess["username"] = "reviewer"
        sess["role"] = "reviewer"

    # Send empty payload
    response = client.post("/predict")
    assert response.status_code == 400
    assert b"No file uploaded" in response.data

    # Send invalid filename/empty file
    data = {"file": (io.BytesIO(b""), "")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Empty filename" in response.data

    # Send text file (which fails and raises Exception)
    data = {"file": (io.BytesIO(b"this is a text file"), "test.txt")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 500
    # FLASK_DEBUG off by default gates detailed error, returns "Internal server error"
    assert b"Internal server error" in response.data

def test_csrf_protection(client):
    # Pre-populate dummy record so feedback endpoint has a record to update
    save_result("PX-12345", {"confidence": 0.5, "is_tb": False, "prediction": "Normal"})

    # Verify POST /feedback fails without CSRF headers
    response = client.post("/feedback", json={"patient_id": "PX-12345"})
    assert response.status_code == 403
    assert b"CSRF validation failed" in response.data

    # Verify POST /patients/PX-12345/save fails without CSRF headers
    response = client.post("/patients/PX-12345/save", json={"confidence": 0.5})
    assert response.status_code == 403
    assert b"CSRF validation failed" in response.data
    
    # Use a fresh test client to guarantee a clean cookie jar
    fresh_client = app.test_client()
    health_response = fresh_client.get("/health")
    csrf_cookie = None
    for cookie_header in health_response.headers.getlist("Set-Cookie"):
        if "csrf_token=" in cookie_header:
            csrf_cookie = cookie_header.split(";")[0].split("=")[1]
            break
            
    assert csrf_cookie is not None, "csrf_token cookie not set by backend"
    
    # Authenticate fresh_client session
    with fresh_client.session_transaction() as sess:
        sess["username"] = "reviewer"
        sess["role"] = "reviewer"

    # Send valid POST request with matching header and cookie
    fresh_client.set_cookie("csrf_token", csrf_cookie)
    headers = {"X-CSRF-Token": csrf_cookie}
    
    feedback_response = fresh_client.post(
        "/feedback", 
        json={"patient_id": "PX-12345", "clinician_prediction": "Normal"},
        headers=headers
    )
    assert feedback_response.status_code == 200

def test_root_index(client):
    response = client.get("/")
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json["status"] == "active"
    assert "PulmonaryAI API Gateway" in res_json["message"]

def test_patient_crud(client):
    # Authenticate client
    with client.session_transaction() as sess:
        sess["username"] = "admin"
        sess["role"] = "admin"

    # Clean up prior test runs
    from utils.patient_db import get_connection
    with get_connection() as conn:
        with conn:
            conn.execute("DELETE FROM patients WHERE id = ?", ("PX-P2-999",))

    # Create patient
    response = client.post("/patients", json={
        "id": "PX-P2-999",
        "name": "Jane Doe",
        "age": "32",
        "sex": "Female",
        "notes": "Healthy clinical volunteer."
    })
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json["success"] is True
    assert res_json["patient"]["name"] == "Jane Doe"

    # Edit patient
    response = client.put("/patients/PX-P2-999", json={
        "name": "Jane Smith",
        "age": "33",
        "sex": "Female",
        "notes": "Updated notes."
    })
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json["success"] is True
    assert res_json["patient"]["name"] == "Jane Smith"

    # Search patient
    response = client.get("/patients?search=Smith")
    assert response.status_code == 200
    res_json = response.get_json()
    assert len(res_json["patients"]) >= 1
    assert res_json["patients"][0]["name"] == "Jane Smith"

    # Archive patient
    response = client.post("/patients/PX-P2-999/archive", json={"archive": True})
    assert response.status_code == 200
    assert response.get_json()["archived"] is True

    # Check search excluding archived
    response = client.get("/patients?search=Smith")
    assert response.status_code == 200
    assert len(response.get_json()["patients"]) == 0

def test_dashboard_stats(client):
    with client.session_transaction() as sess:
        sess["username"] = "admin"
        sess["role"] = "admin"

    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    res_json = response.get_json()
    assert "total_cases" in res_json
    assert "tb_positive_cases" in res_json
    assert "pending_reviews" in res_json
    assert "completed_reviews" in res_json

def test_notifications(client):
    with client.session_transaction() as sess:
        sess["username"] = "admin"
        sess["role"] = "admin"

    response = client.get("/notifications")
    assert response.status_code == 200
    res_json = response.get_json()
    assert "notifications" in res_json

def test_study_audit_trail(client):
    with client.session_transaction() as sess:
        sess["username"] = "admin"
        sess["role"] = "admin"

    # Prepopulate a study audit event
    from utils.patient_db import log_audit_event, get_connection
    with get_connection() as conn:
        with conn:
            conn.execute("DELETE FROM audit_trail WHERE study_id = ?", ("STUDY-AUDIT-TEST",))
    
    log_audit_event("STUDY-AUDIT-TEST", "admin", "Upload")
    log_audit_event("STUDY-AUDIT-TEST", "system", "Inference")

    response = client.get("/studies/STUDY-AUDIT-TEST/audit")
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json["study_id"] == "STUDY-AUDIT-TEST"
    assert len(res_json["audit_trail"]) == 2
    assert res_json["audit_trail"][0]["action"] == "Upload"
    assert res_json["audit_trail"][0]["username"] == "admin"
    assert res_json["audit_trail"][1]["action"] == "Inference"

def test_model_metadata(client):
    with client.session_transaction() as sess:
        sess["username"] = "reviewer"
        sess["role"] = "reviewer"
    response = client.get("/model/metadata")
    assert response.status_code == 200
    res_json = response.get_json()
    assert "metrics" in res_json
    assert "accuracy" in res_json["metrics"]
    assert "dataset_tracking" in res_json
    assert "model_version" in res_json["dataset_tracking"]

def test_similar_cases(client):
    with client.session_transaction() as sess:
        sess["username"] = "reviewer"
        sess["role"] = "reviewer"
        
    # Prepopulate database records to retrieve similar cases
    from utils.patient_db import save_result
    save_result("PX-SIM-BASE", {"confidence": 0.88, "is_tb": True, "prediction": "Tuberculosis"})
    save_result("PX-SIM-MATCH", {"confidence": 0.85, "is_tb": True, "prediction": "Tuberculosis", "metadata": {"patient_name": "Match One"}})
    
    # We need base study_id. Let's find it.
    from utils.patient_db import get_history
    base_history = get_history("PX-SIM-BASE")
    assert len(base_history) > 0
    base_study_id = base_history[0]["study_id"]
    
    response = client.get(f"/studies/{base_study_id}/similar")
    assert response.status_code == 200
    res_json = response.get_json()
    assert "tb_similar" in res_json
    assert "normal_similar" in res_json
    # Should have matched PX-SIM-MATCH
    assert len(res_json["tb_similar"]) >= 1
    assert any(c["patient_id"] == "PX-SIM-MATCH" for c in res_json["tb_similar"])

def test_export_research(client):
    with client.session_transaction() as sess:
        sess["username"] = "reviewer"
        sess["role"] = "reviewer"
        
    # Export JSON
    response = client.get("/export/research?format=json")
    assert response.status_code == 200
    res_json = response.get_json()
    assert "studies" in res_json
    assert len(res_json["studies"]) > 0
    
    # Export CSV
    response = client.get("/export/research?format=csv")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv"
    assert b"study_id,patient_id,patient_name" in response.data

def test_dashboard_stats_extended(client):
    with client.session_transaction() as sess:
        sess["username"] = "reviewer"
        sess["role"] = "reviewer"
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    res_json = response.get_json()
    assert "disease_distribution" in res_json
    assert "confidence_distribution" in res_json
    assert "model_performance" in res_json
    assert "reviewer_agreement_rate" in res_json
