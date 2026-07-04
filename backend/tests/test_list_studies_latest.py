"""
Regression test for Bug #7: list_studies GROUP BY prediction subquery.

The list_studies query used `SELECT * FROM predictions GROUP BY study_id
HAVING max(id)`, which is non-deterministic in SQLite — it returns an
arbitrary row per study rather than the latest. After retries or reanalysis,
list_studies may surface an OLD prediction's confidence/prediction instead
of the newest one.

This test bypasses the Flask app import (which pulls in keras/torch) and
exercises utils.patient_db directly.
"""
import os
import sys
import tempfile

# Ensure backend/ is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Redirect DB to a temp file BEFORE importing patient_db so we don't touch
# the real production DB.
_tmpdir = tempfile.mkdtemp(prefix="patient_db_test_")
os.environ["DB_FILE_OVERRIDE"] = os.path.join(_tmpdir, "patients.db")

# patient_db reads DB_FILE at import time. Patch it before init_db.
import utils.patient_db as pdb  # noqa: E402
pdb.DB_FILE = os.environ["DB_FILE_OVERRIDE"]
pdb.init_db()


def test_list_studies_picks_latest_prediction():
    patient_id = "PX-BUG7-TEST"

    # Clean any prior fixture rows.
    with pdb.get_connection() as conn:
        with conn:
            conn.execute("DELETE FROM predictions WHERE study_id IN (SELECT id FROM studies WHERE patient_id = ?)", (patient_id,))
            conn.execute("DELETE FROM studies WHERE patient_id = ?", (patient_id,))
            conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))

    # Seed MANY predictions for the same study with widely varying
    # confidences so the bug (returning an arbitrary row) is exposed.
    # The OLDEST row will have confidence 0.10; the NEWEST will have 0.99.
    pdb.save_result(patient_id, {
        "confidence": 0.10,
        "is_tb": True,
        "prediction": "Tuberculosis",
        "metadata": {"patient_name": "Regression Patient"},
    })
    history = pdb.get_history(patient_id)
    seeded_study_id = history[0]["study_id"]
    assert seeded_study_id, "save_result did not return a study_id"

    # Insert 5 more predictions with strictly increasing created_at.
    # The last one (newest) will be id=N with confidence 0.99.
    newer_confidences = [0.25, 0.40, 0.60, 0.80, 0.99]
    base_ts = "2099-01-01T00:00:00Z"
    for i, c in enumerate(newer_confidences, start=1):
        with pdb.get_connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO predictions (
                        study_id, confidence, is_tb, prediction, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (seeded_study_id, c, True, "Tuberculosis", f"2099-01-0{i}T00:00:00Z"),
                )

    # Now repeatedly call list_studies. The fixed implementation must
    # ALWAYS return 0.99 (the latest). The buggy implementation will
    # sometimes return older values.
    observed = set()
    for _ in range(50):
        rows = pdb.list_studies()
        target = [r for r in rows if r.get("id") == seeded_study_id]
        assert target, f"study {seeded_study_id} missing from list_studies"
        observed.add(target[0]["confidence"])

    # Strict assertion: the ONLY observed confidence must be 0.99 (latest).
    # If the bug is present, we'll see 0.10 / 0.25 / 0.40 / etc.
    assert observed == {0.99}, (
        f"list_studies returned multiple distinct confidences across 50 calls: "
        f"{sorted(observed)}. Expected only {{0.99}} (the latest prediction). "
        "Bug #7 unfixed: GROUP BY subquery is non-deterministic."
    )

    # Cleanup fixture rows.
    with pdb.get_connection() as conn:
        with conn:
            conn.execute("DELETE FROM predictions WHERE study_id = ?", (seeded_study_id,))
            conn.execute("DELETE FROM studies WHERE id = ?", (seeded_study_id,))
            conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))