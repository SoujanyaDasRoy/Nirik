"""
fhir_mock.py — Simulated HL7 FHIR R4-compatible patient data for EHR integration demo.
No external faker library needed — uses curated realistic Indian medical records.
"""
import random
import uuid
from datetime import date, timedelta

# ── Realistic Indian Patient Name Banks ─────────────────────────────────────
FIRST_NAMES_MALE = [
    "Arjun", "Rohan", "Suresh", "Vikram", "Pradeep", "Rahul", "Anil",
    "Rajesh", "Mohan", "Sanjay", "Dinesh", "Ganesh", "Ramesh", "Kiran",
    "Ashok", "Deepak", "Harish", "Manoj", "Nikhil", "Pavan"
]
FIRST_NAMES_FEMALE = [
    "Priya", "Sunita", "Anita", "Kavya", "Meena", "Rekha", "Deepa",
    "Lalitha", "Usha", "Nandini", "Pooja", "Sneha", "Divya", "Asha",
    "Lakshmi", "Geeta", "Savita", "Radha", "Shobha", "Ranjini"
]
LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Reddy", "Nair", "Iyer",
    "Verma", "Gupta", "Mehta", "Joshi", "Pillai", "Rao", "Bose",
    "Chatterjee", "Mukherjee", "Das", "Mishra", "Tiwari", "Kapoor"
]

SYMPTOMS = [
    "Persistent cough > 2 weeks",
    "Haemoptysis",
    "Night sweats",
    "Unexplained weight loss",
    "Fever of unknown origin",
    "Chest pain on inspiration",
    "Fatigue and malaise",
    "Dyspnoea at rest",
    "Loss of appetite",
    "Productive cough with sputum"
]

REFERRING_DOCTORS = [
    "Dr. A. Krishnaswamy (Pulmonology)",
    "Dr. S. Mehta (General Medicine)",
    "Dr. P. Sharma (Infectious Diseases)",
    "Dr. R. Nair (Emergency Medicine)",
    "Dr. V. Gupta (Respiratory Medicine)",
]

HOSPITALS = [
    "AIIMS New Delhi",
    "Fortis Healthcare",
    "Apollo Hospitals",
    "Narayana Health",
    "Manipal Hospital",
    "KEM Hospital Mumbai",
    "PGIMER Chandigarh",
]


def _random_dob(min_age: int = 18, max_age: int = 75, rng=None) -> str:
    if rng is None:
        rng = random
    today = date.today()
    age_days = rng.randint(min_age * 365, max_age * 365)
    dob = today - timedelta(days=age_days)
    return dob.isoformat()


def _compute_age(dob_str: str) -> int:
    dob = date.fromisoformat(dob_str)
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def generate_patient(seed_name: str = "", rng=None) -> dict:
    """Generate a single mock FHIR Patient resource."""
    if rng is None:
        rng = random
    gender = rng.choice(["male", "female"])
    first = rng.choice(FIRST_NAMES_MALE if gender == "male" else FIRST_NAMES_FEMALE)
    last = rng.choice(LAST_NAMES)

    # If a seed name was given, bias toward it
    if seed_name:
        for word in seed_name.split():
            word_cap = word.capitalize()
            if any(word_cap in n for n in FIRST_NAMES_MALE + FIRST_NAMES_FEMALE):
                first = word_cap
            if any(word_cap in n for n in LAST_NAMES):
                last = word_cap

    dob = _random_dob(rng=rng)
    age = _compute_age(dob)
    mrn = f"MRN-{rng.randint(100000, 999999)}"
    symptoms = rng.sample(SYMPTOMS, k=rng.randint(2, 4))

    return {
        "resourceType": "Patient",
        "id": str(uuid.uuid4())[:8].upper(),
        "mrn": mrn,
        "name": f"{first} {last}",
        "given": first,
        "family": last,
        "gender": gender,
        "birthDate": dob,
        "age": age,
        "presenting_symptoms": symptoms,
        "referring_doctor": rng.choice(REFERRING_DOCTORS),
        "hospital": rng.choice(HOSPITALS),
        "blood_group": rng.choice(["A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-"]),
    }


def search_patients(query: str, count: int = 8) -> list:
    """Return a list of mock patients that 'match' the search query (thread-safe)."""
    # Seed local Random instance based on query to make results predictable yet thread-safe
    if query:
        # Use sum of chars as integer seed
        seed_val = sum(ord(c) for c in query.lower().strip())
    else:
        seed_val = 42
    rng = random.Random(seed_val)
    patients = [generate_patient(seed_name=query, rng=rng) for _ in range(count)]
    return patients


def get_pacs_status() -> list:
    """Return a mock list of DICOM PACS node statuses (thread-safe)."""
    rng = random.Random()
    nodes = [
        {"name": "PACS-Primary-Node", "host": "192.168.1.101", "port": 4242, "ae_title": "PACS_PRIMARY"},
        {"name": "Worklist-Server", "host": "192.168.1.102", "port": 4243, "ae_title": "WORKLIST"},
        {"name": "Archive-Server", "host": "192.168.1.103", "port": 4244, "ae_title": "ARCHIVE"},
        {"name": "DR-Station-01", "host": "192.168.1.201", "port": 104, "ae_title": "DR01"},
        {"name": "DR-Station-02", "host": "192.168.1.202", "port": 104, "ae_title": "DR02"},
        {"name": "RIS-Server", "host": "192.168.1.110", "port": 8080, "ae_title": "RIS"},
    ]
    statuses = ["online", "online", "online", "online", "degraded", "offline"]
    for i, node in enumerate(nodes):
        node["status"] = statuses[i % len(statuses)]
        node["ping_ms"] = rng.randint(2, 45) if node["status"] == "online" else None
        node["last_seen"] = "Just now" if node["status"] == "online" else f"{rng.randint(5, 60)} min ago"
    return nodes
