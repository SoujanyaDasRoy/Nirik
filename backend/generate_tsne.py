import json
import random
import os
import math

def generate_mock_tsne():
    datasets = ["Montgomery", "Shenzhen", "TB-Database", "NIRT"]
    data = []
    
    # Centers for Normal
    normal_centers = {
        "Montgomery": (-10, -5),
        "Shenzhen": (-8, 5),
        "TB-Database": (-12, 0),
        "NIRT": (-5, -8)
    }
    
    # Centers for TB
    tb_centers = {
        "Montgomery": (10, 5),
        "Shenzhen": (8, -5),
        "TB-Database": (12, 0),
        "NIRT": (5, 8)
    }
    
    for ds in datasets:
        # Generate Normal points
        nx, ny = normal_centers[ds]
        for _ in range(50):
            data.append({
                "x": nx + random.gauss(0, 3),
                "y": ny + random.gauss(0, 3),
                "dataset": ds,
                "label": "Normal",
                "patient_id": f"P-{random.randint(1000, 9999)}"
            })
            
        # Generate TB points
        tx, ty = tb_centers[ds]
        for _ in range(30):
            data.append({
                "x": tx + random.gauss(0, 3),
                "y": ty + random.gauss(0, 3),
                "dataset": ds,
                "label": "Tuberculosis",
                "patient_id": f"P-{random.randint(1000, 9999)}"
            })
            
    # Add some outliers
    for _ in range(20):
        data.append({
            "x": random.uniform(-20, 20),
            "y": random.uniform(-20, 20),
            "dataset": random.choice(datasets),
            "label": random.choice(["Normal", "Tuberculosis"]),
            "patient_id": f"P-{random.randint(1000, 9999)}"
        })
        
    out_path = os.path.join(os.path.dirname(__file__), "tsne_embeddings.json")
    with open(out_path, "w") as f:
        json.dump(data, f)
    
    print(f"Generated {len(data)} mock t-SNE points at {out_path}")

if __name__ == "__main__":
    generate_mock_tsne()
