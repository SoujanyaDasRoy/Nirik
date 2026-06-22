import json

nb_path = r"C:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\__notebook_source__.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

print("Searching notebook cells...")
for idx, cell in enumerate(notebook.get("cells", [])):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if any(term in source for term in ["preprocess_input", "normalize", "ImageDataGenerator", "rescale", "resize", "std", "mean"]):
            print(f"\n--- Cell {idx} ---")
            print(source[:800])
            print("...")
