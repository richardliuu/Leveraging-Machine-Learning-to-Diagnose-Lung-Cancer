import zipfile
import os

zip_path = r"C:\Users\richa\OneDrive\Desktop\lung_dataset\healthy-20250430T115637Z-001.zip"
extract_to = r"C:\Users\richa\OneDrive\Desktop\science2\wavfiles"  # Choose a destination

# Create the folder if it doesn't exist
os.makedirs(extract_to, exist_ok=True)

# Extract the ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Extraction complete. Files extracted to:", extract_to)
