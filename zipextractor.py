import zipfile
import os

zip_path = "wanted file"
extract_to = "wanted destination"

# Create the folder if it doesn't exist
os.makedirs(extract_to, exist_ok=True)

# Extract the ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Extraction complete. Files extracted to:", extract_to)
