
import os
import pandas as pd

# Define paths
CSV_PATH = "/Users/abhinandan/Desktop/PEOJECT/MMTD++_Enhanced_Code/edp_dataset.csv"
IMG_DIR = "/Users/abhinandan/Desktop/PEOJECT/MMTD++_Enhanced_Code/pics"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Initialize tracker
valid_rows = []
missing_images = []

# Check each image path
for _, row in df.iterrows():
    image_rel = row['image_path']
    full_path = os.path.join(IMG_DIR, image_rel)
    if os.path.exists(full_path):
        valid_rows.append(row)
    else:
        missing_images.append(image_rel)

# Create cleaned DataFrame
clean_df = pd.DataFrame(valid_rows)

# Save clean CSV
clean_df.to_csv(CSV_PATH, index=False, encoding='utf-8')
print(f"✅ Cleaned CSV saved. Valid rows: {len(clean_df)}, Missing images: {len(missing_images)}")

# Show a few missing files if needed
if missing_images:
    print("❌ Example missing image paths:")
    for m in missing_images[:10]:
        print(m)
