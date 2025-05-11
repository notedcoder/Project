import pandas as pd

csv_path = "/Users/abhinandan/Desktop/PEOJECT/MMTD++_Enhanced_Code/edp_dataset.csv"

# Try reading with fallback encoding and ignore broken lines
df = pd.read_csv(csv_path, encoding='ISO-8859-1', on_bad_lines='skip')

print("🔍 Raw headers:", df.columns.tolist())
print("🧪 Sample rows:\n", df.head())

# Step 1: If your data only has 1 column (e.g. everything dumped in 1 column)
if len(df.columns) == 1:
    print("❌ CSV is corrupted: single-column format. Cannot continue.")
    print("Please open the CSV manually and check the format (should be: text,image_path,label).")
    exit()

# Step 2: Rename columns if needed
expected = {'text', 'image_path', 'label'}
if not expected.issubset(set(df.columns)):
    rename_map = {}
    if 'message' in df.columns: rename_map['message'] = 'text'
    if 'filename' in df.columns: rename_map['filename'] = 'image_path'
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    else:
        print("❌ Could not find expected columns. Please verify manually.")
        exit()

# Step 3: Save clean CSV
df[['text', 'image_path', 'label']].to_csv(csv_path, index=False, encoding='utf-8')
print("✅ Cleaned and saved CSV to UTF-8:", csv_path)
