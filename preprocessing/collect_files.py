import os
import re
import wfdb
from preprocess import preprocess_ecg

def collect(base_dir):
    files = []
    record_paths = []

    excluded_records = {"102", "104"}

    for entry in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, entry)

        if not os.path.isdir(folder_path):
            continue

        if not re.fullmatch(r"\d+", entry):  # only numbered folders
            continue

        if entry in excluded_records:
            continue

        edf_path = os.path.join(folder_path, f"{entry}.edf")
        atr_path = os.path.join(folder_path, f"{entry}.atr")
        record_path = os.path.join(folder_path, entry)  # wfdb wants path without extension

        if os.path.exists(edf_path) and os.path.exists(atr_path):
            files.append(edf_path)
            record_paths.append(record_path)

    print(f"Found {len(files)} valid records.")
    print(f"Excluded records: {sorted(excluded_records)}")

    return files, record_paths