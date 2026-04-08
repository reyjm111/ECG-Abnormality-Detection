from pathlib import Path
import shutil

ROOT = Path("/Users/reymendoza/Downloads/mit-bih-arrhythmia-database-1.0.0")

# File extensions that should be grouped with each participant ID
TARGET_EXTS = {".dat", ".hea", ".atr", ".xws"}

def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Folder does not exist: {ROOT}")

    moved_count = 0

    # Only process files directly inside ROOT
    for item in ROOT.iterdir():
        if not item.is_file():
            continue

        ext = item.suffix.lower()
        if ext not in TARGET_EXTS:
            continue

        pid = item.stem  # e.g. "001" from "001.dat"
        participant_dir = ROOT / pid
        participant_dir.mkdir(exist_ok=True)

        destination = participant_dir / item.name

        if destination.exists():
            print(f"Skipping, already exists: {destination}")
            continue

        print(f"Moving {item.name} -> {participant_dir.name}/")
        shutil.move(str(item), str(destination))
        moved_count += 1

    print(f"\nDone. Moved {moved_count} file(s).")

if __name__ == "__main__":
    main()