from pathlib import Path
import numpy as np
import wfdb
import pyedflib

ROOT = Path("/Users/reymendoza/Downloads/mit-bih-arrhythmia-database-1.0.0")
OVERWRITE = False  # change to True if you want to replace existing EDFs

def infer_dimension(record, channel_index: int) -> str:
    """
    Try to infer a sensible physical unit for EDF metadata.
    Defaults to mV for ECG-style PhysioNet records if not available.
    """
    units = getattr(record, "units", None)
    if units and channel_index < len(units) and units[channel_index]:
        return str(units[channel_index])
    return "mV"

def convert_record_folder(folder: Path):
    pid = folder.name
    hea = folder / f"{pid}.hea"
    dat = folder / f"{pid}.dat"
    edf = folder / f"{pid}.edf"

    if not hea.exists() or not dat.exists():
        print(f"Skipping {pid}: missing .hea or .dat")
        return

    if edf.exists() and not OVERWRITE:
        print(f"Skipping {pid}: EDF already exists")
        return

    # wfdb.rdrecord expects the record path without extension
    record_path = str(folder / pid)
    record = wfdb.rdrecord(record_path)

    if record.p_signal is None:
        raise ValueError(f"{pid}: record.p_signal is None")

    signals = record.p_signal  # shape: (n_samples, n_channels)
    fs = float(record.fs)
    ch_names = list(record.sig_name)

    if signals.ndim != 2:
        raise ValueError(f"{pid}: expected 2D signal array, got shape {signals.shape}")

    n_samples, n_channels = signals.shape
    channel_data = [signals[:, i].astype(np.float64) for i in range(n_channels)]

    signal_headers = []
    for i in range(n_channels):
        sig = channel_data[i]

        physical_min = float(np.min(sig))
        physical_max = float(np.max(sig))

        # Avoid equal min/max, which can break EDF writing
        if physical_min == physical_max:
            physical_min -= 1.0
            physical_max += 1.0

        header = {
            "label": str(ch_names[i])[:16],  # EDF label length is limited
            "dimension": infer_dimension(record, i),
            "sample_frequency": fs,
            "physical_min": physical_min,
            "physical_max": physical_max,
            "digital_min": -32768,
            "digital_max": 32767,
            "transducer": "",
            "prefilter": "",
        }
        signal_headers.append(header)

    writer = pyedflib.EdfWriter(
        str(edf),
        n_channels=n_channels,
        file_type=pyedflib.FILETYPE_EDFPLUS,
    )

    try:
        writer.setSignalHeaders(signal_headers)
        writer.writeSamples(channel_data)
    finally:
        writer.close()

    print(f"Created: {edf}")

def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Root folder not found: {ROOT}")

    participant_dirs = [p for p in ROOT.iterdir() if p.is_dir() and p.name.isdigit()]
    participant_dirs.sort(key=lambda p: int(p.name))

    if not participant_dirs:
        print("No participant folders found.")
        return

    for folder in participant_dirs:
        try:
            convert_record_folder(folder)
        except Exception as e:
            print(f"Error in {folder.name}: {e}")

if __name__ == "__main__":
    main()