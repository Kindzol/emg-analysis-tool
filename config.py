FS = 1000 # I NEED TO CHECK THIS VALUE, BECAUSE IT IS NOT MENTIONED IN THE DATASET
K = 0.75
WINDOW_MS = 200
TRIM_PERCENT = 0.1 # trim 10% from the start and end of each contraction

STATISTICS = {
    "raw": [
        ("samples", lambda e, f: f"{len(e):,}"),
        ("duration", lambda e, f: f"{len(e)/1000:.1f} s"),
        ("max amplitude", lambda e, f: f"{e['emg value'].abs().max():.3f} mV"),
    ],
    "onset": [
        ("contractions", lambda e, f: str(len(f))),
        ("threshold", lambda e, f: f"{e['threshold'].iloc[0]:.4f} mV"),
        ("avg duration", lambda e, f: f"{f['duration samples'].mean():.0f} smp" if len(f) else "—"),
    ],
    "features": [
        ("contractions", lambda e, f: str(len(f))),
        ("avg RMS", lambda e, f: f"{f['RMS'].mean():.4f} mV" if len(f) else "—"),
        ("avg MAV", lambda e, f: f"{f['MAV'].mean():.4f} mV" if len(f) else "—"),
    ],
}


VIEW_NAMES = {
    "raw": "raw_vs_filtered",
    "onset": "onset_detection",
    "features": "features",
}