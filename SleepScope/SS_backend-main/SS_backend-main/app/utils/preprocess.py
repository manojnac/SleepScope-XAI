# import numpy as np
# import mne
# import os

# def read_hypnogram(hyp_file):
#     """
#     Reads a hypnogram from TXT or CSV.
#     Expected format: one sleep stage label per line or per row.
#     Returns a 1D array of integers.
#     """

#     try:
#         import pandas as pd

#         # Try CSV first
#         if hyp_file.endswith(".csv"):
#             df = pd.read_csv(hyp_file, header=None)
#         else:
#             df = pd.read_csv(hyp_file, header=None, delim_whitespace=True)

#         values = df.iloc[:, 0].values.astype(int)
#         return values

#     except Exception:
#         # Fallback to manual line reading
#         with open(hyp_file, "r") as f:
#             lines = f.readlines()
#             values = []
#             for line in lines:
#                 line = line.strip()
#                 if line.isdigit():
#                     values.append(int(line))
#         return np.array(values, dtype=int)


# def extract_psg_features(edf_path, hyp_path):
#     """
#     Extracts PSG features EXACTLY as used during model training.

#     Features returned:
#         1. TST_hours
#         2. WASO_minutes
#         3. SOL_minutes
#         4. N1_minutes
#         5. N2_minutes
#         6. N3_minutes
#         7. REM_minutes
#         8. Sleep_Efficiency
#         9. Total_Time_hours
#     """

#     # -------------------------
#     # Load PSG (EDF) signals
#     # -------------------------
#     raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

#     # -------------------------
#     # Load hypnogram
#     # -------------------------
#     hyp = read_hypnogram(hyp_path)

#     # Filter invalid labels (training code does this)
#     valid = hyp[hyp >= 0]

#     # -------------------------
#     # Sleep Stage Durations
#     # -------------------------
#     # Each label represents a 30-second epoch
#     TST = (np.sum(valid != 0) * 30) / 3600       # hours
#     WASO = (np.sum(valid == 0) * 30) / 60        # minutes

#     N1 = np.sum(valid == 1) * 30 / 60
#     N2 = np.sum(valid == 2) * 30 / 60
#     N3 = np.sum(valid == 3) * 30 / 60
#     REM = np.sum(valid == 4) * 30 / 60

#     # -------------------------
#     # Sleep Latency (SOL)
#     # -------------------------
#     try:
#         SOL = list(valid).index(1) * 30 / 60   # first N1 event
#     except ValueError:
#         SOL = np.nan

#     # -------------------------
#     # Sleep Efficiency
#     # -------------------------
#     total_time = len(valid) * 30 / 3600      # hours
#     SE = (TST / total_time) * 100 if total_time > 0 else 0

#     # -------------------------
#     # Return structured feature dict
#     # -------------------------
#     return {
#         "TST_hours": float(TST),
#         "WASO_minutes": float(WASO),
#         "SOL_minutes": float(SOL) if SOL == SOL else 0.0,  # Replace NaN with 0
#         "N1_minutes": float(N1),
#         "N2_minutes": float(N2),
#         "N3_minutes": float(N3),
#         "REM_minutes": float(REM),
#         "Sleep_Efficiency": float(SE),
#         "Total_Time_hours": float(total_time)
#     }


import numpy as np
import mne

def read_hypnogram(path):
    ann = mne.read_annotations(path)
    STAGE_MAP = {'W':0,'1':1,'2':2,'3':3,'4':3,'R':4,'M':-1,'?':-1}
    
    hyp = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        label = desc.replace("Sleep stage ", "").strip()
        stage = STAGE_MAP.get(label, -1)
        epochs = int(dur / 30)
        hyp.extend([stage] * epochs)

    return np.array(hyp)


def extract_psg_features(psg_file, hyp_file=None):
    """
    Extracts the same 10 features used during model training.
    """
    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)

    if hyp_file is None:
        raise ValueError("Hypnogram file is required for correct feature extraction.")

    hyp = read_hypnogram(hyp_file)
    valid = hyp[hyp >= 0]

    # Durations
    TST = (np.sum(valid != 0) * 30) / 3600
    WASO = (np.sum(valid == 0) * 30) / 60
    N1 = np.sum(valid == 1) * 30 / 60
    N2 = np.sum(valid == 2) * 30 / 60
    N3 = np.sum(valid == 3) * 30 / 60
    REM = np.sum(valid == 4) * 30 / 60

    total_time = len(valid) * 30 / 3600
    SE = (TST / total_time) * 100

    # Ratios (must match training order)
    N1_ratio = N1 / (TST * 60)
    N2_ratio = N2 / (TST * 60)
    N3_ratio = N3 / (TST * 60)
    REM_ratio = REM / (TST * 60)

    WASO_rate = WASO / (total_time * 60)
    SOL = next((i for i, x in enumerate(valid) if x == 1), -1)
    SOL_minutes = SOL * 30 / 60
    SOL_rate = SOL_minutes / total_time

    Fragmentation = (N1 + WASO) / (TST * 60)

    # Return EXACT 10 feature values in the correct order:
    return {
        "TST_hours": TST,
        "WASO_rate": WASO_rate,
        "SOL_rate": SOL_rate,
        "N1_ratio": N1_ratio,
        "N2_ratio": N2_ratio,
        "N3_ratio": N3_ratio,
        "REM_ratio": REM_ratio,
        "Sleep_Efficiency": SE,
        "Fragmentation": Fragmentation,
        "Total_Time_hours": total_time
    }
