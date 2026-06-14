import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from config import FS, K, WINDOW_MS, TRIM_PERCENT


def load_csv(filepath):
    df = pd.read_csv(filepath)

    # extracting only emg columns: "emg smp0" and "emg val0", because next chanel was always equal to 0
    emg = df[["emg smp0", "emg val0"]].copy()
    emg.columns = ["sample", "emg value"] #changed names for better understanding

    # removing rows where the signal is equal to 0
    before = len(emg)
    emg = emg[emg["sample"].diff().fillna(1) > 0]
    emg = emg.reset_index(drop=True)
    after = len(emg)
    removed = before - after

    return emg, removed




# bandpass + notch filter
# bandpass 20–450 Hz keeps only the real EMG signal
# notch at 50 Hz removes electrical noise from the power grid

# FS = 1000 from config file, I NEED TO CHECK THIS VALUE, BECAUSE IT IS NOT MENTIONED IN THE DATASET


def bandpass_filter(signal, lowcut=20, highcut=450, fs=FS, order=4):
    b, a = butter(order, [lowcut, highcut], btype="band", fs=fs)
    return filtfilt(b, a, signal) # no shift in filtfilt


def notch_filter(signal, freq=50, fs=FS, quality=30):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal)


def apply_filters(emg):
    raw = emg["emg value"].values
    filtered = bandpass_filter(raw)
    filtered = notch_filter(filtered)
    emg["emg filtered"] = filtered

    # rectification
    emg["emg rectified"] = np.abs(filtered)
    return emg



# onset detection- rms-based

def onset_detection(emg):

    window = int(WINDOW_MS * FS / 1000)# number of samples in the window

    rms = (emg["emg rectified"].pow(2) .rolling(window=window, center=True).mean().apply(np.sqrt))
    emg["emg rms"] = rms

    #threshold = np.percentile(rms.dropna(), 50) * K # threshold based on the median of rms of entire signal
    threshold = rms.dropna().mean() * K
    emg["threshold"] = threshold
    emg["active"] = (rms > threshold).astype(int)
    return emg


def get_contractions(emg, min_duration=150, trim_percent=TRIM_PERCENT):
    # trim 10% from the start and end of each contraction
    contractions = []
    active = emg["active"].values
    in_contraction = False
    start = 0

    for i in range(len(active)):
        if active[i] == 1 and not in_contraction:
            start = i
            in_contraction = True
        elif active[i] == 0 and in_contraction:
            in_contraction = False
            duration = i - start
            if duration >= min_duration:
                trim = int(duration * trim_percent)
                contractions.append(emg.iloc[start + trim: i - trim].copy())

    # marking active regions but already trimmed at the beginning and at the end
    emg["active trimmed"] = 0
    for contraction in contractions:
        emg.loc[contraction.index, "active trimmed"] = 1

    return emg, contractions


def time_domain_features(contraction_df):
    signal = contraction_df["emg filtered"].values
    length = len(signal)

    rms = np.sqrt(np.mean(signal ** 2))
    mav = np.mean(np.abs(signal))
    zcr = np.sum(np.diff(np.sign(signal)) != 0) / length
    wl = np.sum(np.abs(np.diff(signal)))

    return {
        "RMS": rms,
        "MAV": mav,
        "ZCR": zcr,
        "WL": wl,
        "duration samples": length
    }



def process_file(filepath):
    emg, removed = load_csv(filepath)

    emg = apply_filters(emg)
    emg = onset_detection(emg)
    emg, contractions = get_contractions(emg)

    features_list = []
    for i, contraction in enumerate(contractions):
        features = time_domain_features(contraction)
        features["contraction id"] = i + 1
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    return emg, contractions, features_df, removed