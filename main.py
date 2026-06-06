import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt

# latex ui
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": True,
    "axes.spines.right": True,
})

datasets = [
    "emg_datasets/EP_2807_MADC_p6_kl_EMG_trening1",
    "emg_datasets/EP_2807_MADC_p6_kl_EMG_przysiadyprzed",
    "emg_datasets/EP_2807_MADC_p6_kl_EMG_przysiadypo",
    "emg_datasets/EP_2807_MADC_p6_kl_EMG_kon",
    "emg_datasets/EP_2807_MADC_p6_kl_EMG_kon5kg",
    "emg_datasets/EP_2807_MADC_p6_kl_EMG_izo"
]

data = {}

for dataset in datasets:
    df = pd.read_csv(f"{dataset}.csv")
    data[dataset] = df
    print(f"{dataset}.csv loaded")

# extracting only emg columns: "emg smp0" and "emg val0", because next chanel was always equal to 0
emg_data = {}

for dataset, df in data.items():
    emg = df[["emg smp0", "emg val0"]].copy()
    emg.columns = ["sample", "emg value"] #changed names for better understanding
    emg_data[dataset] = emg

# quick check for one file
print(emg_data["emg_datasets/EP_2807_MADC_p6_kl_EMG_izo"].head(10))
print(emg_data["emg_datasets/EP_2807_MADC_p6_kl_EMG_izo"].describe())

# removing rows where the signal is equal to 0
for dataset in emg_data:
    before = len(emg_data[dataset])
    emg_data[dataset] = emg_data[dataset][emg_data[dataset]["sample"].diff().fillna(1) > 0]
    emg_data[dataset] = emg_data[dataset].reset_index(drop=True)
    after = len(emg_data[dataset])
    print(f"{dataset}: removed {before - after} rows (padding), {after} rows remaining")


# bandpass + notch filter
# bandpass 20–450 Hz keeps only the real EMG signal
# notch at 50 Hz removes electrical noise from the power grid

FS = 1000 # I NEED TO CHECK THIS VALUE, BECAUSE IT IS NOT MENTIONED IN THE DATASET

def bandpass_filter(signal, lowcut=20, highcut=450, fs=FS, order=4):
    b, a = butter(order, [lowcut, highcut], btype="band", fs=fs)
    return filtfilt(b, a, signal) # no shift in filtfilt

def notch_filter(signal, freq=50, fs=FS, quality=30):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal)


for dataset in emg_data:
    raw = emg_data[dataset]["emg value"].values
    filtered = bandpass_filter(raw)
    filtered = notch_filter(filtered)
    emg_data[dataset]["emg filtered"] = filtered

print(f"{dataset}: applied bandpass and notch filters")

#retrification
for dataset in emg_data:
    emg_data[dataset]["emg rectified"] = emg_data[dataset]["emg filtered"].abs()

print("applied rectification")

# quick check if the filtering works properly
dataset = "emg_datasets/EP_2807_MADC_p6_kl_EMG_izo"
df = emg_data[dataset]

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(df["sample"], df["emg value"], linewidth=0.5, color="blue")
axes[0].set_title("raw EMG signal")
axes[0].set_ylabel("amplitude [mV]")
axes[0].set_xlabel("sample number")

axes[1].plot(df["sample"], df["emg filtered"], linewidth=0.5, color="red")
axes[1].set_title("filtered EMG signal (bandpass 20-450 Hz + notch 50 Hz)")
axes[1].set_ylabel("amplitude [mV]")
axes[1].set_xlabel("sample number")

plt.tight_layout()
plt.show()



# oneset detection- rms-based

WINDOW_MS = 200

WINDOW = int(WINDOW_MS * FS / 1000) # number of samples in the window

K = 0.5
for dataset in emg_data:
    rms = (emg_data[dataset]["emg rectified"].pow(2) .rolling(window=WINDOW, center=True).mean().apply(np.sqrt))
    emg_data[dataset]["emg rms"] = rms

    threshold = np.percentile(rms.dropna(), 50) * K  # threshold based on the median of rms of entire signal
    emg_data[dataset]["threshold"] = threshold
    emg_data[dataset]["active"] = (rms > threshold).astype(int)


print("applied RMS-based onset detection")




TRIM_PERCENT = 0.1  # trim 10% from the start and end of each contraction
def get_contraction(df, min_duration=150, trim_percent=TRIM_PERCENT):
    contractions = []
    active = df["active"].values
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
                contractions.append(df.iloc[start + trim : i - trim].copy())

    return contractions




# time domain

def compute_time_domain_features(contraction_df):
    signal = contraction_df["emg filtered"].values
    length = len(signal)

    rms = np.sqrt(np.mean(signal**2))
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

all_features = {}

for dataset in emg_data:
    df = emg_data[dataset]
    contractions = get_contraction(df, min_duration=150)

    emg_data[dataset]["active trimmed"] = 0
    for contraction in contractions:
        emg_data[dataset].loc[contraction.index, "active trimmed"] = 1

    features_list = []
    for i, contraction in enumerate(contractions):
        features = compute_time_domain_features(contraction)
        features["contraction id"] = i +1
        features_list.append(features)

    all_features[dataset] = pd.DataFrame(features_list)
    print(f"{dataset}: extracted features from {len(contractions)} contractions")

# quick check
print(all_features["emg_datasets/EP_2807_MADC_p6_kl_EMG_izo"])


# quick check if oneset detection works properly
dataset = "emg_datasets/EP_2807_MADC_p6_kl_EMG_izo"
df = emg_data[dataset]

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# top: filtered signal + active regions highlighted
axes[0].plot(df["sample"], df["emg filtered"], linewidth=0.5, color="red", label="filtered EMG")
axes[0].fill_between(df["sample"], df["emg filtered"].min(), df["emg filtered"].max(), where=df["active trimmed"] == 1, alpha=0.2, color="red", label="detected active")
axes[0].set_ylabel("amplitude [mV]")
axes[0].set_xlabel("sample number")
axes[0].set_title("EMG signal with detected onsets")
axes[0].legend(fontsize=9)

# bottom: RMS envelope + threshold line
axes[1].plot(df["sample"], df["emg rms"], linewidth=1, color="blue", label="RMS envelope")
axes[1].axhline(y=df["threshold"].iloc[0], color="red", linewidth=1, linestyle="--", label=f"threshold")
axes[1].set_ylabel("amplitude [mV]")
axes[1].set_xlabel("sample number")
axes[1].set_title("RMS envelope and threshold")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()


# features plot

dataset = "emg_datasets/EP_2807_MADC_p6_kl_EMG_izo"
df_signal = emg_data[dataset]
df_features = all_features[dataset]

fig, axes = plt.subplots(5, 1, figsize=(12, 14))

# top emg signal with onset detection
axes[0].plot(df_signal["sample"], df_signal["emg filtered"], linewidth=0.5, color="red", label="filtered EMG")
axes[0].fill_between(df["sample"], df["emg filtered"].min(), df["emg filtered"].max(),
                     where=df["active trimmed"] == 1, alpha=0.2, color="red", label="detected active")
axes[0].set_ylabel("amplitude [mV]")
axes[0].set_xlabel("sample number")
axes[0].set_title("EMG signal with detected onsets")


# features
features_to_plot = ["RMS", "MAV", "ZCR", "WL"]
colors = ["darkorange", "green", "purple", "royalblue"]
ylabels = ["amplitude [mV]", "amplitude [mV]", "rate [crossings/sample]", "amplitude [mV]"]

for i, (feature, color, ylabel) in enumerate(zip(features_to_plot, colors, ylabels)):
    ax = axes[i + 1]
    ax.plot(df_features["contraction id"], df_features[feature], marker="o", linewidth=1.5, markersize=5, color=color)
    ax.set_title(f"{feature} per contraction")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("contraction id")
    ax.set_xticks(df_features["contraction id"])
    ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Characteristic parameters in time domain", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 1])

plt.tight_layout()
plt.show()




