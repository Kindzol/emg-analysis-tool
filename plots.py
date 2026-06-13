import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 125,
    "axes.spines.top": True,
    "axes.spines.right": True,
})


def plot_raw_vs_filtered(emg):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(emg["sample"], emg["emg value"], linewidth=0.5, color="blue")
    axes[0].set_title("raw EMG signal")
    axes[0].set_ylabel("amplitude [mV]")
    axes[0].set_xlabel("sample number")
    axes[0].tick_params(labelbottom=True)

    axes[1].plot(emg["sample"], emg["emg filtered"], linewidth=0.5, color="red")
    axes[1].set_title("filtered EMG signal (bandpass 20-450 Hz + notch 50 Hz)")
    axes[1].set_ylabel("amplitude [mV]")
    axes[1].set_xlabel("sample number")

    plt.tight_layout()
    return fig


def plot_onset_detection(emg):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(emg["sample"], emg["emg filtered"], linewidth=0.5, color="red", label="filtered EMG")
    axes[0].fill_between(emg["sample"], emg["emg filtered"].min(), emg["emg filtered"].max(),
                         where=emg["active trimmed"] == 1,alpha=0.2, color="red", label="detected active")
    axes[0].set_title("EMG signal with detected onsets")
    axes[0].set_ylabel("amplitude [mV]")
    axes[0].set_xlabel("sample number")
    axes[0].legend(fontsize=9)
    axes[0].tick_params(labelbottom=True)

    axes[1].plot(emg["sample"], emg["emg rms"], linewidth=1, color="blue", label="RMS envelope")
    axes[1].axhline(y=emg["threshold"].iloc[0], color="red", linewidth=1, linestyle="--", label="threshold")
    axes[1].set_title("RMS envelope and threshold")
    axes[1].set_ylabel("amplitude [mV]")
    axes[1].set_xlabel("sample number")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_features(emg, features_df):
    fig, axes = plt.subplots(5, 1, figsize=(10, 11))

    axes[0].plot(emg["sample"], emg["emg filtered"], linewidth=0.5, color="red", label="filtered EMG")
    axes[0].fill_between(emg["sample"],emg["emg filtered"].min(), emg["emg filtered"].max(),
                         where=emg["active trimmed"] == 1,alpha=0.2, color="red", label="detected active")
    axes[0].set_title("EMG signal with detected onsets")
    axes[0].set_ylabel("amplitude [mV]")
    axes[0].set_xlabel("sample number")
    axes[0].legend(fontsize=9)

    features_to_plot = ["RMS", "MAV", "ZCR", "WL"]
    colors = ["darkorange", "green", "purple", "royalblue"]
    ylabels = ["amplitude [mV]", "amplitude [mV]", "rate [crossings/sample]", "amplitude [mV]"]

    for i, (feature, color, ylabel) in enumerate(zip(features_to_plot, colors, ylabels)):
        ax = axes[i + 1]
        ax.plot(features_df["contraction id"], features_df[feature], marker="o", linewidth=1.5, markersize=5, color=color)
        ax.set_title(f"{feature} per contraction")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("contraction id")
        ax.set_xticks(features_df["contraction id"])
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Characteristic parameters in time domain", fontsize=14)
    plt.tight_layout()
    return fig