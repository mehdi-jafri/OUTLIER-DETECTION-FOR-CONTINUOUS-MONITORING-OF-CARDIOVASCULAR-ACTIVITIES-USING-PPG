import wfdb				#Preprocessing_all_signals
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os
import glob


# === Functions ===
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def min_max_normalize(data, new_min=0.0, new_max=1.0):
    d_min, d_max = np.min(data), np.max(data)
    if d_max - d_min == 0:
        return np.full_like(data, new_min)
    normalized = (data - d_min) / (d_max - d_min)
    return normalized * (new_max - new_min) + new_min

# === Paths ===
base_dir = "downloaded/p100/p10014354/81739927"
output_folder = "preprocessed"
os.makedirs(output_folder, exist_ok=True)

# === Process all WFDB records ===
record_paths = glob.glob(os.path.join(base_dir, "*.hea"))  # WFDB header files

for record_path in record_paths:
    segment_name = os.path.splitext(os.path.basename(record_path))[0]
    rec_path = os.path.join(base_dir, segment_name)

    print(f"\nProcessing: {segment_name}")

    try:
        signals, fields = wfdb.rdsamp(rec_path)
    except Exception as e:
        print(f"⚠️ Could not read {segment_name}: {e}")
        continue

    fs = fields['fs']
    total_samples = len(signals)
    total_time_sec = total_samples / fs

    print(f"Sampling rate: {fs} Hz, Total samples: {total_samples}, Duration: {total_time_sec:.2f} s")
    print("Available channels:", fields['sig_name'])

    # Find PPG channel
    possible_names = ["PPG", "PLETH", "PULSE", "SPO2", "OXY"]
    ppg_channel_idx = None
    for i, sig_name in enumerate(fields['sig_name']):
        if any(name in sig_name.upper() for name in possible_names):
            ppg_channel_idx = i
            break

    if ppg_channel_idx is None:
        print(f"❌ No PPG-like channel found in {segment_name}, skipping.")
        continue

    print(f"Selected PPG channel: {fields['sig_name'][ppg_channel_idx]}")

    # Preprocess
    ppg_signal = np.nan_to_num(signals[:, ppg_channel_idx])
    filtered_ppg = butter_bandpass_filter(ppg_signal, lowcut=0.5, highcut=8.0, fs=fs, order=4)
    normalized_ppg = min_max_normalize(filtered_ppg)

    # Save processed signal
    np.save(os.path.join(output_folder, f"{segment_name}_processed_ppg_signal.npy"), normalized_ppg)

    # Plot
    t = np.arange(len(ppg_signal)) / fs
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, ppg_signal)
    plt.title("Original PPG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(t, filtered_ppg, color='tab:orange')
    plt.title("Butterworth Bandpass Filtered PPG (0.5–8 Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 3)
    plt.plot(t, normalized_ppg, color='tab:green')
    plt.title("Normalized PPG [0, 1]")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{segment_name}_ppg_plots.png"))
    plt.close()

    print(f"✅ Processed & saved {segment_name}")

print("\n🎯 All files processed.")
