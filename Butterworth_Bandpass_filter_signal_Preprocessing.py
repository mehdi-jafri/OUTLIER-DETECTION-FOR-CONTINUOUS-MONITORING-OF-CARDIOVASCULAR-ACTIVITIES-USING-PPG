import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data)
    return y

def min_max_normalize(data, new_min=0.0, new_max=1.0):
    d_min, d_max = np.min(data), np.max(data)
    if d_max - d_min == 0:
        return np.full_like(data, new_min)
    normalized = (data - d_min) / (d_max - d_min)
    return normalized * (new_max - new_min) + new_min


segment_name = "81739927_0002"
rec_path = f'downloaded/p100/p10014354/81739927/{segment_name}'


signals, fields = wfdb.rdsamp(rec_path)
fs = fields['fs']

total_samples = len(signals)

total_time_sec = total_samples / fs

print(f"Sampling rate: {fs} Hz")
print(f"Total samples: {total_samples}")
print(f"Total duration: {total_time_sec:.2f} seconds")


print("Available channels:", fields['sig_name'])


possible_names = ["PPG", "PLETH", "PULSE", "SPO2", "OXY"]
ppg_channel_idx = None
for i, sig_name in enumerate(fields['sig_name']):
    if any(name in sig_name.upper() for name in possible_names):
        ppg_channel_idx = i
        break

if ppg_channel_idx is None:
    raise ValueError(f"No PPG-like channel found. Available channels: {fields['sig_name']}")

print(f"Selected PPG channel: {fields['sig_name'][ppg_channel_idx]}")


ppg_signal = np.nan_to_num(signals[:, ppg_channel_idx])

# Step 1: Bandpass filter for PPG (0.5–8 Hz typical)
filtered_ppg = butter_bandpass_filter(ppg_signal, lowcut=0.5, highcut=8.0, fs=fs, order=4)

# Step 2: Min-Max normalize to [0, 1]
normalized_ppg = min_max_normalize(filtered_ppg, new_min=0.0, new_max=1.0)

output_folder = "preprocessed"
os.makedirs(output_folder, exist_ok=True)  
output_filename = f"{segment_name}_processed_ppg_signal.npy"
output_path = os.path.join(output_folder, output_filename)

np.save(output_path, normalized_ppg)
print(f"Processed PPG saved to: {output_path}")

t = np.arange(len(ppg_signal)) / fs

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, ppg_signal)
plt.title("Original PPG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude") 

plt.subplot(3, 1, 2)
plt.plot(t, filtered_ppg,color='tab:orange')
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
print(f"Plots saved to: {output_folder}")   
plt.show()


