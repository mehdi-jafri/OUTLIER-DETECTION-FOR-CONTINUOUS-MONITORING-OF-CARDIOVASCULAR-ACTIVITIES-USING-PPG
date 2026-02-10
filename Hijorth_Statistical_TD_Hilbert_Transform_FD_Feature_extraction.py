import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')

# === Feature functions ===

def hijorth_parameters(signal):
    activity = np.var(signal)
    diff_signal = np.diff(signal)
    mobility = np.sqrt(np.var(diff_signal) / activity)
    diff2_signal = np.diff(diff_signal)
    mobility_diff = np.sqrt(np.var(diff2_signal) / np.var(diff_signal))
    complexity = mobility_diff / mobility
    return activity, mobility, complexity

def adaptive_threshold(features, factor=1.5):
    activity_vals = np.array([f[0] for f in features])
    threshold = np.mean(activity_vals) + factor * np.std(activity_vals)
    return threshold

def detect_peaks(features, threshold):
    return [i for i, f in enumerate(features) if f[0] >= threshold]

def pulse_features(window):
    max_amp = np.max(window)
    threshold = 0.5 * max_amp
    above_thresh = np.where(window >= threshold)[0]
    width = (above_thresh[-1] - above_thresh[0] + 1) if len(above_thresh) > 0 else 0
    amp_var = np.std(window) / (np.mean(window) + 1e-6)
    return width, amp_var

def hilbert_features(window, fs):
    analytic_signal = hilbert(window)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2*np.pi)) * fs
    mean_freq = np.mean(instantaneous_frequency)
    std_freq = np.std(instantaneous_frequency)
    return amplitude_envelope, instantaneous_frequency, mean_freq, std_freq

def process_channel(signal, fs, window_sec, step_sec, threshold_factor=1.5):
    window_size = int(window_sec * fs)
    step_size = int(step_sec * fs)

    features = []
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start+window_size]
        act, mob, comp = hijorth_parameters(window)
        features.append([act, mob, comp])
        windows.append(window)

    th = adaptive_threshold(features, threshold_factor)
    peak_indices = detect_peaks(features, th)

    time_domain_results = []
    freq_domain_results = []

    for idx in peak_indices:
        window = windows[idx]
        pulse_width, amp_var = pulse_features(window)
        time_domain_results.append({
            'index': idx,
            'activity': features[idx][0],
            'mobility': features[idx][1],
            'complexity': features[idx][2],
            'pulse_width': pulse_width,
            'amplitude_variation': amp_var
        })
        amp_env, inst_freq, mean_freq, std_freq = hilbert_features(window, fs)
        freq_domain_results.append({
            'index': idx,
            'mean_instantaneous_frequency': mean_freq,
            'std_instantaneous_frequency': std_freq,
            'instantaneous_amplitude_envelope': amp_env.tolist(),
            'instantaneous_frequency': inst_freq.tolist()
        })

    return time_domain_results, freq_domain_results

# === Main processing ===

folder_path = "preprocessed"
sampling_rate = 62.47  # Hz
window_sec = 5
step_sec = 1

file_list = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
if not file_list:
    raise FileNotFoundError(f"No .npy files found in {folder_path}")

all_results = []

for file_path in file_list:
    file_name = os.path.basename(file_path)
    signal = np.load(file_path)

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    n_channels = signal.shape[1]
    print(f"Processing {file_name} with shape {signal.shape}")

    for ch in range(n_channels):
        ch_signal = signal[:, ch]
        time_feats, freq_feats = process_channel(ch_signal, sampling_rate, window_sec, step_sec)

        all_results.append({
            'file': file_name,
            'channel': ch,
            'time_domain_features': time_feats,
            'frequency_domain_features': freq_feats
        })

# === Save results to CSV ===

csv_output_path = "extracted_features.csv"
header = [
    'file', 'channel', 'window_index',
    'activity', 'mobility', 'complexity',
    'pulse_width', 'amplitude_variation',
    'mean_instantaneous_frequency', 'std_instantaneous_frequency'
]

with open(csv_output_path, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()

    for res in all_results:
        time_feats = res['time_domain_features']
        freq_feats = res['frequency_domain_features']

        for t_feat, f_feat in zip(time_feats, freq_feats):
            row = {
                'file': res['file'],
                'channel': res['channel'],
                'window_index': t_feat['index'],
                'activity': t_feat['activity'],
                'mobility': t_feat['mobility'],
                'complexity': t_feat['complexity'],
                'pulse_width': t_feat['pulse_width'],
                'amplitude_variation': t_feat['amplitude_variation'],
                'mean_instantaneous_frequency': f_feat['mean_instantaneous_frequency'],
                'std_instantaneous_frequency': f_feat['std_instantaneous_frequency']
            }
            writer.writerow(row)

print(f"Features saved to {csv_output_path}")

# === Visualization ===

for result in all_results:
    if result['time_domain_features'] and result['frequency_domain_features']:
        first_result = result
        break

time_feats = first_result['time_domain_features']
freq_feats = first_result['frequency_domain_features']

activities = [f['activity'] for f in time_feats]
mean_freqs = [f['mean_instantaneous_frequency'] for f in freq_feats]
indices = [f['index'] for f in time_feats]  # window indices of detected peaks

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(indices, activities, marker='o', label='Activity')
plt.scatter(indices, activities, color='red', zorder=5, label='Detected Peaks')
for peak in indices:
    plt.axvline(x=peak, color='red', linestyle='--', alpha=0.3)
plt.title(f"Activity (Time Domain) with Peaks\nFile: {first_result['file']} Channel: {first_result['channel']}")
plt.xlabel('Window Index')
plt.ylabel('Activity')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(indices, mean_freqs, marker='x', color='orange', label='Mean Inst. Frequency')
plt.scatter(indices, mean_freqs, color='red', zorder=5, label='Detected Peaks')
for peak in indices:
    plt.axvline(x=peak, color='red', linestyle='--', alpha=0.3)
plt.title("Mean Instantaneous Frequency with Peaks")
plt.xlabel('Window Index')
plt.ylabel('Frequency (Hz)')
plt.legend()

plt.tight_layout()
plt.show()

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, find_peaks

# === Signal Filtering ===

def bandpass_filter(signal, fs, lowcut=0.5, highcut=5.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

# === Hjorth Parameters ===

def hijorth_parameters(signal):
    activity = np.var(signal)
    diff_signal = np.diff(signal)
    mobility = np.sqrt(np.var(diff_signal) / activity)
    diff2_signal = np.diff(diff_signal)
    mobility_diff = np.sqrt(np.var(diff2_signal) / np.var(diff_signal))
    complexity = mobility_diff / mobility
    return activity, mobility, complexity

# === Pulse Feature Extraction ===

def adaptive_threshold(features, factor=1.5):
    activity_vals = np.array([f[0] for f in features])
    return np.mean(activity_vals) + factor * np.std(activity_vals)

def detect_peaks(features, threshold):
    return [i for i, f in enumerate(features) if f[0] >= threshold]

def pulse_features(window):
    max_amp = np.max(window)
    threshold = 0.5 * max_amp
    above_thresh = np.where(window >= threshold)[0]
    width = (above_thresh[-1] - above_thresh[0] + 1) if len(above_thresh) > 0 else 0
    amp_var = np.std(window) / (np.mean(window) + 1e-6)
    return width, amp_var

def hilbert_features(window, fs):
    analytic_signal = hilbert(window)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2 * np.pi)) * fs

    if len(instantaneous_frequency) >= 5:
        instantaneous_frequency = np.convolve(instantaneous_frequency, np.ones(5)/5, mode='same')
    instantaneous_frequency = np.clip(instantaneous_frequency, 0, None)

    mean_freq = np.mean(instantaneous_frequency)
    std_freq = np.std(instantaneous_frequency)
    return amplitude_envelope, instantaneous_frequency, mean_freq, std_freq

# === Heart Rate Estimation ===

def detect_inverted_peaks(signal, fs, min_distance_sec=0.4):
    inverted = -signal
    distance_samples = int(min_distance_sec * fs)
    peaks, _ = find_peaks(inverted, distance=distance_samples)
    return peaks

def score_peaks(peaks, fs):
    if len(peaks) < 2:
        return 0, 0
    intervals = np.diff(peaks) / fs
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    bpm = 60 / mean_interval
    if bpm < 40 or bpm > 180:
        return 0, bpm
    score = 1 / (1 + std_interval)
    return score, bpm

def estimate_hr_from_peaks(signal, fs, window_sec=5, step_sec=1):
    window_size = int(window_sec * fs)
    step_size = int(step_sec * fs)
    bpm_results = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        peaks = detect_inverted_peaks(window, fs)
        score, bpm = score_peaks(peaks, fs)
        bpm_results.append({
            'start_sample': start,
            'window_index': start // step_size,
            'bpm': bpm,
            'score': score,
            'peak_count': len(peaks)
        })

    return bpm_results

# === Full Channel Processing ===

def process_channel(signal, fs, window_sec, step_sec, threshold_factor=1.5):
    window_size = int(window_sec * fs)
    step_size = int(step_sec * fs)

    features = []
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start+window_size]
        act, mob, comp = hijorth_parameters(window)
        features.append([act, mob, comp])
        windows.append(window)

    th = adaptive_threshold(features, threshold_factor)
    peak_indices = detect_peaks(features, th)

    time_results = []
    freq_results = []

    for idx in peak_indices:
        window = windows[idx]
        pulse_width, amp_var = pulse_features(window)
        time_results.append({
            'index': idx,
            'activity': features[idx][0],
            'mobility': features[idx][1],
            'complexity': features[idx][2],
            'pulse_width': pulse_width,
            'amplitude_variation': amp_var
        })
        amp_env, inst_freq, mean_freq, std_freq = hilbert_features(window, fs)
        freq_results.append({
            'index': idx,
            'mean_instantaneous_frequency': mean_freq,
            'std_instantaneous_frequency': std_freq
        })

    return time_results, freq_results

# === Main Execution ===

folder_path = "preprocessed"
sampling_rate = 62.47
window_sec = 5
step_sec = 1

file_list = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
if not file_list:
    raise FileNotFoundError("No .npy files found in preprocessed/")

all_results = []

for file_path in file_list:
    file_name = os.path.basename(file_path)
    signal = np.load(file_path)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    for ch in range(signal.shape[1]):
        ch_signal = signal[:, ch]
        ch_signal = bandpass_filter(ch_signal, sampling_rate)

        time_feats, freq_feats = process_channel(ch_signal, sampling_rate, window_sec, step_sec)
        hr_results = estimate_hr_from_peaks(ch_signal, sampling_rate, window_sec, step_sec)

        all_results.append({
            'file': file_name,
            'channel': ch,
            'filtered_signal': ch_signal,
            'time_domain_features': time_feats,
            'frequency_domain_features': freq_feats,
            'hr_results': hr_results
        })

# === Save Features ===
'''
with open("estimated_hr.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "channel", "window_index", "bpm", "score", "peak_count"])
    writer.writeheader()
    for res in all_results:
        for hr in res['hr_results']:
            writer.writerow({
                "file": res['file'],
                "channel": res['channel'],
                "window_index": hr['window_index'],
                "bpm": hr['bpm'],
                "score": hr['score'],
                "peak_count": hr['peak_count']
            })
'''
# === Visualizations ===

for res in all_results:
    if res['time_domain_features'] and res['hr_results']:
        first_result = res
        break

# === Visualizations ===

import os

plot_folder = "plots"
os.makedirs(plot_folder, exist_ok=True)

for result in all_results:
    time_feats = result['time_domain_features']
    freq_feats = result['frequency_domain_features']

    if not time_feats or not freq_feats:
        continue

    file_name = result['file']
    channel = result['channel']

    activities = [f['activity'] for f in time_feats]
    mean_freqs = [f['mean_instantaneous_frequency'] for f in freq_feats]
    indices = [f['index'] for f in time_feats]

    # Estimate BPM based on intervals between detected peaks
    step_sec = 1  # your step size in seconds

    if len(indices) > 1:
        intervals_sec = np.diff(indices) * step_sec
        avg_interval = np.mean(intervals_sec)
        bpm = 60 / avg_interval if avg_interval > 0 else 0
    else:
        bpm = 0  # Not enough peaks to estimate BPM

    plt.figure(figsize=(8, 12))  # Taller figure for vertical layout

    plt.subplot(3, 1, 1)
    plt.plot(indices, activities, marker='o', label='Activity')
    plt.scatter(indices, activities, color='red', zorder=5, label='Detected Peaks')
    for peak in indices:
        plt.axvline(x=peak, color='red', linestyle='--', alpha=0.3)
    plt.title(f"Hjorth Activity with Peaks\nFile: {file_name} Channel: {channel}")
    plt.xlabel('Window Index')
    plt.ylabel('Activity')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(indices, mean_freqs, marker='x', color='orange', label='Mean Inst. Frequency')
    plt.scatter(indices, mean_freqs, color='red', zorder=5, label='Detected Peaks')
    for peak in indices:
        plt.axvline(x=peak, color='red', linestyle='--', alpha=0.3)
    plt.title("Mean Instantaneous Frequency with Peaks")
    plt.xlabel('Window Index')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    

    plt.subplot(3, 1, 3)
    if len(indices) > 1:
        plt.plot(indices[1:], 60 / (np.diff(indices) * step_sec), marker='d', color='green', label='Instantaneous BPM')
    plt.axhline(y=bpm, color='blue', linestyle='--', label=f'Avg BPM: {bpm:.2f}')
    plt.title("Heart Rate Estimation")
    plt.xlabel('Window Index')
    plt.ylabel('BPM')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()

    safe_filename = file_name.replace(".npy", "")
    save_path = os.path.join(plot_folder, f"{safe_filename}_channel_{channel}.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()

print(f"All enhanced plots saved to folder: {plot_folder}")
