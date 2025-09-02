import wfdb
import numpy as np
import matplotlib.pyplot as plt

segment_name = "81739927_0005"

rec_path = f'downloaded/P100/p10014354/81739927/{segment_name}'


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


t = np.arange(len(ppg_signal)) / fs

plt.figure(figsize=(12, 6))
plt.plot(t, ppg_signal)
plt.title("Original PPG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude") 
plt.show()
