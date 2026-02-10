#Sequential Elliptic Envelope Outlier Detection with LSTM Autoencoder
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- Config ----------------
FEATURE_CSV = "extracted_features.csv"   
OUTPUT_CSV = "extracted_features_with_lstm_outliers.csv"
feature_cols = [
    'activity',
    'mobility',
    'complexity',
    'pulse_width',
    'amplitude_variation',
    'mean_instantaneous_frequency',
    'std_instantaneous_frequency'
]

# LSTM & training params
SEQ_LEN = 10                # length of sequence (tunable)
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10                # early stopping
RANDOM_STATE = 42

# Elliptic Envelope params
CONTAMINATION = 0.05        # expected fraction of outliers

SEQUENTIAL_EE_WINDOW = 0    # set to >0 to perform sequential EE; 0 = fit once on training errors

# Training data selection
# If you have labeled "normal" portion, supply indices. Otherwise we assume the earliest fraction is normal.
ASSUME_NORMAL_FRACTION = 0.5  # fraction of earliest sequences used as "normal" training (tunable)



def create_sequences(X, seq_len):
    
    n = X.shape[0]
    seqs = []
    last_indices = []
    for start in range(0, n - seq_len + 1):
        seq = X[start:start + seq_len]
        seqs.append(seq)
        last_indices.append(start + seq_len - 1)  # map sequence -> last window index
    return np.array(seqs), np.array(last_indices)

def build_lstm_autoencoder(n_features, seq_len, latent_dim=64):
    
    inputs = Input(shape=(seq_len, n_features))   
    x = LSTM(latent_dim, activation='tanh')(inputs)    
    x = RepeatVector(seq_len)(x)    
    x = LSTM(latent_dim, activation='tanh', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(n_features))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

df = pd.read_csv(FEATURE_CSV)
if not set(feature_cols).issubset(df.columns):
    raise ValueError("Feature CSV does not contain the required feature columns.")

X_raw = df[feature_cols].values  # shape (n_windows, n_features)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_seq, seq_last_indices = create_sequences(X_scaled, SEQ_LEN)
n_seqs, seq_len, n_features = X_seq.shape
print(f"Built {n_seqs} sequences of length {seq_len} with {n_features} features.")


n_train_seqs = max( int(n_seqs * ASSUME_NORMAL_FRACTION), 1 )
train_idx_end = n_train_seqs
X_train_seq = X_seq[:train_idx_end]

print(f"Using first {n_train_seqs} sequences as 'normal' training data for the LSTM.")

tf.random.set_seed(RANDOM_STATE)
ae = build_lstm_autoencoder(n_features=n_features, seq_len=seq_len, latent_dim=LSTM_UNITS)
es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)


X_tr, X_val = train_test_split(X_train_seq, test_size=0.2, random_state=RANDOM_STATE)

history = ae.fit(
    X_tr, X_tr,
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=1
)

X_pred = ae.predict(X_seq, batch_size=BATCH_SIZE)

mse_per_feature = np.mean((X_seq - X_pred)**2, axis=1)  
mse_scalar = np.mean(mse_per_feature, axis=1)

error_vectors = mse_per_feature  

if SEQUENTIAL_EE_WINDOW and SEQUENTIAL_EE_WINDOW > 0:
    
    outlier_labels = np.ones(n_seqs)  # default: inlier=1
    outlier_scores = np.zeros(n_seqs)
    for start in range(0, n_seqs, SEQUENTIAL_EE_WINDOW):
        end = min(start + SEQUENTIAL_EE_WINDOW, n_seqs)
        
        if end - start < max(10, n_features + 1):
            continue
        ee = EllipticEnvelope(contamination=CONTAMINATION, random_state=RANDOM_STATE)
        ee.fit(error_vectors[start:end])
        labels = ee.predict(error_vectors[start:end])  # 1 inlier, -1 outlier
        scores = ee.decision_function(error_vectors[start:end])
        outlier_labels[start:end] = labels
        outlier_scores[start:end] = scores
else:
    
    ee = EllipticEnvelope(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    ee.fit(error_vectors[:train_idx_end])
    outlier_labels = ee.predict(error_vectors)   # 1 inlier, -1 outlier (shape n_seqs,)
    outlier_scores = ee.decision_function(error_vectors)


seq_df = pd.DataFrame({
    'sequence_index': np.arange(n_seqs),
    'last_window_index': seq_last_indices,
    'mse_scalar': mse_scalar,
    'outlier_label': outlier_labels,
    'outlier_score': outlier_scores
})

for i, fname in enumerate(feature_cols):
    seq_df[f'mse_{fname}'] = mse_per_feature[:, i]


seq_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved sequence-level results with outliers to {OUTPUT_CSV}")


n_windows = X_raw.shape[0]
window_df = pd.DataFrame({
    'window_index': np.arange(n_windows),
})

window_df['mse_scalar'] = np.nan
window_df['outlier_label'] = 1   # default inlier
window_df['outlier_score'] = np.nan


for _, row in seq_df.iterrows():
    widx = int(row['last_window_index'])
    window_df.at[widx, 'mse_scalar'] = row['mse_scalar']
    window_df.at[widx, 'outlier_score'] = row['outlier_score']
    window_df.at[widx, 'outlier_label'] = row['outlier_label']


window_output_csv = "extracted_features_window_level_outliers.csv"
window_df.to_csv(window_output_csv, index=False)
print(f"Saved window-level results to {window_output_csv}")



plt.figure(figsize=(12,5))
plt.plot(seq_df['sequence_index'], seq_df['mse_scalar'], label='Sequence MSE (reconstruction)')

outlier_seq_idx = seq_df[seq_df['outlier_label'] == -1]['sequence_index'].values
#print(f"Detected {(outlier_seq_idx)} outlier sequences.")
plt.scatter(outlier_seq_idx, seq_df.loc[outlier_seq_idx, 'mse_scalar'], color='red', label='Detected Outliers', zorder=5)
plt.xlabel('Sequence Index')
plt.ylabel('Reconstruction MSE')
plt.title('LSTM reconstruction error and Elliptic Envelope outliers')
plt.axhline(y=np.percentile(seq_df['mse_scalar'], 100 * (1 - CONTAMINATION)), color='orange', linestyle='--', label='EE Threshold')
print(f"Elliptic Envelope threshold for outliers: {np.percentile(seq_df['mse_scalar'], 100 * (1 - CONTAMINATION))}")
plt.grid()
plt.legend()
plt.savefig('lstm_elliptic_outlier_detection.png',dpi=300)
plt.show()

plt.figure(figsize=(8,4))
plt.hist(seq_df['mse_scalar'], bins=60)
plt.title('Histogram of reconstruction MSE')
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

