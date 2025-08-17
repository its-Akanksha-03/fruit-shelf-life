import numpy as np

def make_sequences(X, y, seq_len=5):
    X_seq, y_seq = [], []
    for i in range(0, len(X) - seq_len, seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)
