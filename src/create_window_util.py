import numpy as np

def create_windows(X, y, window_size, step_size):
    X_win, y_win = [], []
    for i in range(0, len(X) - window_size, step_size):
        window = X.iloc[i:i + window_size].values
        label = y.iloc[i + window_size]
        X_win.append(window)
        y_win.append(label)
    return np.array(X_win), np.array(y_win)