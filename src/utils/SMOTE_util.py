import os
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from collections import Counter
from src.utils.create_window_util import create_windows

FILE_PATH = "../processed_data/SDHAR/final_processed_data_ALL_DAYS.csv"
TARGET_COLUMN = 'activity_user_1'
WINDOW_SIZE = 60
STEP_SIZE = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVE_PATH = "../processed_data/SDHAR/"
LSTM_DATA_FILE = os.path.join(SAVE_PATH, "lstm_smote_data.npz")
TREE_DATA_FILE = os.path.join(SAVE_PATH, "tree_smote_data.npz")

def _load_and_split_data():
    print("Loading final processed data...")
    df = pd.read_csv(FILE_PATH)

    print("Separating Features and Target...")
    df.dropna(inplace=True)
    X = df.drop(columns=[col for col in df.columns if 'activity' in col])
    y = df[TARGET_COLUMN].astype(int)  # Use 1D integer labels

    print(f"Creating sliding windows (size={WINDOW_SIZE}, step={STEP_SIZE})...")
    # X_win is 3D (samples, timesteps, features)
    # y_win is 1D (samples,) with integer labels
    X_win, y_win = create_windows(X, y, WINDOW_SIZE, STEP_SIZE)
    print(f"  - Windowed X shape: {X_win.shape}")
    print(f"  - Windowed y shape: {y_win.shape}")

    print("Splitting data into training and test sets...")
    X_train_3d, X_test_3d, y_train_1d, y_test_1d = train_test_split(
        X_win, y_win,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_win
    )

    print(f"  - Original training distribution: {Counter(y_train_1d)}")
    return X_train_3d, X_test_3d, y_train_1d, y_test_1d


def _apply_smote(X_train_flat, y_train_1d):
    print("\nApplying SMOTE (Over-sampling) and RandomUnderSampler (Under-sampling)...")

    counts = Counter(y_train_1d)

    # Define our sample targets
    MIN_SAMPLES = 2000  # Floor: Bring all classes *up* to this
    MAX_SAMPLES = 5000  # Ceiling: Bring all classes *down* to this

    # 1. Define Over-sampling (SMOTE) strategy
    #    Bring all classes with < MIN_SAMPLES up to MIN_SAMPLES
    over_strategy = {c: MIN_SAMPLES for c, n in counts.items() if n < MIN_SAMPLES}
    # k_neighbors=3 is required for your tiniest classes (like "LAUNDRY")
    over = SMOTE(sampling_strategy=over_strategy, k_neighbors=3, random_state=RANDOM_STATE)

    # 2. Define Under-sampling (RandomUnderSampler) strategy
    #    Bring all classes with > MAX_SAMPLES down to MAX_SAMPLES
    under_strategy = {c: MAX_SAMPLES for c, n in counts.items() if n > MAX_SAMPLES}
    under = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_STATE)

    # 3. Create the pipeline
    #    This will apply over-sampling first, then under-sampling.
    pipeline = Pipeline(steps=[('o', over), ('u', under)])

    print("  - Fitting pipeline (this may take a moment)...")
    X_train_flat_res, y_train_res = pipeline.fit_resample(X_train_flat, y_train_1d)

    print(f"  - Resampled training distribution: {Counter(y_train_res)}")
    return X_train_flat_res, y_train_res


def process_and_save_lstm_data():
    print("--- Starting Preprocessing for LSTM ---")
    X_train_3d, X_test_3d, y_train_1d, y_test_1d = _load_and_split_data()
    n_samples, n_timesteps, n_features = X_train_3d.shape
    num_classes = len(np.unique(y_train_1d))

    X_train_flat = X_train_3d.reshape((n_samples, n_timesteps * n_features))

    X_train_flat_res, y_train_res = _apply_smote(X_train_flat, y_train_1d)

    X_train_3d_res = X_train_flat_res.reshape((-1, n_timesteps, n_features))

    # 5. One-hot encode labels for Keras
    y_train_cat_res = to_categorical(y_train_res, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_1d, num_classes=num_classes)

    print(f"\nSaving processed LSTM data to: {LSTM_DATA_FILE}")
    np.savez_compressed(
        LSTM_DATA_FILE,
        X_train=X_train_3d_res,
        y_train=y_train_cat_res,
        X_test=X_test_3d,
        y_test=y_test_cat,
        y_test_1d_labels=y_test_1d
    )
    print("--- LSTM Preprocessing Complete ---")


def process_and_save_tree_data():
    print("\n--- Starting Preprocessing for Tree Models (RF/DT) ---")

    X_train_3d, X_test_3d, y_train_1d, y_test_1d = _load_and_split_data()
    n_samples_train, n_timesteps, n_features = X_train_3d.shape
    n_samples_test = X_test_3d.shape[0]

    X_train_flat = X_train_3d.reshape((n_samples_train, n_timesteps * n_features))
    X_test_flat = X_test_3d.reshape((n_samples_test, n_timesteps * n_features))

    X_train_flat_res, y_train_res = _apply_smote(X_train_flat, y_train_1d)

    print(f"\nSaving processed Tree data to: {TREE_DATA_FILE}")
    np.savez_compressed(
        TREE_DATA_FILE,
        X_train=X_train_flat_res,
        y_train=y_train_res,
        X_test=X_test_flat,
        y_test=y_test_1d
    )
    print("--- Tree Model Preprocessing Complete ---")


if __name__ == "__main__":
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print(f"Created directory: {SAVE_PATH}")

    process_and_save_lstm_data()
    process_and_save_tree_data()

    print(f"\nAll processing complete. Files saved in {SAVE_PATH}")