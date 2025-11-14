import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.utils import to_categorical
from src.utils.create_window_util import create_windows


def train_and_save_lstm(data_path, save_path, target_column, window_size, step_size):

    print("Loading final processed data...")
    df = pd.read_csv(data_path)

    print("Separating data...")
    df.dropna(inplace=True)
    X = df.drop(columns=[col for col in df.columns if 'activity' in col])
    y = df[target_column].astype(int)
    num_classes = len(y.unique())
    y_categorical = to_categorical(y, num_classes=num_classes)
    print(f"Creating sliding windows (size={window_size}, step={step_size})...")
    X_win, y_win = create_windows(X, pd.Series(y_categorical.tolist()), window_size, step_size)
    print(f"  - Windowed X shape: {X_win.shape}")
    print(f"  - Windowed y shape: {y_win.shape}")

    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
    print(f"  - Training set size: {len(X_train)}")
    print(f"  - Test set size: {len(X_test)}")

    print("Building the LSTM model...")
    model = Sequential([
        # The input layer must match the shape of our windows (window_size, num_features)
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # The output layer has one neuron per activity
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Start with a few epochs to see how it goes
        batch_size=128,
        validation_split=0.1,  # Use part of the training data for validation
        verbose=1
    )
    model.save(save_path)

    print("\nEvaluating the model on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred))

def train_and_save_rf(data_path, save_path, target_column, window_size, step_size):
    print("Loading final processed data...")
    df = pd.read_csv(data_path)

    print("Separating Features and Target...")
    df.dropna(inplace=True)
    X = df.drop(columns=[col for col in df.columns if 'activity' in col])
    y = df[target_column].astype(int)

    print(f"Creating sliding windows (size={window_size}, step={step_size})...")
    X_win, y_win = create_windows(X, y, window_size, step_size)
    print(f"  - Initial windowed X shape: {X_win.shape}")

    print("Flattening window data...")
    n_samples, n_timesteps, n_features = X_win.shape
    X_flattened = X_win.reshape((n_samples, n_timesteps * n_features))
    print(f"  - Flattened X shape: {X_flattened.shape}")

    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_win, test_size=0.2, random_state=42)
    print(f"  - Training set size: {len(X_train)}")
    print(f"  - Test set size: {len(X_test)}")

    print("\nBuilding and training the Random Forest model...")
    # n_estimators is the number of trees in the forest.
    # n_jobs=-1 uses all available CPU cores for faster training.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    print("\nEvaluating the model on the test set...")
    y_pred = model.predict(X_test)
    joblib.dump(model, save_path)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    # You may need to create a mapping from integer back to activity name for readability
    activity_names = ["BATHROOM ACTIVITY", "CHORES", "COOK", "DISHWASHING", "DRESS", "EAT", "LAUNDRY",
                      "MAKE SIMPLE FOOD", "OUT HOME", "PET", "READ", "RELAX", "SHOWER", "SLEEP",
                      "TAKE MEDS", "WATCH TV", "WORK", "OTHER"]
    print(classification_report(y_test, y_pred, target_names=activity_names))


def train_and_save_dt(data_path, save_path, target_column, window_size, step_size):
    print("Loading final processed data...")
    df = pd.read_csv(data_path)

    print("Separating Features and Target...")
    df.dropna(inplace=True)
    X = df.drop(columns=[col for col in df.columns if 'activity' in col])
    y = df[target_column].astype(int)

    print(f"Creating sliding windows (size={window_size}, step={step_size})...")
    X_win, y_win = create_windows(X, y, window_size, step_size)
    print(f"  - Initial windowed X shape: {X_win.shape}")

    print("Flattening window data...")
    n_samples, n_timesteps, n_features = X_win.shape
    X_flattened = X_win.reshape((n_samples, n_timesteps * n_features))
    print(f"  - Flattened X shape: {X_flattened.shape}")

    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_win, test_size=0.2, random_state=42)
    print(f"  - Training set size: {len(X_train)}")
    print(f"  - Test set size: {len(X_test)}")

    print("Building and training the Decision Tree model...")
    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)
    print("  - Model training complete!")

    print("Evaluating the model on the test set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    activity_names = ["BATHROOM ACTIVITY", "CHORES", "COOK", "DISHWASHING", "DRESS", "EAT", "LAUNDRY",
                      "MAKE SIMPLE FOOD", "OUT HOME", "PET", "READ", "RELAX", "SHOWER", "SLEEP",
                      "TAKE MEDS", "WATCH TV", "WORK", "OTHER"]
    print(classification_report(y_test, y_pred, target_names=activity_names))

    print("Saving the Decision Tree model...")
    joblib.dump(model, save_path)
    print("  - Model saved successfully!")

def train_and_save_gru(data_path, save_path, target_column, window_size, step_size):
    print("Loading final processed data...")
    df = pd.read_csv(data_path)

    print("Separating data...")
    df.dropna(inplace=True)
    X = df.drop(columns=[col for col in df.columns if 'activity' in col])
    y = df[target_column].astype(int)
    num_classes = len(y.unique())
    y_categorical = to_categorical(y, num_classes=num_classes)

    print(f"Creating sliding windows (size={window_size}, step={step_size})...")
    X_win, y_win = create_windows(X, pd.Series(y_categorical.tolist()), window_size, step_size)
    print(f"  - Windowed X shape: {X_win.shape}")
    print(f"  - Windowed y shape: {y_win.shape}")

    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
    print(f"  - Training set size: {len(X_train)}")
    print(f"  - Test set size: {len(X_test)}")

    print("Building the GRU model...")
    model = Sequential([
        # The input layer must match the shape of our windows (window_size, num_features)
        GRU(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.5),
        GRU(64),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # The output layer has one neuron per activity
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    model.save(save_path)

    print("\nEvaluating the model on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred))