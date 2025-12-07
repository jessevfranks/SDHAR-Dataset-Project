from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def train_and_save_lstm_numpy(save_path, data_file, output_name):
    activity_names = ["BATHROOM ACTIVITY", "CHORES", "COOK", "DISHWASHING", "DRESS", "EAT", "LAUNDRY",
                      "MAKE SIMPLE FOOD", "OUT HOME", "PET", "READ", "RELAX", "SHOWER", "SLEEP",
                      "TAKE MEDS", "WATCH TV", "WORK", "OTHER"]

    print(f"Loading data from {data_file}...")
    # 1. Load the preprocessed LSTM data
    data = np.load(data_file)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    y_test_1d_labels = data['y_test_1d_labels']  # For the final report

    # Get shape info from the loaded data
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_classes = y_train.shape[1]

    print(f"  - Training X shape: {X_train.shape}")
    print(f"  - Training y shape: {y_train.shape}")
    print(f"  - Test X shape: {X_test.shape}")
    print(f"  - Test y shape: {y_test.shape}")

    print("\nBuilding the LSTM model...")
    model_lstm = Sequential([
        LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])

    model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_lstm.summary()

    print("\nTraining the LSTM model on SMOTE data...")
    history = model_lstm.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    model_lstm.save(os.path.join(save_path, output_name))

    print("\nEvaluating the LSTM model on the *original* (unbalanced) test set...")
    loss, accuracy = model_lstm.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    y_pred_probs = model_lstm.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)  # Convert one-hot back to 1D

    print("\nClassification Report (LSTM with SMOTE for small window model):")
    # Compare the 1D predicted labels with the 1D true labels
    print(classification_report(y_test_1d_labels, y_pred_labels, target_names=activity_names))



def train_and_save_rf_numpy(save_path, data_file, output_name):
    activity_names = ["BATHROOM ACTIVITY", "CHORES", "COOK", "DISHWASHING", "DRESS", "EAT", "LAUNDRY",
                      "MAKE SIMPLE FOOD", "OUT HOME", "PET", "READ", "RELAX", "SHOWER", "SLEEP",
                      "TAKE MEDS", "WATCH TV", "WORK", "OTHER"]

    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"  - Training X shape: {X_train.shape}")
    print(f"  - Training y shape: {y_train.shape}")
    print(f"  - Test X shape: {X_test.shape}")
    print(f"  - Test y shape: {y_test.shape}")

    print("\nBuilding and training the Random Forest model on SMOTE data...")
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model_rf.fit(X_train, y_train)
    print("  - Model training complete!")

    print("\nEvaluating the RF model on the *original* (unbalanced) test set...")
    y_pred = model_rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report (Random Forest with SMOTE):")
    print(classification_report(y_test, y_pred, target_names=activity_names))

    print("Saving the RF model...")
    joblib.dump(model_rf, os.path.join(save_path, output_name))
    print("  - Model saved successfully!")


def train_and_save_dt_numpy(save_path, data_file, output_name):
    activity_names = ["BATHROOM ACTIVITY", "CHORES", "COOK", "DISHWASHING", "DRESS", "EAT", "LAUNDRY",
                      "MAKE SIMPLE FOOD", "OUT HOME", "PET", "READ", "RELAX", "SHOWER", "SLEEP",
                      "TAKE MEDS", "WATCH TV", "WORK", "OTHER"]

    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"  - Training X shape: {X_train.shape}")
    print(f"  - Training y shape: {y_train.shape}")
    print(f"  - Test X shape: {X_test.shape}")
    print(f"  - Test y shape: {y_test.shape}")

    print("\nBuilding and training the Decision Tree model on SMOTE data...")
    model_dt = DecisionTreeClassifier(random_state=42)

    model_dt.fit(X_train, y_train)
    print("  - Model training complete!")

    print("\nEvaluating the DT model on the *original* (unbalanced) test set...")
    y_pred = model_dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report (Decision Tree with SMOTE):")
    print(classification_report(y_test, y_pred, target_names=activity_names))

    print("Saving the DT model...")
    joblib.dump(model_dt, os.path.join(save_path, output_name))
    print("  - Model saved successfully!")