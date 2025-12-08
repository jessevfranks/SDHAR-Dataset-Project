# synthetic_llm_interface.py
"""
Interactive LLM-driven synthetic data tester for SDHAR models.
UPDATED VERSION:
- Displays valid activities.
- Maps unknowns to 'OTHER'.
- Prints Confusion Matrix.
- robust Normalization handling.
"""

import os
import json
import textwrap
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from openai import OpenAI, OpenAIError


# -------------------------- CONFIG --------------------------------------

# Path to your processed SDHAR data
PROCESSED_DATA_CSV = "../../processed_data/SDHAR/final_processed_data_ALL_DAYS.csv"

# Window configuration
WINDOW_SIZE = 60
STEP_SIZE = 30
SECONDS_PER_ROW = 2.0

LABEL_COL = "activity_user_1"

# Activity names
activity_names = [
    "BATHROOM ACTIVITY", "CHORES", "COOK", "DISHWASHING", "DRESS", "EAT",
    "LAUNDRY", "MAKE SIMPLE FOOD", "OUT HOME", "PET", "READ", "RELAX",
    "SHOWER", "SLEEP", "TAKE MEDS", "WATCH TV", "WORK", "OTHER"
]
activity_id_to_name = {i: name for i, name in enumerate(activity_names)}
activity_name_to_id = {name: i for i, name in enumerate(activity_names)}

# Model configuration
MODEL_CONFIG = {
    "decision_tree_first": {
        "pretty_name": "Decision Tree (base)",
        "path": "../../models/SDHAR/DecisionTree_first_iteration.joblib",
        "kind": "tree",
        "normalized": False,
    },
    "decision_tree_normalized": {
        "pretty_name": "Decision Tree (normalized)",
        "path": "../../models/SDHAR/DecisionTree_normzlized.joblib",
        "kind": "tree",
        "normalized": True,
    },
    "random_forest_first": {
        "pretty_name": "Random Forest (base)",
        "path": "../../models/SDHAR/RandomForest_first_iteration.joblib",
        "kind": "tree",
        "normalized": False,
    },
    "random_forest_normalized": {
        "pretty_name": "Random Forest (normalized)",
        "path": "../../models/SDHAR/RandomForest_normalized.joblib",
        "kind": "tree",
        "normalized": True,
    },
    "lstm_first": {
        "pretty_name": "LSTM (base)",
        "path": "../../models/SDHAR/LSTM_first_iteration.keras",
        "kind": "lstm",
        "normalized": False,
    },
    "lstm_normalized": {
        "pretty_name": "LSTM (normalized)",
        "path": "../../models/SDHAR/LSTM_normalized.keras",
        "kind": "lstm",
        "normalized": True,
    },
}

LLM_MODEL = "gpt-4o"


# ----------------------- DATA & SYNTHETIC BUILDER ------------------------


def build_activity_segments(df: pd.DataFrame, label_col: str) -> dict:
    labels = df[label_col].values
    segments_by_label = {}
    start_idx = None
    current_label = None

    for idx, label in enumerate(labels):
        if pd.isna(label):
            if current_label is not None:
                segments_by_label.setdefault(current_label, []).append((start_idx, idx))
                current_label = None
                start_idx = None
            continue

        label_int = int(label)

        if current_label is None:
            current_label = label_int
            start_idx = idx
        elif label_int != current_label:
            segments_by_label.setdefault(current_label, []).append((start_idx, idx))
            current_label = label_int
            start_idx = idx

    if current_label is not None and start_idx is not None:
        segments_by_label.setdefault(current_label, []).append((start_idx, len(labels)))

    return segments_by_label


def script_names_to_numeric(script_by_name, name_to_id, seconds_per_row=SECONDS_PER_ROW):
    numeric_script = []
    for step in script_by_name:
        name = step["activity"]
        # Fallback for safety, though LLM should handle it
        if name not in name_to_id:
            print(f"[WARN] Unknown activity '{name}' found. Mapping to OTHER.")
            name = "OTHER"
        
        label_id = name_to_id[name]
        length_rows = int((step["minutes"] * 60.0) / seconds_per_row)
        numeric_script.append({"label": label_id, "length_rows": length_rows})

    return numeric_script


def build_synthetic_from_numeric_script(
    df: pd.DataFrame,
    segments_by_label: dict,
    script,
    random_state=None,
    recompute_time_features: bool = False,
    seconds_per_row: float = SECONDS_PER_ROW,
    min_segment_len: Optional[int] = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    pieces = []

    for step in script:
        label = int(step["label"])
        target_len = int(step["length_rows"])

        if label not in segments_by_label:
            # If we don't have segments for this label (e.g. OTHER never happened in source)
            # We skip it or raise error. 
            print(f"[WARN] No source segments found for label ID {label}. Skipping.")
            continue

        segs = segments_by_label[label]
        if min_segment_len is not None:
            valid_segs = [(s, e) for (s, e) in segs if (e - s) >= min_segment_len]
            # Fallback: if no segments are long enough, take the longest available
            if not valid_segs and segs:
                # find max len
                longest = max(segs, key=lambda x: x[1]-x[0])
                valid_segs = [longest]
            segs = valid_segs

        if not segs:
             print(f"[WARN] No valid segments for label ID {label}. Skipping.")
             continue

        length_left = target_len
        while length_left > 0:
            start, end = segs[rng.integers(len(segs))]
            seg_len = end - start
            if seg_len <= 0: continue

            take_len = min(seg_len, length_left)
            
            # If segment is shorter than window, we just take it all. 
            # (Note: this might cause windowing issues later if total length < window_size)
            if seg_len > take_len:
                max_offset = seg_len - take_len
                offset = rng.integers(max_offset + 1)
            else:
                offset = 0

            sub_start = start + offset
            sub_end = sub_start + take_len

            piece = df.iloc[sub_start:sub_end]
            pieces.append(piece)
            length_left -= take_len

    if not pieces:
        raise ValueError("Generated synthetic data is empty (no valid segments found).")
        
    synthetic = pd.concat(pieces, ignore_index=True)

    if recompute_time_features and "sin_time" in synthetic.columns:
        n = len(synthetic)
        t = np.arange(n) * seconds_per_row
        angles = 2 * np.pi * (t % 86400) / 86400.0
        synthetic["sin_time"] = np.sin(angles)
        synthetic["cos_time"] = np.cos(angles)

    return synthetic


def build_synthetic_from_name_script(
    df: pd.DataFrame,
    segments_by_label: dict,
    script_by_name,
    activity_name_to_id: dict,
    random_state=None,
    recompute_time_features: bool = False,
    seconds_per_row: float = SECONDS_PER_ROW,
    min_segment_len: Optional[int] = None,
) -> pd.DataFrame:
    numeric_script = script_names_to_numeric(
        script_by_name,
        activity_name_to_id,
        seconds_per_row=seconds_per_row,
    )
    return build_synthetic_from_numeric_script(
        df=df,
        segments_by_label=segments_by_label,
        script=numeric_script,
        random_state=random_state,
        recompute_time_features=recompute_time_features,
        seconds_per_row=seconds_per_row,
        min_segment_len=min_segment_len,
    )


# -------------------------- MODEL EVAL HELPERS ---------------------------


def create_windows(X: pd.DataFrame, y: pd.Series, window_size: int, step_size: int):
    X_win, y_win = [], []
    for i in range(0, len(X) - window_size, step_size):
        window = X.iloc[i : i + window_size].values
        label = y.iloc[i + window_size]
        X_win.append(window)
        y_win.append(label)
    return np.array(X_win), np.array(y_win)


def df_to_windows(
    df: pd.DataFrame,
    label_col: str,
    window_size: int,
    step_size: int,
    scaler: Optional[MinMaxScaler] = None,
):
    # Drop activity columns
    X_all = df.drop(columns=[c for c in df.columns if "activity" in c.lower()])
    y_all = df[label_col]

    # Handle NaNs in target
    mask = y_all.notna()
    X = X_all[mask].reset_index(drop=True)
    y = y_all[mask].astype(int).reset_index(drop=True)

    # === NORMALIZATION FIX ===
    # If a scaler is provided (because model is normalized), we transform
    # the synthetic data using the scaler fitted on the SOURCE data.
    if scaler is not None:
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    X_win, y_win = create_windows(X, y, window_size, step_size)
    return X_win, y_win


def fit_scaler_for_normalized_models(df_source: pd.DataFrame) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on all non-activity columns of the SOURCE data.
    """
    X = df_source.drop(columns=[c for c in df_source.columns if "activity" in c.lower()])
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler


def load_all_models() -> dict:
    models = {}
    for model_id, cfg in MODEL_CONFIG.items():
        path = cfg["path"]
        if not os.path.exists(path):
            # Just a warning so script doesn't crash if one model is missing
            print(f"[WARN] Model path missing for {model_id}: {path}")
            continue

        if cfg["kind"] == "tree":
            models[model_id] = joblib.load(path)
        elif cfg["kind"] == "lstm":
            models[model_id] = load_model(path)
        else:
            raise ValueError(f"Unknown model kind: {cfg['kind']}")
    return models


def evaluate_model_on_synthetic(
    model_id: str,
    synthetic_df: pd.DataFrame,
    models: dict,
    scaler: MinMaxScaler,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
):
    if model_id not in models:
        raise ValueError(f"Model '{model_id}' is not loaded.")

    cfg = MODEL_CONFIG[model_id]
    model = models[model_id]

    # Decide whether to use the scaler
    use_scaler = scaler if cfg["normalized"] else None

    X_win, y_win = df_to_windows(
        synthetic_df,
        label_col=LABEL_COL,
        window_size=window_size,
        step_size=step_size,
        scaler=use_scaler,
    )
    
    if len(X_win) == 0:
        raise ValueError("Synthetic data generated 0 windows. Make sure duration > window_size.")

    y_true = y_win.astype(int)

    # Predict
    if cfg["kind"] == "tree":
        n_samples, n_steps, n_feats = X_win.shape
        X_flat = X_win.reshape(n_samples, n_steps * n_feats)
        y_pred = model.predict(X_flat).astype(int)
    elif cfg["kind"] == "lstm":
        y_prob = model.predict(X_win, verbose=0)
        y_pred = np.argmax(y_prob, axis=1).astype(int)
    else:
        raise ValueError(f"Unknown model kind: {cfg['kind']}")

    accuracy = accuracy_score(y_true, y_pred)
    
    # Classification Report
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(activity_names))),
        target_names=activity_names,
        zero_division=0,
        digits=3,
    )

    # === CONFUSION MATRIX ===
    # Generate matrix
    cm = confusion_matrix(
        y_true, 
        y_pred, 
        labels=list(range(len(activity_names)))
    )
    # Convert to pandas DF for pretty printing
    cm_df = pd.DataFrame(
        cm, 
        index=[f"True {n}" for n in activity_names],
        columns=[f"Pred {n}" for n in activity_names]
    )

    return accuracy, report, cm_df, y_true, y_pred


# ---------------------------- LLM HELPERS --------------------------------


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            if lines[-1].strip().startswith("```"):
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            s = "\n".join(lines).strip()
    return s


def llm_json_request(client: OpenAI, system_prompt: str, user_prompt: str) -> dict:
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = completion.choices[0].message.content
    content_clean = strip_code_fences(content)
    try:
        return json.loads(content_clean)
    except json.JSONDecodeError:
        print(f"[LLM RAW RESPONSE]\n{content}\n")
        raise


def llm_generate_activity_script(client: OpenAI, user_description: str):
    valid_activities_text = "\n".join(f"- {name}" for name in activity_names)

    # UPDATED PROMPT: Explicitly handle unknown activities by mapping to OTHER
    system_prompt = f"""
You are an assistant that converts natural-language descriptions of daily routines
into a structured activity script for a smart-home activity recognition dataset.

VALID activity names (must match exactly, case-sensitive):
{valid_activities_text}

The user will describe a sequence of activities and how long they do them.
Rules:
1. Map each described segment to one of the VALID activity names above.
2. If the user describes an activity that is NOT in the list (e.g. "Skydiving", "Painting"), 
   you MUST map it to "OTHER". Do not invent new names.
3. Estimate duration in whole minutes.

Return ONLY valid JSON:
{{
  "activities": [
    {{"activity": "<ACTIVITY_NAME>", "minutes": <integer_minutes>}},
    ...
  ]
}}
""".strip()

    user_prompt = textwrap.dedent(
        f"""
        User description:
        {user_description}

        Convert to JSON.
        """
    )

    try:
        data = llm_json_request(client, system_prompt, user_prompt)
        activities = data.get("activities", [])
        if not activities:
            return None
        return activities
    except Exception as e:
        print(f"[ERROR] Failed to parse activity script from LLM: {e}")
        return None


def llm_choose_model(client: OpenAI, user_model_request: str) -> Optional[str]:
    options_text = "\n".join(
        f"- {model_id}: {cfg['pretty_name']}"
        for model_id, cfg in MODEL_CONFIG.items()
    )

    system_prompt = f"""
You are an assistant that selects one of several machine learning models.
Available models:
{options_text}

Choose the single best matching model_id.
Return ONLY JSON:
{{ "model_id": "<selected_model_id>" }}
"""
    user_prompt = f"""User request: "{user_model_request}" """

    try:
        data = llm_json_request(client, system_prompt, user_prompt)
        return data.get("model_id", "")
    except Exception as e:
        print(f"[ERROR] LLM model choice failed: {e}")
        return None


def llm_explain_metrics(
    client: OpenAI,
    model_name: str,
    accuracy: float,
    report: str,
    cm_df: pd.DataFrame
) -> str:
    # We pass the confusion matrix summary to the LLM as well
    # To keep token count reasonable, we might just pass the "most confused" pairs
    # But passing the whole report is usually enough.
    
    system_prompt = """
You are a data scientist explaining the performance of a classifier.
Highlight accuracy, per-class metrics, and any confusion between classes.
Keep it concise and easy to understand.
"""
    user_prompt = textwrap.dedent(
        f"""
        Model: {model_name}
        Overall accuracy: {accuracy:.4f}

        Classification Report:
        {report}

        (You may also infer confusion from the low recall/precision of specific classes).

        Explain the results. 
        """
    )

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    return completion.choices[0].message.content


# ------------------------------- MAIN LOOP -------------------------------


def main():
    print("=" * 80)
    print("SDHAR Synthetic Data LLM Tester")
    print("=" * 80)

    # 1. API KEY CHECK
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[!] OPENAI_API_KEY not found.")
        print("    Please enter your OpenAI API key (sk-...):")
        api_key = input("    API Key: ").strip()
        if not api_key:
            return

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"[Error] Init OpenAI: {e}")
        return

    # 2. LOAD DATA
    print("\n[INFO] Loading processed data...")
    if not os.path.exists(PROCESSED_DATA_CSV):
        print(f"[ERROR] File not found: {PROCESSED_DATA_CSV}")
        return

    df_source = pd.read_csv(PROCESSED_DATA_CSV)
    segments_by_label = build_activity_segments(df_source, LABEL_COL)
    
    # 3. FIT SCALER (Fix for Normalization Issue)
    # We fit the scaler on the ENTIRE source dataset once.
    # When testing normalized models, we apply this global scaler to the synthetic slices.
    print("[INFO] Fitting global scaler for normalized models...")
    scaler = fit_scaler_for_normalized_models(df_source)

    # 4. LOAD MODELS
    print("[INFO] Loading models...")
    models = load_all_models()
    print("[INFO] Models loaded:", list(models.keys()))
    print()

    # 5. LOOP
    while True:
        print("-" * 80)
        print("Supported Activities:")
        # Print activities in a readable way
        print(textwrap.fill(", ".join(activity_names), width=80))
        print("-" * 80)

        user_desc = input("\nDescribe activity sequence (or 'quit'): ").strip()
        if user_desc.lower() in {"quit", "exit"}:
            break

        activity_script = llm_generate_activity_script(client, user_desc)
        if not activity_script:
            print("Could not interpret script. Try again.")
            continue

        print("\nInterpreted Script:")
        for step in activity_script:
            print(f"  - {step['activity']}: {step['minutes']} mins")

        # Build Synthetic
        print("\n[INFO] Building synthetic dataset...")
        try:
            synthetic_df = build_synthetic_from_name_script(
                df=df_source,
                segments_by_label=segments_by_label,
                script_by_name=activity_script,
                activity_name_to_id=activity_name_to_id,
                random_state=42,
                min_segment_len=WINDOW_SIZE,
            )
            print(f"[INFO] Synthetic data rows: {len(synthetic_df)}")
        except ValueError as e:
            print(f"[ERROR] {e}")
            continue

        # Test Models Loop
        while True:
            print("\nWhich model to test? (e.g. 'normalized LSTM', 'base tree')")
            print("Type 'back' for new sequence, 'quit' to exit.")
            user_model_req = input("Model: ").strip()

            if user_model_req.lower() in {"quit", "exit"}:
                return
            if user_model_req.lower() == "back":
                break

            model_id = llm_choose_model(client, user_model_req)
            if not model_id or model_id not in models:
                print("Model not found/recognized. Try again.")
                continue

            cfg = MODEL_CONFIG[model_id]
            print(f"\n[INFO] Evaluating '{model_id}'...")

            try:
                acc, rep, cm_df, y_true, y_pred = evaluate_model_on_synthetic(
                    model_id,
                    synthetic_df,
                    models,
                    scaler=scaler, # Passing the globally fitted scaler
                )
            except Exception as e:
                print(f"[ERROR] Evaluation failed: {e}")
                continue

            print("\n" + "="*30 + " RESULTS " + "="*30)
            print(f"Model: {cfg['pretty_name']}")
            print(f"Accuracy: {acc:.4f}\n")
            
            print("--- Confusion Matrix (Subset of active classes) ---")
            # Only show columns/rows that actually appear in y_true or y_pred to save space
            unique_labels = sorted(list(set(y_true) | set(y_pred)))
            relevant_names = [activity_names[i] for i in unique_labels]
            # Filter the big DF down to just relevant rows/cols
            cm_small = cm_df.iloc[unique_labels, unique_labels]
            print(cm_small)
            print("\n")
            
            print("--- Classification Report ---")
            print(rep)

            print("--- LLM Interpretation ---")
            expl = llm_explain_metrics(client, cfg["pretty_name"], acc, rep, cm_df)
            print(expl)

            if input("\nTest another model on this data? (y/n): ").lower() != 'y':
                break

if __name__ == "__main__":
    main()