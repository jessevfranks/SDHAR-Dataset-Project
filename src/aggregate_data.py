import pandas as pd
import re
import glob
import os
import numpy as np

def load_data_and_activities(given_day_path):
    """
    Loads specified columns from CSV files and combines them into a single time-indexed DataFrame.
    The result will be a very sparse dataset.
    """
    print(f"Searching for all data files in: {given_day_path}")
    all_files = glob.glob(os.path.join(given_day_path, "*", "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {given_day_path}.")

    # Defines which columns to load from files in each directory
    column_map = {
        "Contact": ["state"],
        "Illuminance": ["illuminance_lux"],
        "Movement": ["state"],
        "Power": ["power"],
        "Presences": ["state", "distance"],
        "TempHum": ["humidity", "temperature"],
        "Vibration": ["state"]
    }

    df_list = []
    for f in all_files:
        file_name = os.path.basename(f)
        parent_dir = os.path.basename(os.path.dirname(f))

        cols_to_load = []
        try:
            # Always get the first column (timestamp)
            all_file_cols = pd.read_csv(f, nrows=0).columns
            time_col_name = all_file_cols[0]
            cols_to_load.append(time_col_name)
        except IndexError:
            print(f"Skipping empty or invalid file: {f}")
            continue

        if "activity_user" in file_name:
            if len(all_file_cols) > 1:
                # For activity files, just get the second column
                cols_to_load.append(all_file_cols[1])
        elif parent_dir in column_map:
            cols_to_load.extend(column_map[parent_dir])
        else:
            continue

        try:
            valid_cols_to_load = [col for col in cols_to_load if col in all_file_cols]
            if len(valid_cols_to_load) <= 1:
                continue

            df = pd.read_csv(f, usecols=valid_cols_to_load)
        except Exception as e:
            print(f"Warning: Could not process file {f}. Error: {e}")
            continue

        df["timestamp"] = pd.to_datetime(df[time_col_name], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        try:
            # If the timestamp is already timezone-aware, convert it to UTC.
            df["timestamp"] = df["timestamp"].dt.tz_convert('UTC')
        except TypeError:
            # If the timestamp is timezone-naive, assign the UTC timezone.
            df["timestamp"] = df["timestamp"].dt.tz_localize('UTC')

        df = df.drop_duplicates(subset=["timestamp"], keep='last')
        df = df.set_index("timestamp")

        if "activity_user" in file_name:
            user_name = file_name.split(".")[0].split("_", 1)[1]
            activity_col = [col for col in df.columns if col != time_col_name][0]
            df = df.rename(columns={activity_col: f"activity_{user_name}"})
            df = df.drop(columns=[time_col_name])
            df = df[[f"activity_{user_name}"]]
        else:
            sensor_prefix = f"{os.path.basename(os.path.dirname(f))}_{os.path.splitext(file_name)[0]}"
            df = df.drop(columns=[time_col_name])
            df = df.rename(columns={col: f"{sensor_prefix}_{col}" for col in df.columns})

        df_list.append(df)

    if not df_list:
        raise ValueError("No data was successfully loaded with the specified columns.")

    print("Combining data...")
    master_df = pd.concat(df_list, axis=1)

    print(f"Loaded {len(master_df)} total data points across {master_df.shape[1]} columns.")
    return master_df


def densify_dataset(sparse_input_df):
    """
    Loads a sparse time-series dataset, makes it dense using a forward-fill
    strategy, and saves it to a new CSV file.
    """
    print("Applying forward-fill to make the dataset dense...")
    df_dense = sparse_input_df.ffill()

    # Fill remaining NaNs in object/categorical columns with 'OFF'
    object_cols = df_dense.select_dtypes(include=['object']).columns
    df_dense[object_cols] = df_dense[object_cols].fillna('OFF')

    # Fill remaining NaNs in numeric columns with 0
    numeric_cols = df_dense.select_dtypes(include=['number']).columns
    df_dense[numeric_cols] = df_dense[numeric_cols].fillna(0)

    print(f"New dense dataset saved with {df_dense.shape[0]} rows and {df_dense.shape[1]} columns.")
    return df_dense

def rename_columns(df):
    """
    Loads a dataset, renames columns based on a set of specific rules,
    and saves the result to a new CSV file.
    """

    print("Renaming columns based on specified rules...")

    new_column_names = {}
    for col in df.columns:
        new_name = col  # Default to the old name
        if col.startswith('Contact_c') and col.endswith('_state'):
            new_name = col.split('_')[1]
        # Illuminance_l1_illuminance_lux -> l1 (for l1-l2)
        elif col.startswith('Illuminance_l') and col.endswith('_lux'):
            new_name = col.split('_')[1]
        # Movement_m1_state -> m1 (for m1-m8)
        elif col.startswith('Movement_m') and col.endswith('_state'):
            new_name = col.split('_')[1]
        # Power_p1_power -> p1 (for p1-p2)
        elif col.startswith('Power_p') and col.endswith('_power'):
            new_name = col.split('_')[1]
        # Presences_user1_presence_state -> user1_state (for user1-2)
        elif 'Presences_user' in col and 'presence_state' in col:
            new_name = col.split('_')[1] + '_state'
        # Presences_user1_presence_distance -> user1_distance (for user1-2)
        elif 'Presences_user' in col and 'presence_distance' in col:
            new_name = col.split('_')[1] + '_distance'
        # TempHum_th1_humidity -> h1 (for th1-2)
        elif col.startswith('TempHum_th') and col.endswith('_humidity'):
            num = re.search(r'th(\d+)', col).group(1)
            new_name = f'h{num}'
        # TempHum_th1_temperature -> t1 (for th1-2)
        elif col.startswith('TempHum_th') and col.endswith('_temperature'):
            num = re.search(r'th(\d+)', col).group(1)
            new_name = f't{num}'
        # Vibration_v1_state -> v1 (for v1-v11)
        elif col.startswith('Vibration_v') and col.endswith('_state'):
            new_name = col.split('_')[1]

        new_column_names[col] = new_name

    df.rename(columns=new_column_names, inplace=True)
    print("Columns successfully renamed.")
    return df


def encode_data(df):
    activity_map = {
        "BATHROOM ACTIVITY": 0, "CHORES": 1, "COOK": 2, "DISHWASHING": 3, "DRESS": 4,
        "EAT": 5, "LAUNDRY": 6, "MAKE SIMPLE FOOD": 7, "OUT HOME": 8, "PET": 9,
        "READ": 10, "RELAX": 11, "SHOWER": 12, "SLEEP": 13, "TAKE MEDS": 14,
        "WATCH TV": 15, "WORK": 16, "OTHER": 17, "OFF": 13
    }

    vibration_map = {
        'OFF': 0, 'vibration': 1, 'tilt': 2, 'drop': 3
    }

    print("Encoding the activities...")

    if 'activity_user_1' in df.columns:
        df['activity_user_1'] = df['activity_user_1'].map(activity_map)
    if 'activity_user_2' in df.columns:
        df['activity_user_2'] = df['activity_user_2'].map(activity_map)

    print("Encoding the locations...")

    state_categories = ["banho", "cocina", "dormitorio", "estudio", "not_home", "pasillo", "salon1"]
    state_map = {category: i for i, category in enumerate(state_categories)}
    state_map["OFF"] = state_map["dormitorio"]
    if 'user1_state' in df.columns:
        df['user1_state'] = df['user1_state'].map(state_map)
    if 'user2_state' in df.columns:
        df['user2_state'] = df['user2_state'].map(state_map)

    print("Encoding the vibrations...")
    vibration_cols = [col for col in df.columns if col.startswith('v')]
    if vibration_cols:
        print(f"Applying vibration map: {vibration_map}")
        for col in vibration_cols:
            df[col] = df[col].map(vibration_map)

    print("Applying cyclical encoding to the timestamp...")
    seconds_since_midnight = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
    seconds_in_a_day = 24 * 60 * 60

    df['sin_time'] = np.sin(2 * np.pi * seconds_since_midnight / seconds_in_a_day)
    df['cos_time'] = np.cos(2 * np.pi * seconds_since_midnight / seconds_in_a_day)
    df.reset_index(drop=True, inplace=True)

    print("Encoding OFF/ON to binary...")

    binary_map = {'OFF': 0, 'ON': 1, 'off': 0, 'on': 1}
    remaining_object_cols = df.select_dtypes(include=['object']).columns
    for col in remaining_object_cols:
        df[col] = df[col].map(binary_map)
    return df

if __name__ == "__main__":
    ROOT_DATA_DIR = "../data/SDHAR"
    SAVE_PATH = "../processed_data/"
    TIME_WINDOW = "2s"

    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    all_days_dirs = sorted(
        glob.glob(os.path.join(ROOT_DATA_DIR, 'Day_*')),
        key= lambda path: int(os.path.basename(path).split('_')[1])
    )
    all_processed_dfs = []

    for day_dir in all_days_dirs:
        day_name = os.path.basename(day_dir)
        day_path = os.path.join(day_dir, day_name)

        sparse_df = load_data_and_activities(day_path)

        if sparse_df is None:
            print(f"No data found for '{day_dir}'.")
            continue

        resampled_sparse_df = sparse_df.resample(TIME_WINDOW).last()
        densified_df = densify_dataset(resampled_sparse_df)
        renamed_df = rename_columns(densified_df)
        day_csv = encode_data(renamed_df)
        all_processed_dfs.append(day_csv)

    final_df = pd.concat(all_processed_dfs)
    output_path = os.path.join(SAVE_PATH, "final_processed_data_ALL_DAYS.csv")
    final_df.to_csv(output_path, index=False)




