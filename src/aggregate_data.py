import pandas as pd
import re
import glob
import os
import numpy as np

def load_data_and_activities(day_path):
    """
    Loads specified columns from CSV files and combines them into a single time-indexed DataFrame.
    The result will be a very sparse dataset.
    """
    print(f"Searching for all data files in: {day_path}")
    all_files = glob.glob(os.path.join(day_path, "*", "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {day_path}.")

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


def densify_dataset(sparse_csv_path, output_csv_path):
    """
    Loads a sparse time-series dataset, makes it dense using a forward-fill
    strategy, and saves it to a new CSV file.
    """
    try:
        df = pd.read_csv(sparse_csv_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(
            f"ERROR: The file was not found at '{sparse_csv_path}'. Please ensure the file is in the correct directory.")
        return

    print("Applying forward-fill to make the dataset dense...")
    df_dense = df.ffill()

    # Fill remaining NaNs in object/categorical columns with 'OFF'
    object_cols = df_dense.select_dtypes(include=['object']).columns
    df_dense[object_cols] = df_dense[object_cols].fillna('OFF')

    # Fill remaining NaNs in numeric columns with 0
    numeric_cols = df_dense.select_dtypes(include=['number']).columns
    df_dense[numeric_cols] = df_dense[numeric_cols].fillna(0)
    df_dense.to_csv(output_csv_path)

    print(f"New dense dataset saved with {df_dense.shape[0]} rows and {df_dense.shape[1]} columns.")

def rename_columns(input_csv_path, output_csv_path):
    """
    Loads a dataset, renames columns based on a set of specific rules,
    and saves the result to a new CSV file.
    """
    try:
        df = pd.read_csv(input_csv_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{input_csv_path}'. Please ensure the file exists.")
        return

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
    df.to_csv(output_csv_path)
    print("Columns successfully renamed.")


def encode_data(input_csv_path, output_csv_path):
    activity_map = {
        "BATHROOM ACTIVITY": 0, "CHORES": 1, "COOK": 2, "DISHWASHING": 3, "DRESS": 4,
        "EAT": 5, "LAUNDRY": 6, "MAKE SIMPLE FOOD": 7, "OUT HOME": 8, "PET": 9,
        "READ": 10, "RELAX": 11, "SHOWER": 12, "SLEEP": 13, "TAKE MEDS": 14,
        "WATCH TV": 15, "WORK": 16, "OTHER": 17, "OFF": 13
    }

    vibration_map = {
        'OFF': 0, 'vibration': 1, 'tile': 2, 'drop': 3
    }

    try:
        df = pd.read_csv(input_csv_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{input_csv_path}'. Please ensure the file exists.")
        return None

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
    df.to_csv(output_csv_path, index=False)

    print("Encoding OFF/ON to binary...")

    binary_map = {'OFF': 0, 'ON': 1, 'off': 0, 'on': 1}
    remaining_object_cols = df.select_dtypes(include=['object']).columns
    for col in remaining_object_cols:
        df[col] = df[col].map(binary_map)
    df.to_csv(output_csv_path)
    return df

if __name__ == "__main__":
    ROOT_DATA_DIR = "../data/SDHAR"
    SAVE_PATH = "../processed_data/"
    DAY_NAME = "Day_1"
    TIME_WINDOW = "2s"
    day_path = os.path.join(ROOT_DATA_DIR, DAY_NAME, DAY_NAME)

    all_data_sparse_df = load_data_and_activities(day_path)

    resampled_sparse_df = all_data_sparse_df.resample(TIME_WINDOW).last()

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    SPARSE_CSV_PATH = os.path.join(SAVE_PATH, f"sparse_data_{DAY_NAME}.csv")
    resampled_sparse_df.to_csv(SPARSE_CSV_PATH)


    DENSE_CSV_PATH = os.path.join(SAVE_PATH, f"dense_data_{DAY_NAME}.csv")
    densify_dataset(SPARSE_CSV_PATH, DENSE_CSV_PATH)

    RENAMED_COLUMNS_PATH = os.path.join(SAVE_PATH, f"renamed_columns_{DAY_NAME}.csv")
    rename_columns(DENSE_CSV_PATH, RENAMED_COLUMNS_PATH)

    ENCODED_DATA_PATH = os.path.join(SAVE_PATH, f"encoded_data_{DAY_NAME}.csv")
    unified_csv = encode_data(RENAMED_COLUMNS_PATH, ENCODED_DATA_PATH)




