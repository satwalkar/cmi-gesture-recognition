# data_utils.py

import polars as pl

# Import the project's configuration settings
import config

def get_columns_to_load():
    """
    Builds a list of columns to read from the main CSV based on the master
    switches set in the config.py file. This is for memory efficiency.
    """
    # Start with essential columns for grouping, labeling, and identification
    cols_to_load = ['sequence_id', 'subject', 'gesture', 'sequence_counter']

    # Conditionally add columns for each sensor group based on config flags
    if config.ENABLE_ACC_FEATURES:
        cols_to_load.extend(config.ACC_COLS)
    if config.ENABLE_ROT_FEATURES:
        cols_to_load.extend(config.ROT_COLS)
    if config.ENABLE_THM_FEATURES:
        cols_to_load.extend(config.THM_COLS)
    if config.ENABLE_TOF_FEATURES:
        cols_to_load.extend(config.TOF_COLS)

    print(f"Data Utils: {len(cols_to_load)} columns will be loaded based on the current configuration.")
    # Return a list of unique column names
    return list(set(cols_to_load))

def load_data(is_full_data=False, is_demographics=False, is_metadata=False):
    """
    Centralized function to load data from source CSVs based on specific needs.

    Args:
        is_full_data (bool): If True, loads sensor data based on config switches.
        is_demographics (bool): If True, loads only demographics data.
        is_metadata (bool): If True, loads only the minimal columns needed for setup.
    """
    if is_metadata:
        print("Loading minimal metadata (subject, sequence_id, gesture)...")
        return pl.read_csv(config.TRAIN_CSV_PATH, columns=['subject', 'sequence_id', 'gesture'])

    if is_demographics:
        # Only load demographics if the master switch is enabled
        if not config.ENABLE_DEMOGRAPHICS:
            print("Demographics loading is disabled in config. Skipping.")
            return None
        print("Loading demographics data...")
        return pl.read_csv(config.DEMOGRAPHICS_CSV_PATH)

    if is_full_data:
        # Get the precise list of columns to read from the CSV
        columns_to_read = get_columns_to_load()

        print(f"Loading sensor data from {config.TRAIN_CSV_PATH}...")
        df_train = pl.read_csv(config.TRAIN_CSV_PATH, columns=columns_to_read, ignore_errors=True)

        # Optionally join with demographics if the switch is on
        if config.ENABLE_DEMOGRAPHICS:
            print("Loading and merging demographics...")
            df_demo = pl.read_csv(config.DEMOGRAPHICS_CSV_PATH)
            df_merged = df_train.join(df_demo, on="subject", how="left")
            print(f"Data loaded and merged. Final shape: {df_merged.shape}")
            return df_merged
        else:
            print(f"Sensor data loaded. Final shape: {df_train.shape}")
            return df_train

    # Raise an error if the function was called without a valid option
    raise ValueError("A valid load option (e.g., is_full_data=True) must be specified.")