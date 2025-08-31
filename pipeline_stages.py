# pipeline_stages.py

import os
import gc
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from scipy.interpolate import interp1d
#from tensorflow.keras.optimizers.schedules import CosineDecay # For advanced LR schedules
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #, ReduceLROnPlateau, Callback
from tqdm import tqdm

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Import from our new modules
import config
import data_utils
import feature_engineering
import model_definition
import analysis_tools
# Import the main orchestrator and SPECIFIC custom layers needed for loading models
from model_definition import (
    InstanceNormalization,
    PositionalEmbedding,
    TransformerEncoder,
    SumOverTimeDimension,
    kt
)

# ===================================================================
# Data Augmentation Functions
# ===================================================================
# These helpers are used exclusively by the data generator below.
# They operate on NumPy arrays on-the-fly during training.

# --- 4. Advanced Data Augmentation (NumPy-based, for generator) ---
def time_warp(sequence: np.ndarray, sigma: float = 0.2, num_knots: int = 4) -> np.ndarray:
    if sequence.shape[0] < 2: return sequence
    time_steps = np.arange(sequence.shape[0])
    knot_x = np.linspace(0, sequence.shape[0] - 1, num=num_knots)
    knot_y = np.random.normal(loc=1.0, scale=sigma, size=(num_knots,))
    spline = interp1d(knot_x, knot_y, kind='cubic', fill_value='extrapolate')
    warped_time = np.cumsum(spline(time_steps))
    warped_sequence = np.zeros_like(sequence, dtype=np.float32)
    for i in range(sequence.shape[1]):
        interp_func = interp1d(np.arange(sequence.shape[0]), sequence[:, i], bounds_error=False, fill_value=0.0)
        warped_sequence[:, i] = interp_func(np.clip(warped_time, 0, sequence.shape[0] - 1))
    return warped_sequence

def magnitude_warp(sequence: np.ndarray, sigma: float = 0.2, num_knots: int = 4) -> np.ndarray:
    if sequence.shape[0] < 1: return sequence
    time_steps = np.arange(sequence.shape[0])
    knot_x = np.linspace(0, sequence.shape[0] - 1, num=num_knots)
    knot_y = np.random.normal(loc=1.0, scale=sigma, size=(num_knots,))
    spline = interp1d(knot_x, knot_y, kind='cubic', fill_value='extrapolate')
    warping_factor = spline(time_steps)
    return sequence * warping_factor[:, np.newaxis]

def jitter(sequence: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    return sequence + np.random.normal(loc=0., scale=sigma, size=sequence.shape).astype(np.float32)

# ===================================================================
# Data Generator for TensorFlow
# ===================================================================

# --- 5. Data Generator Function for tf.data.Dataset ---
def _data_generator_fn(sequence_ids_for_fold: np.ndarray, labels_for_fold: np.ndarray, sequence_data_cache: dict,
                       demographics_processed_all: pd.DataFrame, seq_to_subj_map: pd.Series,
                       is_training: bool, feature_cols_ordered: list[str], num_classes: int):
    """
    Helper function to yield (inputs, targets) for tf.data.Dataset.from_generator.
    This will process one sequence at a time. Batching will be handled by tf.data.
    """
    indexes = np.arange(len(sequence_ids_for_fold))
    if is_training:
        np.random.shuffle(indexes)

    for i in indexes:
        seq_id_val = sequence_ids_for_fold[i]
        label = labels_for_fold[i]

        # --- FIX: Decode the byte string key to a regular string ---
        #key = seq_id_val.decode('utf-8')

        # --- FIX: Decode the byte string key to a regular string ---
        if isinstance(seq_id_val, bytes):
            key = seq_id_val.decode('utf-8')
        else:
            key = str(seq_id_val)

        # Retrieve pre-cached data (already scaled)
        sequence = sequence_data_cache[key].copy()

        # Apply augmentation if training and enabled
        if is_training and config.ENABLE_DATA_AUGMENTATION:
            if np.random.rand() < 0.5: sequence = jitter(sequence)
            if np.random.rand() < 0.5: sequence = time_warp(sequence)
            if np.random.rand() < 0.5: sequence = magnitude_warp(sequence)

        # Padding/Truncation (always apply to ensure consistent shape)
        if sequence.shape[0] > config.MAX_SEQUENCE_LENGTH:
            sequence = sequence[:config.MAX_SEQUENCE_LENGTH]
        elif sequence.shape[0] < config.MAX_SEQUENCE_LENGTH:
            padding = np.zeros((config.MAX_SEQUENCE_LENGTH - sequence.shape[0], sequence.shape[1]), dtype=np.float32)
            sequence = np.vstack([sequence, padding])

        # Handle NaN/Inf after augmentation (if any)
        if not np.all(np.isfinite(sequence)):
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)

        # Get demographics data only if it's enabled
        if demographics_processed_all is not None:
            subject_id = seq_to_subj_map.loc[key]
            demographics_data = demographics_processed_all.loc[subject_id].values.astype('float32')
        else:
            # Otherwise, create an empty placeholder
            demographics_data = np.array([], dtype='float32')

        # --- START DEBUG BLOCK ---
        """
        if i < 2: # Print for the first 2 items to avoid spamming the console
            print("\n--- DEBUG: Generator Yield ---")
            print(f"Sequence ID: {seq_id_val}")
            # Check the time-series data
            ts_data = sequence
            print(f"  Time Series Type: {type(ts_data)}, Dtype: {ts_data.dtype}, Shape: {ts_data.shape}")
            # Check the demographics data
            demo_data = demographics_data
            print(f"  Demographics Type: {type(demo_data)}, Dtype: {demo_data.dtype}, Shape: {demo_data.shape}")
            # Check the label data
            label_data = label
            print(f"  Label Type: {type(label_data)}, Dtype: {label_data.dtype}, Shape: {label_data.shape}")
            print("--- END DEBUG BLOCK ---\n")
        """
        # --- END DEBUG BLOCK ---

        yield {'time_series_input': sequence, 'demographics_input': demographics_data}, label

# ===================================================================
# Main Pipeline Stage Functions
# ===================================================================

def run_setup_stage():
    """
    Handles global setup: loading metadata, processing demographics, and creating encoders.
    This stage is checkpointed.
    """

    print("\n" + "="*80 + "\n--- STAGE 1: Global Setup & Preprocessing ---\n" + "="*80)
    artifacts = {

        "seq_info": os.path.join(config.MODEL_SAVE_DIR, "sequence_info.pkl"),
        "demo_proc": os.path.join(config.MODEL_SAVE_DIR, "demographics_processed.pkl"),
        "demo_scaler": os.path.join(config.MODEL_SAVE_DIR, "demo_scaler.pkl"),
        "demo_ohe": os.path.join(config.MODEL_SAVE_DIR, "demo_ohe.pkl"),
        "label_enc": os.path.join(config.MODEL_SAVE_DIR, "label_encoder.pkl")
    }

    if all(os.path.exists(p) for p in artifacts.values()) and not config.FORCE_RERUN_SETUP:
        print("Setup artifacts found. Loading from disk.")
        sequence_info_all_pd = joblib.load(artifacts["seq_info"])
        demographics_processed_all = joblib.load(artifacts["demo_proc"])
        global_gesture_label_encoder = joblib.load(artifacts["label_enc"])
        return sequence_info_all_pd, demographics_processed_all, global_gesture_label_encoder

    print("Running setup and preprocessing from scratch...")

    # --- Prepare overall sequence_info and LabelEncoder ---
    # This is done once on the full training set to ensure consistent labeling across folds.
    print("Loading metadata for setup (labels, groups)...")
    # --- MODIFICATION: Using the centralized load_data function ---
    df_for_seq_info = data_utils.load_data(is_metadata=True)
    sequence_info_all_pd = df_for_seq_info.unique().to_pandas().set_index('sequence_id')

    global_gesture_label_encoder = LabelEncoder().fit(list(sequence_info_all_pd['gesture'].unique()))
    sequence_info_all_pd['gesture_encoded'] = global_gesture_label_encoder.transform(sequence_info_all_pd['gesture'])

    # Initialize a placeholder for the processed demographics
    demographics_processed_all = None
    demo_scaler = None
    demo_ohe = None

    # Only process demographics if the switch is enabled in the config
    if config.ENABLE_DEMOGRAPHICS:
        print("Processing enabled demographics data...")
        df_for_demo_fit = data_utils.load_data(is_demographics=True)

        # Also check that data was actually loaded
        if df_for_demo_fit is not None and df_for_demo_fit.height > 0:
            demographics_for_scaler_fit = df_for_demo_fit.unique(subset=['subject']).to_pandas()
            demographics_for_scaler_fit.columns = demographics_for_scaler_fit.columns.str.strip()

            # Fit OHE and Scaler on the entire training demographics
            demo_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(demographics_for_scaler_fit[config.DEMO_CATEGORICAL_COLS])
            demo_scaler = StandardScaler().fit(demographics_for_scaler_fit[config.DEMO_NUMERICAL_COLS])

            # Transform all demographics data using the fitted scalers
            encoded_cats_demo = demo_ohe.transform(demographics_for_scaler_fit[config.DEMO_CATEGORICAL_COLS])
            scaled_nums_demo = demo_scaler.transform(demographics_for_scaler_fit[config.DEMO_NUMERICAL_COLS])

            demographics_processed_all = pd.DataFrame(
                np.hstack([encoded_cats_demo, scaled_nums_demo]),
                index=demographics_for_scaler_fit['subject'],
                columns=demo_ohe.get_feature_names_out(config.DEMO_CATEGORICAL_COLS).tolist() + config.DEMO_NUMERICAL_COLS
            )
    else:
        print("Skipping demographics processing as per config.")

    print("Saving setup artifacts...")
    joblib.dump(sequence_info_all_pd, artifacts["seq_info"])
    joblib.dump(demographics_processed_all, artifacts["demo_proc"])
    joblib.dump(demo_scaler, artifacts["demo_scaler"])
    joblib.dump(demo_ohe, artifacts["demo_ohe"])
    joblib.dump(global_gesture_label_encoder, artifacts["label_enc"])

    print("Setup stage complete.")
    return sequence_info_all_pd, demographics_processed_all, global_gesture_label_encoder

def _get_current_feature_config():
    """
    Precisely collects feature-relevant flags by reading the list of keys
    defined in the config.py file itself.
    """
    config_dict = {}
    # Iterate over the list of keys imported from the config module
    for key in config.FEATURE_CONFIG_KEYS:
        if hasattr(config, key):
            config_dict[key] = getattr(config, key)

    return config_dict

""" WORKING REVISION BUT ALL or NONE approach
def run_feature_engineering_stage():

    #Orchestrates the entire feature engineering process by checking for existing
    #features and running parallel processing if they need to be regenerated.

    print("\n" + "="*80)
    print("--- STAGE 2: Feature Engineering ---")
    print("="*80)

    feature_output_dir = config.PERMANENT_FEATURE_DIR
    config_file_path = config.CONFIG_FILE_PATH
    current_config = _get_current_feature_config()
    force_regeneration = config.FORCE_RERUN_FEATURE_ENGINEERING

    # Check for the existence of ALL required feature artifacts before skipping
    all_artifacts_exist = (
        os.path.exists(config.PERMANENT_FEATURE_DIR) and
        os.path.exists(config.CONFIG_FILE_PATH) and
        os.path.exists(config.FEATURE_NAMES_FILE_PATH)
    )

    # 1. Decide if regeneration is necessary
    if not force_regeneration and all_artifacts_exist:
        with open(config_file_path, 'r') as f:
            saved_config = json.load(f)

        if saved_config == current_config:
            print("✅ Existing features match current configuration. Skipping generation.")
            # Sanity check to ensure files actually exist
            if not any(f.endswith('.parquet') for f in os.listdir(feature_output_dir)):
                 print("⚠️ Warning: Feature directory is empty. Forcing regeneration.")
                 force_regeneration = True
            else:
                return # Exit the function as no work is needed
        else:
            print("⚙️ Configuration has changed. Forcing feature regeneration.")
            force_regeneration = True
    else:
        print("🚀 No valid existing features found. Starting feature generation.")
        force_regeneration = True

    # 2. Execute feature generation if required
    if force_regeneration:
        # Clean up old features first
        if os.path.exists(feature_output_dir):
            shutil.rmtree(feature_output_dir)
        os.makedirs(feature_output_dir)

        print("Loading raw data for feature engineering...")
        df_pl = data_utils.load_data(is_full_data=True)

        if df_pl is None or df_pl.height == 0:
            raise ValueError("Data loading returned an empty DataFrame. Cannot proceed.")

        # Pre-fill missing sensor data to prevent errors in feature calculations
        print("Applying forward/backward fill to raw sensor data...")
        for col in config.ALL_RAW_SENSOR_COLS:
            if col in df_pl.columns:
                df_pl = df_pl.with_columns(pl.col(col).fill_null(strategy="forward").over("sequence_id"))
                df_pl = df_pl.with_columns(pl.col(col).fill_null(strategy="backward").over("sequence_id"))
        df_pl = df_pl.fill_null(0.0)

        # Use a temporary directory for safety during parallel processing
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory for processing: {temp_dir}")

        try:
            unique_sequence_ids = df_pl['sequence_id'].unique().to_list()
            print(f"Processing {len(unique_sequence_ids)} unique sequences in parallel...")

            # Call the low-level worker from the feature_engineering module
            joblib.Parallel(n_jobs=1, verbose=0)(
                joblib.delayed(feature_engineering._process_single_sequence_for_fe)(
                    df_pl.filter(pl.col('sequence_id') == seq_id),
                    temp_dir
                ) for seq_id in tqdm(unique_sequence_ids, desc="Generating Features")
            )

            # 3. Finalize and Save Artifacts
            # Move processed files from temp to permanent directory
            print("Moving processed files to permanent feature directory...")
            for f in os.listdir(temp_dir):
                shutil.move(os.path.join(temp_dir, f), feature_output_dir)

            # Save the configuration that generated these features
            with open(config_file_path, 'w') as f:
                json.dump(current_config, f, indent=4)
            print(f"Feature configuration saved to: {config_file_path}")

            # Save the final list of feature names for the model to use
            # We get the schema from a sample file instead of loading all data

            # Get a list of ONLY the parquet files in the directory
            parquet_files = [f for f in os.listdir(feature_output_dir) if f.endswith('.parquet')]
            if not parquet_files:
                raise FileNotFoundError("No Parquet feature files found to read the schema from!")

            # Select the first valid parquet file as the sample
            sample_file = os.path.join(feature_output_dir, parquet_files[0])
            sample_df = pl.read_parquet(sample_file)

            feature_names = feature_engineering.get_all_feature_columns(sample_df)
            with open(config.FEATURE_NAMES_FILE_PATH, 'w') as f:
                json.dump(feature_names, f, indent=4)
            print(f"List of {len(feature_names)} feature names saved.")

        finally:
            # Clean up the temporary directory and memory
            shutil.rmtree(temp_dir)
            del df_pl
            gc.collect()

    print("\nFeature engineering stage complete.")
"""

def run_feature_engineering_stage(sequence_info_all_pd):
    """
    Orchestrates a robust and resumable feature engineering process.
    - Forces a full, clean regeneration if the config.py file has changed. and running parallel processing if they need to be regenerated.
    - Resumes an interrupted run by only processing missing files if the config is the same.
    - Guarantees a full, clean regeneration if the config.py file has changed.
    - Safely resumes an interrupted run ONLY if the config can be verified.
    - Final, robust, and resumable feature engineering. Guarantees a full, clean regeneration if the config changes or if a previous run is in an ambiguous, interrupted state.
    """
    print("\n" + "="*80 + "\n--- STAGE 2: Feature Engineering (Robust & Resumable) ---\n" + "="*80)

    feature_output_dir = config.PERMANENT_FEATURE_DIR
    config_file_path = config.CONFIG_FILE_PATH
    current_config = _get_current_feature_config()
    force_regeneration = config.FORCE_RERUN_FEATURE_ENGINEERING

    # --- 1. Determine the state of the feature set ---
    saved_config = None
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            saved_config = json.load(f)

    # Check for any existing parquet files
    os.makedirs(feature_output_dir, exist_ok=True)
    parquet_files_exist = any(f.endswith('.parquet') for f in os.listdir(feature_output_dir))

    # --- 2. Decide if a full, clean rerun is required ---
    # A full rerun is needed if:
    #   A) The user forces it.
    #   B) A config file exists, but it doesn't match the current config.
    #   C) No config file exists, BUT some parquet files exist (the ambiguous interrupted state).

    if force_regeneration or \
       (saved_config is not None and saved_config != current_config) or \
       (saved_config is None and parquet_files_exist):

        if force_regeneration:
            print("⚙️ Rerun is forced. Starting a fresh run.")
        elif saved_config is not None and saved_config != current_config:
            print("⚙️ Configuration has changed. Starting a fresh run.")
        else:
            print("⚙️ Ambiguous state detected (partial files exist without a config file). Starting a fresh run to ensure data integrity.")

        if os.path.exists(feature_output_dir):
            shutil.rmtree(feature_output_dir)
        os.makedirs(feature_output_dir)

    # --- 3. Proceed with resumable logic on what is now a clean or valid resumable directory ---
    all_required_ids = set(sequence_info_all_pd.index.to_list())

    existing_files = [f for f in os.listdir(feature_output_dir) if f.endswith('.parquet')]
    existing_ids = set(f.removeprefix('seq_').removesuffix('_features.parquet') for f in existing_files)
    missing_ids = list(all_required_ids - existing_ids)

    if not missing_ids:
        print("✅ All feature files already exist and are up-to-date.")
    else:
        print(f"Found {len(existing_ids)} existing files. Processing {len(missing_ids)} missing sequences...")
        ids_to_process = missing_ids

        if ids_to_process:
            df_pl = data_utils.load_data(is_full_data=True)

            if df_pl is None or df_pl.height == 0:
                raise ValueError("Data loading returned an empty DataFrame. Cannot proceed.")

            # Pre-fill missing sensor data to prevent errors in feature calculations
            print("Applying forward/backward fill to raw sensor data...")
            for col in config.ALL_RAW_SENSOR_COLS:
                if col in df_pl.columns:
                    df_pl = df_pl.with_columns(pl.col(col).fill_null(strategy="forward").over("sequence_id"))
                    df_pl = df_pl.with_columns(pl.col(col).fill_null(strategy="backward").over("sequence_id"))
            df_pl = df_pl.fill_null(0.0)

            # Ensure we only process IDs that are actually present in the raw data
            valid_ids_in_data = set(df_pl['sequence_id'].unique().to_list())
            final_ids_to_process = [id for id in ids_to_process if id in valid_ids_in_data]

            if len(final_ids_to_process) != len(ids_to_process):
                print(f"Warning: {len(ids_to_process) - len(final_ids_to_process)} sequence IDs from metadata were not found in the data file and will be skipped.")

            joblib.Parallel(n_jobs=-1, verbose=0)(
                joblib.delayed(feature_engineering._process_single_sequence_for_fe)(
                    df_pl.filter(pl.col('sequence_id') == seq_id),
                    feature_output_dir
                ) for seq_id in tqdm(final_ids_to_process, desc="Generating Missing Features")
            )
        del df_pl
        gc.collect()

    # 5. Finalize and save metadata only if ALL files are now present
    final_files_count = len([f for f in os.listdir(feature_output_dir) if f.endswith('.parquet')])
    if final_files_count == len(all_required_ids):
        print("All features generated successfully. Saving metadata files.")
        with open(config_file_path, 'w') as f:
            json.dump(current_config, f, indent=4)

        # First, get a list of ONLY the parquet files
        parquet_files = [f for f in os.listdir(feature_output_dir) if f.endswith('.parquet')]
        if not parquet_files:
            raise FileNotFoundError("No Parquet files found to read the schema from!")

        # ... (code to get schema from a sample file and save feature_names.json) ...
        # Get schema from a sample file to create feature_names.json
        # Then, select the first file from that safe list

        sample_file = os.path.join(feature_output_dir, parquet_files[0])
        sample_df = pl.read_parquet(sample_file)

        feature_names = feature_engineering.get_all_feature_columns(sample_df)
        with open(config.FEATURE_NAMES_FILE_PATH, 'w') as f:
            json.dump(feature_names, f, indent=4)
        print(f"List of {len(feature_names)} feature names saved.")
    else:
        print(f"⚠️ Warning: Process still incomplete. {len(all_required_ids) - final_files_count} files still missing.")

    print("\nFeature engineering stage complete.")

# --- NEW FUNCTION TO RESTORE INTER-SEQUENCE FEATURES ---

# Add tqdm to your imports at the top of the file
def run_inter_sequence_feature_stage(sequence_info_all_pd):
    """
    Adds inter-sequence features by processing subjects in memory-efficient chunks. This version has a smart checkpoint that
    detects if the underlying features have changed.
    """
    if not config.ENABLE_INTER_SEQUENCE_FEATURES:
        print("Skipping inter-sequence feature generation as per config.")
        return

    print("\n" + "="*80 + "\n--- STAGE 3: Inter-Sequence Feature Generation ---\n" + "="*80)

    # Define paths for the main feature config and this stage's own config "fingerprint"
    main_feature_config_path = config.CONFIG_FILE_PATH
    inter_seq_config_path = os.path.join(config.MODEL_SAVE_DIR, 'inter_sequence_config.json')

    # --- Smarter Checkpoint Logic ---
    # Check if the main features have changed since this stage last ran successfully.
    if os.path.exists(inter_seq_config_path) and os.path.exists(main_feature_config_path):
        with open(inter_seq_config_path, 'r') as f:
            last_run_config = json.load(f)
        with open(main_feature_config_path, 'r') as f:
            current_feature_config = json.load(f)

        # If the underlying features are the same and we're not forcing a rerun, skip.
        if last_run_config == current_feature_config and not config.FORCE_RERUN_INTER_SEQUENCE:
            print("✅ Inter-sequence features are up-to-date with current features. Skipping.")
            return

    # 1. Get all unique subject IDs
    all_subjects = sequence_info_all_pd['subject'].unique().tolist()
    chunk_size = 20  # Process 20 subjects at a time (can be tuned)
    all_subject_stats = []

    # 2. Start a loop to iterate through the subjects in chunks
    for i in tqdm(range(0, len(all_subjects), chunk_size), desc="Processing Subject Chunks"):
        # 3a. Get the current chunk of subjects
        subject_chunk = all_subjects[i:i + chunk_size]

        # 3b. Find all sequences that belong to these subjects
        sequences_for_chunk = sequence_info_all_pd[
            sequence_info_all_pd['subject'].isin(subject_chunk)
        ].index.to_list()

        if not sequences_for_chunk:
            continue

        # 3c. Load only their Parquet files
        chunk_files = [os.path.join(config.PERMANENT_FEATURE_DIR, f"seq_{sid}_features.parquet") for sid in sequences_for_chunk]
        df_chunk = pl.read_parquet(chunk_files)

        # 3d. Apply the feature logic (this part is the same as before)
        static_feature_cols = [col for col in df_chunk.columns if '_mean' in col or '_std' in col]

        subject_stats_chunk = df_chunk.group_by('subject').agg(
            [pl.col(col).mean().alias(f"{col}_subject_mean") for col in static_feature_cols] +
            [pl.col(col).std().alias(f"{col}_subject_std").fill_null(1e-9) for col in static_feature_cols]
        )
        all_subject_stats.append(subject_stats_chunk)

        df_chunk_enriched = df_chunk.join(subject_stats_chunk, on='subject', how='left')

        deviation_expressions = [
            (pl.col(col) - pl.col(f"{col}_subject_mean")).alias(f'{col}_dev_from_subj_mean')
            for col in static_feature_cols
        ] + [
            (pl.col(col) / pl.col(f"{col}_subject_std")).alias(f'{col}_dev_from_subj_std')
            for col in static_feature_cols
        ]

        df_chunk_enriched = df_chunk_enriched.with_columns(deviation_expressions)
        df_chunk_enriched = df_chunk_enriched.with_columns(
            pl.col('sequence_counter').rank(method='dense').over('subject').alias('sequence_position_in_subject')
        )

        cols_to_drop = [col for col in df_chunk_enriched.columns if col.endswith('_subject_mean') or col.endswith('_subject_std')]
        df_chunk_final = df_chunk_enriched.drop(cols_to_drop).fill_nan(0.0).fill_null(0.0)

        # 3e. Save the updated files for this chunk by overwriting them
        for seq_id, group_df in df_chunk_final.group_by("sequence_id"):
            output_path = os.path.join(config.PERMANENT_FEATURE_DIR, f"seq_{seq_id[0]}_features.parquet")
            group_df.write_parquet(output_path)

        # 3f. Clean up memory before the next chunk
        del df_chunk, df_chunk_enriched, df_chunk_final, subject_stats_chunk
        gc.collect()

    # --- Finalization: Save the new config "fingerprint" ---
    # After the loop, combine the stats from all chunks and save the final file
    # This only runs after a successful completion of the stage.
    if all_subject_stats:
        final_subject_stats = pl.concat(all_subject_stats)
        joblib.dump(final_subject_stats, config.SUBJECT_STATS_PATH)
        print(f"✅ Subject-level statistics saved to {config.SUBJECT_STATS_PATH}")

        # Save a copy of the main feature config as this stage's fingerprint
        shutil.copy(main_feature_config_path, inter_seq_config_path)
        print("Saved inter-sequence config fingerprint.")

    print("Inter-sequence feature stage complete.")

def _create_tf_datasets(train_ids, val_ids, data_cache, label_map, seq_to_subj_map, demographics_processed_all, feature_columns, num_classes, output_signature):
    """
    Creates and prepares training and validation tf.data.Dataset objects.
    This version correctly flattens/reconstructs data for tf.py_function.
    """

    # 1. Create a lightweight generator that just yields IDs and labels
    def id_generator(ids):
        for seq_id in ids:
            yield (seq_id, label_map[seq_id])

    # 2. Create a processing function that returns a FLAT TUPLE
    def _process_data(seq_id_tensor, label):
        seq_id = seq_id_tensor.numpy().decode('utf-8')
        sequence = data_cache[seq_id].copy()

        # Data Augmentation
        if config.ENABLE_DATA_AUGMENTATION and np.random.rand() < 0.5:
             sequence = jitter(sequence) # Simplified augmentation for clarity

        # Padding/Truncation
        if sequence.shape[0] > config.MAX_SEQUENCE_LENGTH:
            sequence = sequence[:config.MAX_SEQUENCE_LENGTH]
        elif sequence.shape[0] < config.MAX_SEQUENCE_LENGTH:
            padding = np.zeros((config.MAX_SEQUENCE_LENGTH - sequence.shape[0], sequence.shape[1]), dtype=np.float32)
            sequence = np.vstack([sequence, padding])

        # Demographics
        if demographics_processed_all is not None:
            subject_id = seq_to_subj_map.loc[seq_id]
            demographics_data = demographics_processed_all.loc[subject_id].values.astype('float32')
        else:
            demographics_data = np.array([], dtype='float32')

        # Return a flat tuple of numpy arrays
        return sequence, demographics_data, label

    # 3. Wrap the processing function in tf.py_function and RECONSTRUCT the dictionary
    def _tf_process_data(seq_id, label):
        # The Tout argument must now be a flat list of data types
        sequence, demographics, label_out = tf.py_function(
            _process_data,
            inp=[seq_id, label],
            Tout=[tf.float32, tf.float32, tf.float32]
        )

        # Reconstruct the dictionary structure the model expects
        inputs = {
            'time_series_input': sequence,
            'demographics_input': demographics
        }

        # Set shapes manually after the py_function
        inputs['time_series_input'].set_shape(output_signature[0]['time_series_input'].shape)
        inputs['demographics_input'].set_shape(output_signature[0]['demographics_input'].shape)
        label_out.set_shape(output_signature[1].shape)
        return inputs, label_out

    # 4. Build the efficient tf.data pipelines (this part remains the same)
    id_generator_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
    )
    train_dataset = tf.data.Dataset.from_generator(lambda: id_generator(train_ids), output_signature=id_generator_signature)
    val_dataset = tf.data.Dataset.from_generator(lambda: id_generator(val_ids), output_signature=id_generator_signature)

    train_dataset = train_dataset.map(_tf_process_data, num_parallel_calls=config.WORKERS_AUTOTUNE)
    val_dataset = val_dataset.map(_tf_process_data, num_parallel_calls=config.WORKERS_AUTOTUNE)

    train_dataset = train_dataset.batch(config.GLOBAL_BATCH_SIZE).repeat().prefetch(config.WORKERS_AUTOTUNE)
    val_dataset = val_dataset.batch(config.GLOBAL_BATCH_SIZE).repeat().prefetch(config.WORKERS_AUTOTUNE)

    steps_per_epoch = (len(train_ids) + config.GLOBAL_BATCH_SIZE - 1) // config.GLOBAL_BATCH_SIZE
    validation_steps = (len(val_ids) + config.GLOBAL_BATCH_SIZE - 1) // config.GLOBAL_BATCH_SIZE

    return train_dataset, val_dataset, steps_per_epoch, validation_steps, np.array([label_map[sid] for sid in train_ids])

"""
# WORKING REVISION WITH SINGLE THREAD?

def _create_tf_datasets(train_ids, val_ids, data_cache, label_map, seq_to_subj_map, demographics_processed_all, feature_columns, num_classes, output_signature):
    # Creates and prepares training and validation tf.data.Dataset objects from IDs.

    # Prepare labels for the current split
    train_labels = np.array([label_map[sid] for sid in train_ids])
    val_labels = np.array([label_map[sid] for sid in val_ids])

    # Create partial generator functions for this split
    train_gen_fn = partial(
        _data_generator_fn,
        sequence_ids_for_fold=train_ids,
        labels_for_fold=train_labels,
        sequence_data_cache=data_cache,
        demographics_processed_all=demographics_processed_all,
        seq_to_subj_map=seq_to_subj_map,
        is_training=True,
        feature_cols_ordered=feature_columns,
        num_classes=num_classes
    )
    val_gen_fn = partial(
        _data_generator_fn,
        sequence_ids_for_fold=val_ids,
        labels_for_fold=val_labels,
        sequence_data_cache=data_cache,
        demographics_processed_all=demographics_processed_all,
        seq_to_subj_map=seq_to_subj_map,
        is_training=False,
        feature_cols_ordered=feature_columns,
        num_classes=num_classes
    )

    # Create the tf.data.Dataset objects
    train_dataset = tf.data.Dataset.from_generator(train_gen_fn, output_signature=output_signature)
    val_dataset = tf.data.Dataset.from_generator(val_gen_fn, output_signature=output_signature)

    # Apply batching and prefetching for performance
    train_dataset = train_dataset.batch(config.GLOBAL_BATCH_SIZE).repeat().prefetch(config.WORKERS_AUTOTUNE)
    val_dataset = val_dataset.batch(config.GLOBAL_BATCH_SIZE).repeat().prefetch(config.WORKERS_AUTOTUNE)

    # Calculate steps for this split
    steps_per_epoch = (len(train_ids) + config.GLOBAL_BATCH_SIZE - 1) // config.GLOBAL_BATCH_SIZE
    validation_steps = (len(val_ids) + config.GLOBAL_BATCH_SIZE - 1) // config.GLOBAL_BATCH_SIZE

    return train_dataset, val_dataset, steps_per_epoch, validation_steps, train_labels
"""

def run_training_stage(sequence_info_all_pd, demographics_processed_all, global_gesture_label_encoder, strategy):
    """Runs the K-Fold training loop with memory-efficient data loading."""
    print("\n" + "="*80)
    print("--- STAGE 4: Model Training ---")
    print("="*80)

    # Pre-training setup
    num_classes = len(global_gesture_label_encoder.classes_)
    labels_categorical = to_categorical(sequence_info_all_pd['gesture_encoded'], num_classes=num_classes)
    label_map = {sid: labels_categorical[idx] for idx, sid in enumerate(sequence_info_all_pd.index)}
    seq_to_subj_map = sequence_info_all_pd['subject']
    # If demographics are disabled, the shape is 0. Otherwise, get it from the dataframe.
    demo_shape = (demographics_processed_all.shape[1],) if demographics_processed_all is not None else (0,)

    # Conditionally load the feature list
    feature_list_path = config.FEATURE_NAMES_FILE_PATH
    if config.ENABLE_FEATURE_SELECTION:
        selected_features_path = os.path.join(config.MODEL_SAVE_DIR, 'selected_feature_names.json')
        if os.path.exists(selected_features_path):
            feature_list_path = selected_features_path
            print(f"Using selected feature set from: {feature_list_path}")

    # Load feature names and create shape tuple
    with open(feature_list_path, 'r') as f:
        feature_columns = json.load(f)
    ts_shape = (config.MAX_SEQUENCE_LENGTH, len(feature_columns))

    output_signature = (
        {'time_series_input': tf.TensorSpec(shape=ts_shape, dtype=tf.float32),
         'demographics_input': tf.TensorSpec(shape=demo_shape, dtype=tf.float32)},
        tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
    )

    # ==========================================================
    # PATH 1: HYPERPARAMETER OPTIMIZATION
    # ==========================================================
    if config.HPO_ENABLED:
        if not model_definition._KERAS_TUNER_AVAILABLE:
            print("❌ Keras Tuner not installed. Cannot run HPO. Aborting.")
            return

        print("🚀 Starting Hyperparameter Optimization (HPO) search...")

        # HPO typically uses a single validation split from the full dataset
        # 1. Create a single, stratified train/validation split for the search
        # We stratify to ensure the validation set has a similar class distribution

        hpo_train_ids, hpo_val_ids = train_test_split(
            sequence_info_all_pd.index,
            test_size=0.2, # e.g., 20% for validation
            random_state=config.RANDOM_STATE,
            stratify=sequence_info_all_pd['gesture_encoded']
        )
        print(f"HPO split: {len(hpo_train_ids)} training sequences, {len(hpo_val_ids)} validation sequences.")

        print("\n➡️ Step 1/4: Loading training set Parquet files...")
        train_files = [os.path.join(config.PERMANENT_FEATURE_DIR, f"seq_{sid}_features.parquet") for sid in hpo_train_ids]
        train_data_pl = pl.read_parquet(train_files)

        print("➡️ Step 2/4: Loading validation set Parquet files...")
        val_files = [os.path.join(config.PERMANENT_FEATURE_DIR, f"seq_{sid}_features.parquet") for sid in hpo_val_ids]
        val_data_pl = pl.read_parquet(val_files)

        print("➡️ Step 3/4: Fitting the data scaler...")
        ts_scaler = StandardScaler().fit(train_data_pl.select(feature_columns).to_numpy())

        print("➡️ Step 4/4: Building in-memory data cache...")
        combined_data_pl = pl.concat([train_data_pl, val_data_pl])
        data_cache = {
            sid[0]: ts_scaler.transform(g.select(feature_columns).to_numpy(allow_copy=True)).astype(np.float32)
            for sid, g in combined_data_pl.group_by("sequence_id")
        }

        # --- Add this block for feedback ---
        # Calculate the total memory size of the numpy arrays in the cache
        cache_size_mb = sum(v.nbytes for v in data_cache.values()) / (1024 * 1024)
        print(f"✅ Data cache created successfully for {len(data_cache)} sequences.")
        print(f"   - Estimated cache size in RAM: {cache_size_mb:.2f} MB")

        print("✅ Data preparation complete. Initializing Keras Tuner.")

        # 2. Load only the necessary data for this split
        train_files = [os.path.join(config.PERMANENT_FEATURE_DIR, f"seq_{sid}_features.parquet") for sid in hpo_train_ids]
        val_files = [os.path.join(config.PERMANENT_FEATURE_DIR, f"seq_{sid}_features.parquet") for sid in hpo_val_ids]
        train_data_pl = pl.read_parquet(train_files)
        val_data_pl = pl.read_parquet(val_files)

        # 3. Prepare scaler and data cache (fit scaler ONLY on HPO training data)
        ts_scaler = StandardScaler().fit(train_data_pl.select(feature_columns).to_numpy())
        combined_data_pl = pl.concat([train_data_pl, val_data_pl])
        data_cache = {
            sid[0]: ts_scaler.transform(g.select(feature_columns).to_numpy()).astype(np.float32)
            for sid, g in combined_data_pl.group_by("sequence_id")
        }

        # Create datasets using the new helper function
        train_dataset, val_dataset, steps_per_epoch, validation_steps, _ = _create_tf_datasets(
            hpo_train_ids, hpo_val_ids, data_cache, label_map, seq_to_subj_map, demographics_processed_all, feature_columns, num_classes, output_signature
        )

        # Create a new function on-the-fly that has the correct signature for the tuner
        model_builder_for_tuner = lambda hp: model_definition.model_builder(
            hp,
            ts_shape=ts_shape,
            demo_shape=demo_shape,
            num_classes=num_classes
        )
        """
        # Initialize and run the tuner
        with strategy.scope():
            tuner = kt.Hyperband(
                model_builder_for_tuner,
                objective='val_accuracy', max_epochs=50, factor=3,
                directory=config.MODEL_SAVE_DIR, project_name='hpo_search', overwrite=True
            )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        tuner.search(train_dataset, validation_data=val_dataset, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=[stop_early])
        """

        #with strategy.scope():
        tuner = kt.RandomSearch(
            #model_definition.model_builder,
            model_builder_for_tuner,
            objective='val_accuracy',
            distribution_strategy=strategy,
            max_trials=config.HPO_NUM_TRIALS,  # <-- NOW THE VARIABLE IS USED
            executions_per_trial=1,            # Train each combination once
            directory=config.MODEL_SAVE_DIR,
            project_name='hpo_search',
            overwrite=True
        )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        # Each trial will now run for up to 100 epochs, stopping early if needed
        tuner.search(
            train_dataset,
            epochs=100,
            validation_data=val_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[stop_early]
        )

        print("\n--- Extracting and Saving Optimal Hyperparameters ---")
        # 1. Get the best TRIAL object, which contains both the score and the HPs
        best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
        
        # 2. Get the hyperparameters dictionary from the trial
        best_hps_dict = best_trial.hyperparameters.values
        
        # 3. Get the best score and add it to the dictionary
        best_score = best_trial.score
        best_hps_dict['best_val_accuracy'] = best_score

        # 4. Save the complete dictionary to a JSON file
        hpo_results_path = os.path.join(config.MODEL_SAVE_DIR, "best_hpo_results.json")
        with open(hpo_results_path, 'w') as f:
            json.dump(best_hps_dict, f, indent=4)
        
        print(f"✅ Optimal hyperparameters and best score ({best_score:.4f}) saved to: {hpo_results_path}")

        # 5. Print the summary to the console
        print("\n--- Optimal Hyperparameters Found ---")
        for param, value in best_hps_dict.items():
            print(f"{param}: {value}")
        print("------------------------------------")

        # 6. Build and save the single best model from the HPO run
        best_model = tuner.hypermodel.build(best_trial.hyperparameters)
        best_model.save(os.path.join(config.MODEL_SAVE_DIR, "best_hpo_model.keras"))
        print("\n✅ Best model from HPO search saved.")
    # ==========================================================
    # PATH 2: STANDARD K-FOLD TRAINING
    # ==========================================================
    else:
        print("🏃 Starting standard K-Fold training with fixed hyperparameters...")
        sgkf = StratifiedGroupKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(sequence_info_all_pd, sequence_info_all_pd['gesture_encoded'], groups=sequence_info_all_pd['subject'])):

            model_path = os.path.join(config.MODEL_SAVE_DIR, f"main_model_fold_{fold+1}.keras")
            train_config_path = os.path.join(config.MODEL_SAVE_DIR, f"training_config_fold_{fold+1}.json")

            # --- NEW SMART CHECKPOINT ---
            hp_placeholder = model_definition.kt.HyperParameters()
            # (Your hp_placeholder.Fixed(...) lines go here to define the current model config)
            current_model_config = hp_placeholder.values

            if os.path.exists(train_config_path) and not config.FORCE_RERUN_TRAINING:
                with open(train_config_path, 'r') as f:
                    saved_model_config = json.load(f)
                if saved_model_config == current_model_config:
                    print(f"✅ Model for fold {fold+1} is up-to-date. Skipping training.")
                    continue
            # --------------------------

            print(f"\n--- Processing Fold {fold+1}/{config.N_SPLITS} ---")

            # 1. Get sequence IDs for the current fold
            train_ids = sequence_info_all_pd.index[train_idx].to_list()
            val_ids = sequence_info_all_pd.index[val_idx].to_list()

            # 2. Load data and prepare cache for this specific fold
            train_files = [os.path.join(config.PERMANENT_FEATURE_DIR, f"seq_{sid}_features.parquet") for sid in train_ids]
            val_files = [os.path.join(config.PERMANENT_FEATURE_DIR, f"seq_{sid}_features.parquet") for sid in val_ids]
            train_data_pl = pl.read_parquet(train_files)
            val_data_pl = pl.read_parquet(val_files)

            ts_scaler = StandardScaler().fit(train_data_pl.select(feature_columns).to_numpy())
            combined_data_pl = pl.concat([train_data_pl, val_data_pl])
            data_cache = {
                sid[0]: ts_scaler.transform(g.select(feature_columns).to_numpy()).astype(np.float32)
                for sid, g in combined_data_pl.group_by("sequence_id")
            }

            # 3. Create datasets for this fold using the helper function
            train_dataset, val_dataset, steps_per_epoch, validation_steps, train_labels = _create_tf_datasets(
                train_ids, val_ids, data_cache, label_map, seq_to_subj_map, demographics_processed_all, feature_columns, num_classes, output_signature
            )

            # 4. Calculate class weights for this fold's training data
            class_weights_dict = dict(enumerate(compute_class_weight(
                class_weight='balanced',
                classes=np.arange(num_classes),
                y=np.argmax(train_labels, axis=1)
            )))

            # 5. Build and compile the model for this fold
            model_path = os.path.join(config.MODEL_SAVE_DIR, f"main_model_fold_{fold+1}.keras")
            with strategy.scope():
                # Use a placeholder HP object to lock in the winning HPO recipe
                hp_placeholder = model_definition.kt.HyperParameters()
                hp_placeholder.Fixed('l2_reg', 0.0001)
                hp_placeholder.Fixed('conv1d_filters_1', 64)
                hp_placeholder.Fixed('conv1d_kernel_1', 5)
                hp_placeholder.Fixed('conv1d_filters_2', 64)
                hp_placeholder.Fixed('conv1d_kernel_2', 5)
                hp_placeholder.Fixed('transformer_num_heads', 4)
                hp_placeholder.Fixed('transformer_ff_dim', 256)
                hp_placeholder.Fixed('transformer_dropout_rate', 0.2)
                hp_placeholder.Fixed('dense_units_demo', 32)
                hp_placeholder.Fixed('combined_dense_units', 128)
                hp_placeholder.Fixed('combined_dropout_rate', 0.3)
                hp_placeholder.Fixed('learning_rate_schedule', 'constant')
                hp_placeholder.Fixed('initial_learning_rate', 0.0014549982711822707)

                # Build the model using the fixed recipe
                model = model_definition.model_builder(hp_placeholder, ts_shape, demo_shape, num_classes)

                # --- THIS IS THE FIX ---
                # Get the winning LR settings FROM the placeholder
                winning_lr = hp_placeholder.get('initial_learning_rate')
                winning_schedule_type = hp_placeholder.get('learning_rate_schedule')

                # Define callbacks
                callbacks = [
                    # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max'),
                    model_definition.LRLogger()
                ]

                # Initialize optimizer with the winning learning rate
                optimizer = tf.keras.optimizers.Adam(learning_rate=winning_lr, clipnorm=1.0)

                # Conditionally set up the learning rate scheduler or callback
                if winning_schedule_type == 'one_cycle':
                    total_steps = steps_per_epoch * 100
                    onecycle_callback = model_definition.OneCycleLR(max_lr=winning_lr, total_steps=total_steps)
                    callbacks.append(onecycle_callback)
                elif winning_schedule_type == 'cosine_decay':
                    decay_steps = int(config.DEFAULT_COSINE_DECAY_EPOCHS * steps_per_epoch)
                    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(winning_lr, decay_steps if decay_steps > 0 else 1)
                    optimizer.learning_rate = lr_schedule

                # Note: The 'constant' case is already handled by initializing the optimizer with winning_lr

                # Compile the model
                model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            # 6. Fit the model
            model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=100,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                class_weight=class_weights_dict,
                verbose=1
            )

            # 7. Save Out-of-Fold predictions and scaler
            if config.SAVE_OOF_PREDICTIONS:
                print(f"Generating OOF predictions for Fold {fold+1}...")
                oof_preds = model.predict(val_dataset, steps=validation_steps)
                oof_preds = oof_preds[:len(val_ids)] # Trim to actual validation set size

                np.save(os.path.join(config.MODEL_SAVE_DIR, f"oof_preds_fold_{fold+1}.npy"), oof_preds)
                np.save(os.path.join(config.MODEL_SAVE_DIR, f"oof_true_labels_fold_{fold+1}.npy"), np.array([label_map[sid] for sid in val_ids]))
                np.save(os.path.join(config.MODEL_SAVE_DIR, f"val_indices_fold_{fold+1}.npy"), val_ids)

            joblib.dump(ts_scaler, os.path.join(config.MODEL_SAVE_DIR, f"ts_scaler_fold_{fold+1}.pkl"))

            # Save the config fingerprint AFTER all other artifacts for this fold are successfully saved.
            with open(train_config_path, 'w') as f:
                json.dump(current_model_config, f, indent=4)
            # ------------------------------------

            # 8. Clean up memory before the next fold
            del train_data_pl, val_data_pl, data_cache, model
            gc.collect()
            K.clear_session()
            # --- END OF THE CORRECTED SECTION ---

    print("Training stage complete.")

def run_analysis_stage(sequence_info_all_pd, demographics_processed_all, global_gesture_label_encoder, strategy):
    """
    Orchestrates post-training analysis by calling modular helper functions.
    Consolidates OOF predictions and runs CM, Permutation Importance, and SHAP analyses.
    """

    # The analysis depends on the trained models. We use the config from fold 1 as a fingerprint.
    train_config_path = os.path.join(config.MODEL_SAVE_DIR, "training_config_fold_1.json")
    analysis_fingerprint_path = os.path.join(config.MODEL_SAVE_DIR, "analysis_fingerprint.json")

    # Check if an analysis has been run before and if the model config it was run on is the same as the current model config.
    if os.path.exists(analysis_fingerprint_path) and os.path.exists(train_config_path) and not config.FORCE_RERUN_ANALYSIS:
        with open(train_config_path, 'r') as f:
            current_model_config = json.load(f)
        with open(analysis_fingerprint_path, 'r') as f:
            last_analysis_config = json.load(f)

        if current_model_config == last_analysis_config:
            print("✅ Analysis is up-to-date with trained models. Skipping.")
            return

    # This now correctly reads the populated dictionary from the config module
    class_names = sorted(list(config.GESTURE_LABELS.keys()))

    print("\n" + "="*80 + "\n--- STAGE 5: Post-Training Analysis ---\n" + "="*80)

    # 1. Consolidate OOF predictions
    all_oof_predictions, all_oof_true_labels = [], []
    if config.SAVE_OOF_PREDICTIONS:
        for fold in range(1, config.N_SPLITS + 1):
            try:
                all_oof_predictions.append(np.load(f"{config.MODEL_SAVE_DIR}/oof_preds_fold_{fold}.npy"))
                all_oof_true_labels.append(np.load(f"{config.MODEL_SAVE_DIR}/oof_true_labels_fold_{fold}.npy"))
            except FileNotFoundError:
                print(f"Warning: OOF files for fold {fold} not found. Cannot perform full analysis.")
                return

    if not all_oof_predictions:
        print("No OOF predictions found to analyze.")
        return

    all_oof_predictions = np.concatenate(all_oof_predictions, axis=0)
    all_oof_true_labels = np.concatenate(all_oof_true_labels, axis=0)

    # 2. Generate Confusion Matrix
    oof_preds_labels = np.argmax(all_oof_predictions, axis=1)
    oof_true_labels = np.argmax(all_oof_true_labels, axis=1)
    class_names = sorted(list(config.GESTURE_LABELS.keys()))
    analysis_tools.generate_and_save_confusion_matrix(oof_true_labels, oof_preds_labels, class_names, config.MODEL_SAVE_DIR)

    # 3. Prepare Data and Model for Importance Analysis
    model_path = os.path.join(config.MODEL_SAVE_DIR, "main_model_fold_1.keras")
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Skipping feature importance.")
        return

    # The scope is needed to load a model with custom objects correctly
    with strategy.scope():
        custom_objects = {'InstanceNormalization': InstanceNormalization, 'PositionalEmbedding': PositionalEmbedding, 'TransformerEncoder': TransformerEncoder, 'SumOverTimeDimension': SumOverTimeDimension}
        model_for_importance = tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)

    # Load the numpy array, which may contain numpy-specific string types
    val_seq_ids_np = np.load(f"{config.MODEL_SAVE_DIR}/val_indices_fold_1.npy")
    # Convert it to a list of standard Python strings
    val_seq_ids = [str(item) for item in val_seq_ids_np]

    ts_scaler = joblib.load(f"{config.MODEL_SAVE_DIR}/ts_scaler_fold_1.pkl")
    with open(config.FEATURE_NAMES_FILE_PATH, 'r') as f:
        ts_feature_names = json.load(f)

    val_files = [os.path.join(config.PERMANENT_FEATURE_DIR, f"seq_{seq_id}_features.parquet") for seq_id in val_seq_ids]
    val_data_pl = pl.read_parquet(val_files)

    # Create a data cache and collect numpy arrays for analysis
    # Add [0] to extract the STRING from the tuple key
    analysis_cache = {seq_id[0]: ts_scaler.transform(group.select(ts_feature_names).to_numpy(allow_copy=True)).astype(np.float32)
                      for seq_id, group in val_data_pl.group_by("sequence_id")}

    # --- ADVANCED DIAGNOSTIC BLOCK ---
    print("\n--- Running Advanced Cache Diagnostics ---")
    lookup_ids = set(val_seq_ids)
    cache_keys = set(analysis_cache.keys())

    print(f"Number of IDs to look up from .npy file: {len(lookup_ids)}")
    print(f"Number of keys created in cache from .parquet files: {len(cache_keys)}")

    if lookup_ids == cache_keys:
        print("✅ SUCCESS: The set of lookup IDs and cache keys appear to be identical.")
    else:
        print("❌ FAILURE: Mismatch found between lookup IDs and cache keys.")
        missing_keys = lookup_ids - cache_keys
        extra_keys = cache_keys - lookup_ids
        print(f"   - IDs in .npy list but NOT in cache ({len(missing_keys)}): {list(missing_keys)[:5]}")
        print(f"   - IDs in cache but NOT in .npy list ({len(extra_keys)}): {list(extra_keys)[:5]}")
    print("--- End of Advanced Diagnostics ---\n")
    # The program will likely still error after this, but the output will tell us why.

    sample_ts_data, sample_demo_data, sample_labels = [], [], []
    label_map = {seq_id: to_categorical(label, num_classes=config.NUM_CLASSES) for seq_id, label in sequence_info_all_pd['gesture_encoded'].items()}

    for seq_id in val_seq_ids:
        # 1. Get the time-series data from the cache
        sequence = analysis_cache[seq_id]

        # 2. Apply padding or truncation to ensure consistent shape
        if sequence.shape[0] > config.MAX_SEQUENCE_LENGTH:
            sequence = sequence[:config.MAX_SEQUENCE_LENGTH]
        elif sequence.shape[0] < config.MAX_SEQUENCE_LENGTH:
            padding = np.zeros((config.MAX_SEQUENCE_LENGTH - sequence.shape[0], sequence.shape[1]), dtype=np.float32)
            sequence = np.vstack([sequence, padding])

        # 3. Always append the time-series data and the label
        sample_ts_data.append(sequence)
        sample_labels.append(label_map[seq_id])

        # 4. Conditionally get and append the demographic data
        if demographics_processed_all is not None:
            subject_id = sequence_info_all_pd.loc[seq_id]['subject']
            demographics_data = demographics_processed_all.loc[subject_id].values.astype('float32')
            sample_demo_data.append(demographics_data)

    sample_ts_data = np.array(sample_ts_data)
    sample_demo_data = np.array(sample_demo_data)
    sample_labels = np.array(sample_labels)

    # Conditionally get feature names if the dataframe exists, otherwise create an empty list
    demo_feature_names = list(demographics_processed_all.columns) if demographics_processed_all is not None else []

    # 4. Run Permutation Importance and SHAP Analyses
    analysis_tools.calculate_and_plot_permutation_importance(model_for_importance, (sample_ts_data, sample_demo_data, sample_labels), ts_feature_names, demo_feature_names, config.MODEL_SAVE_DIR)

    # For SHAP, we need a smaller background and explain set
    num_background = min(100, len(sample_ts_data))
    background_indices = np.random.choice(len(sample_ts_data), num_background, replace=False)
    background_ts = sample_ts_data[background_indices]

    num_explain = min(50, len(sample_ts_data))
    explain_indices = np.random.choice(len(sample_ts_data), num_explain, replace=False)
    explain_ts = sample_ts_data[explain_indices]

    # Conditionally create the demo samples only if demo data exists
    if sample_demo_data.size > 0:
        background_demo = sample_demo_data[background_indices]
        explain_demo = sample_demo_data[explain_indices]
    else:
        # If no demo data, create empty arrays as placeholders
        background_demo = np.array([])
        explain_demo = np.array([])

    analysis_tools.calculate_and_plot_shap_values(model_for_importance, (background_ts, background_demo), (explain_ts, explain_demo), ts_feature_names, demo_feature_names, config.MODEL_SAVE_DIR)

    print("Analysis stage complete.")

def main_orchestrator(strategy):
    """Main function to run the entire pipeline in sequence."""
    print("--- CMI Study Blueprint Execution Started ---")

    # Populate global config variables at the start of the pipeline.
    print("Discovering gesture labels from metadata...")
    temp_df_for_gestures = data_utils.load_data(is_metadata=True)
    unique_gestures = sorted(temp_df_for_gestures['gesture'].unique().to_list())
    config.GESTURE_LABELS = {gesture: i for i, gesture in enumerate(unique_gestures)}
    config.NUM_CLASSES = len(config.GESTURE_LABELS)
    del temp_df_for_gestures

    # Create all output directories at the start
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.PERMANENT_FEATURE_DIR, exist_ok=True)

    # Stage 1: Setup
    sequence_info, demographics, label_encoder = run_setup_stage()

    # Stage 2: Feature Engineering
    run_feature_engineering_stage(sequence_info)

    # Stage 3: Inter-Sequence Features
    run_inter_sequence_feature_stage(sequence_info)

    # Stage 3.1: Automated Feature Selection
    run_feature_selection_stage(sequence_info)

    # Stages 4 & 5: Training and Analysis
    if config.TRAIN:
        # Stage 4: Model Training (HPO or K-Fold)
        run_training_stage(sequence_info, demographics, label_encoder, strategy)

        # Stage 5: Post-Training Analysis
        run_analysis_stage(sequence_info, demographics, label_encoder, strategy)

        # Run final diagnostics to verify outputs
        run_diagnostics(sequence_info)
    else:
        print("Running in Inference Mode (TRAIN=False). Analysis skipped.")

    print("--- CMI Study Blueprint Execution Finished ---")

def run_diagnostics(sequence_info_all_pd):
    """Run post-run validation checks to ensure data integrity."""
    print("\n" + "="*50 + "\n--- RUNNING POST-RUN DIAGNOSTICS ---\n" + "="*50)

    try:
        # 1. Get sequence IDs from metadata
        meta_ids = set(sequence_info_all_pd.index.to_list())

        # 2. Get sequence IDs from the generated Parquet files
        feature_files = [f for f in os.listdir(config.PERMANENT_FEATURE_DIR) if f.endswith('.parquet')]
        # Assumes naming convention is 'seq_SEQ_XXXXXX_features.parquet'

        # This robustly removes the prefix and suffix to get the full ID.
        feature_ids = set(f.removeprefix('seq_').removesuffix('_features.parquet') for f in feature_files)

        # 3. Compare the sets
        if meta_ids == feature_ids:
            print(f"✅ SUCCESS: All {len(meta_ids)} sequences have a corresponding feature file.")
        else:
            missing_in_features = meta_ids - feature_ids
            extra_in_features = feature_ids - meta_ids
            if missing_in_features:
                print(f"⚠️ WARNING: {len(missing_in_features)} sequences are MISSING feature files.")
                print(f"   (Example missing: {list(missing_in_features)[:3]})")
            if extra_in_features:
                print(f"⚠️ WARNING: {len(extra_in_features)} EXTRA feature files were found.")
                print(f"   (Example extra: {list(extra_in_features)[:3]})")
    except Exception as e:
        print(f"❌ ERROR: Diagnostics failed to run. Reason: {e}")

def run_feature_selection_stage(sequence_info):
    """
    Loads all features, runs a selection algorithm, and saves the list of best features.
    """
    if not config.ENABLE_FEATURE_SELECTION:
        return # Skip if not enabled

    print("\n" + "="*80 + "\n--- NEW STAGE: Automated Feature Selection ---\n" + "="*80)

    # Load all feature data (this can be memory intensive)
    all_feature_files = [os.path.join(config.PERMANENT_FEATURE_DIR, f) for f in os.listdir(config.PERMANENT_FEATURE_DIR) if f.endswith('.parquet')]
    df_all = pl.read_parquet(all_feature_files)

    # Use only static features for selection
    static_features_df = df_all.group_by('sequence_id').first()

    # Prepare data
    with open(config.FEATURE_NAMES_FILE_PATH, 'r') as f:
        all_feature_names = json.load(f)
    static_feature_names = [f for f in all_feature_names if any(s in f for s in ['_mean', '_std', '_fft', '_wavelet'])]

    X = static_features_df.select(static_feature_names).to_numpy()
    y = static_features_df.join(sequence_info, on='sequence_id').select('gesture_encoded').to_numpy().ravel()

    # Run selection
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1))
    selector.fit(X, y)

    selected_features = np.array(static_feature_names)[selector.get_support()].tolist()
    print(f"Selected {len(selected_features)} features out of {len(static_feature_names)}.")

    # Save the new list of selected features
    selected_features_path = os.path.join(config.MODEL_SAVE_DIR, 'selected_feature_names.json')
    with open(selected_features_path, 'w') as f:
        json.dump(selected_features, f, indent=4)
    print(f"Selected feature list saved to {selected_features_path}")