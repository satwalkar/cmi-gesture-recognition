# config.py

import os

# ==========================================================
# SECTION 1: CORE PIPELINE CONTROL
# ==========================================================

# --- Environment and Execution Control ---
ENV_PARAM = 0  # 1=Kaggle, 0=Local
TRAIN = True   # Set to False for submission/inference

# --- Framework Rerun Flags ---
# Use these to force specific stages to run again, ignoring checkpoints.
FORCE_RERUN_SETUP = True
FORCE_RERUN_FEATURE_ENGINEERING = True
FORCE_RERUN_INTER_SEQUENCE = True
FORCE_RERUN_TRAINING = True
FORCE_RERUN_ANALYSIS = True

# ==========================================================
# SECTION 2: PATHS & GLOBAL CONSTANTS
# ==========================================================

# --- Path Configuration ---
if ENV_PARAM == 1:
    PARENT_DIR = "/kaggle/input/cmi-detect-behavior-with-sensor-data/"
    TRAIN_CSV_PATH = f"{PARENT_DIR}/train.csv"
    DEMOGRAPHICS_CSV_PATH = f"{PARENT_DIR}/train_demographics.csv"
    TEST_DEMOGRAPHICS_CSV_PATH = f"{PARENT_DIR}/test_demographics.csv"
    TEST_CSV_PATH = f"{PARENT_DIR}/test.csv"
    MODEL_SAVE_DIR = "/kaggle/working/pre_trained_model"
else:
    PARENT_DIR = "C:\\Omky\\Projects\\Python\\Learning\\cmi-detect-behavior-with-sensor-data"
    TRAIN_CSV_PATH = os.path.join(PARENT_DIR, "data", "train.csv")
    DEMOGRAPHICS_CSV_PATH = os.path.join(PARENT_DIR, "data", "train_demographics.csv")
    TEST_DEMOGRAPHICS_CSV_PATH = os.path.join(PARENT_DIR, "data", "test_demographics.csv")
    TEST_CSV_PATH = os.path.join(PARENT_DIR, "data", "test.csv")
    MODEL_SAVE_DIR = os.path.join(PARENT_DIR, "pre_trained_model")

PERMANENT_FEATURE_DIR = os.path.join(MODEL_SAVE_DIR, "pre_engineered_features")
CONFIG_FILE_PATH = os.path.join(PERMANENT_FEATURE_DIR, "feature_config.json")
FEATURE_NAMES_FILE_PATH = os.path.join(PERMANENT_FEATURE_DIR, "feature_names.json")
SUBJECT_STATS_PATH = os.path.join(MODEL_SAVE_DIR, "subject_level_stats.pkl")

# --- Global Constants ---
# CRITICAL: MAX_SEQUENCE_LENGTH. # Impact: Too short may truncate a gesture; too long may add noise/padding.
# Recommended range: 64, 128 (current), 256.
MAX_SEQUENCE_LENGTH = 128 # Maximum length for sensor sequences
N_SPLITS = 5
RANDOM_STATE = 42
WORKERS_AUTOTUNE = -1  # tf.data.AUTOTUNE

# ==========================================================
# SECTION 3: FEATURE ENGINEERING CONFIGURATION
# ==========================================================

# --- 3.1 Master Sensor Switches ---
# Granular Sensor Group Controls for Feature Generation
# These flags control which *base* / raw sensor columns are included in derived features and static feature calculations.
ENABLE_DEMOGRAPHICS = True

# IMU refers to Accelerometer and Rotation (gyroscope/quaternion) data and their derivatives.
ENABLE_ACC_FEATURES = True
ENABLE_ROT_FEATURES = True

# THM refers to Thermistor data.
ENABLE_THM_FEATURES = False

# Impact: Enabling this drastically increases data dimensionality. It can improve performance
# but also significantly increases computational cost and and memory. Use with caution.
ENABLE_TOF_FEATURES = False # TOF refers to Time-of-Flight data. CRITICAL

# --- 3.2 Per-Row Derived Features ---
# Feature Engineering Control Variables - Organized by Level and Granularity, Basic/Fundamental Per-Row Features
ENABLE_GRAVITY_REMOVAL = True # Applies rolling mean-based gravity removal from acc
ENABLE_IMU_DERIVED_FEATURES = True # Calculates acc/rot magnitude, jerk, quat-based ang_vel/dist

# --- 3.3 Inter-Sequence Features ---
# Impact: This feature set is powerful for subject-specific learning and can significantly
# improve performance by normalizing a sequence relative to a subject's behavior.
ENABLE_INTER_SEQUENCE_FEATURES = True # CRITICAL: Deviations from subject-level means/stds, sequence position

# --- 3.4 Comprehensive/Advanced Intra-Sequence Static (Aggregated) Features & Granular Feature Type per Sensor Group Controls ---
# Master switches for feature types (if False, all sub-features of this type are disabled)
# They introduce highly predictive, control large/computationally expensive advanced feature groups.
# Recommended strategy: For experiments, it's recommended to enable them one by one (e.g. basic stats) to measure their impact.

# Basic Statistics
ENABLE_BASIC_STATS = True # Controls global statistical features (mean, std, min, max, median, range, skew, kurtosis)
ENABLE_BASIC_STATS_IMU = True
ENABLE_BASIC_STATS_THM = False
ENABLE_BASIC_STATS_TOF = False

# Dominant FFT Features
ENABLE_DOMINANT_FFT_FEATURES = True # Controls dominant FFT amplitude and frequency features
ENABLE_DOMINANT_FFT_FEATURES_IMU = True
ENABLE_DOMINANT_FFT_FEATURES_THM = False
ENABLE_DOMINANT_FFT_FEATURES_TOF = False

# Comprehensive FFT Spectral Features
ENABLE_COMPREHENSIVE_FFT_SPECTRAL = True # Enhanced frequency domain features (energy bands, centroids, etc.)
ENABLE_COMPREHENSIVE_FFT_SPECTRAL_IMU = True
ENABLE_COMPREHENSIVE_FFT_SPECTRAL_THM = False
ENABLE_COMPREHENSIVE_FFT_SPECTRAL_TOF = False

# Autocorrelation
ENABLE_COMPREHENSIVE_AUTOCORR = True # Autocorrelation at lag 1
ENABLE_COMPREHENSIVE_AUTOCORR_IMU = True
ENABLE_COMPREHENSIVE_AUTOCORR_THM = False
ENABLE_COMPREHENSIVE_AUTOCORR_TOF = False

# NOLDS Entropy
ENABLE_NOLDS_ENTROPY = False  # Sample Entropy (requires nolds library)
ENABLE_NOLDS_ENTROPY_IMU = False
ENABLE_NOLDS_ENTROPY_THM = False
ENABLE_NOLDS_ENTROPY_TOF = False

# Windowed Statistics
ENABLE_COMPREHENSIVE_WINDOWED_STATS = True # Aggregated statistics over sliding windows
ENABLE_COMPREHENSIVE_WINDOWED_STATS_IMU = True
ENABLE_COMPREHENSIVE_WINDOWED_STATS_THM = False
ENABLE_COMPREHENSIVE_WINDOWED_STATS_TOF = False

# Other Comprehensive Features
ENABLE_COMPREHENSIVE_CORRELATIONS = True # Cross-axis correlations
ENABLE_COMPREHENSIVE_PCA = True # PCA explained variance ratios
ENABLE_COMPREHENSIVE_MISSING_INDICATORS = False # Inactivity indicators for TOF/THM

# --- 3.5 Advanced & Experimental Features ---
# Wavelet Transforms add features that describe how the frequency of a signal changes over time, which is excellent for complex gestures
ENABLE_WAVELET_FEATURES = True

# Automated Feature Selection: This adds a new pipeline stage to automatically select the most impactful features.
ENABLE_FEATURE_SELECTION = False

# TOF-Specific Advanced Features # Impact: Determines the dimensionality of the TOF data after reduction.
ENABLE_TOF_DIMENSIONALITY_REDUCTION = False # A.1. Apply PCA/UMAP on 64 TOF bins
TOF_DR_METHOD = 'umap' # 'pca' or 'umap' (if umap-learn is installed)
TOF_DR_COMPONENTS = 3 # Number of components for TOF DR. CRITICAL. # Recommended range: 2 to 10.
ENABLE_TOF_SHAPE_FEATURES = False # A.2. Extract features describing the "shape" of TOF signal

# NEW: Control for dropping raw TOF features after derivation
# Set this to True to use ONLY derived TOF features (like PCA, shape, etc.)
# and exclude the 320 raw TOF columns from the final model input.
DROP_RAW_TOF_FEATURES = False # Set True to use ONLY derived TOF features

# THM-Specific Advanced Features
ENABLE_THM_SPATIAL_TEMPORAL_FEATURES = False # A.3. Spatial/temporal features for THM sensors
ENABLE_INTERACTION_FEATURES = False # A.4. Interaction features between sensor types

# ==========================================================
# SECTION 4: MODEL & TRAINING CONFIGURATION
# ==========================================================

# --- 4.1 Model Architecture ---
MODEL_TYPE = 'hybrid'  # 'gru', 'transformer', or 'hybrid'
ATTENTION_TYPE = 'bahdanau'  # 'simple' or 'bahdanau' (only used for 'gru' model_type)

# --- 4.2 Training Strategy ---
# CRITICAL: GLOBAL_BATCH_SIZE.
# Impact: Affects training speed, memory usage, and convergence stability.
GLOBAL_BATCH_SIZE = 32

# CRITICAL: Data Augmentation (applied during data generation for training)
# Impact: Augmentation is crucial for a robust model on limited data.
ENABLE_DATA_AUGMENTATION = True # Add noise, time warping, magnitude scaling during training

# --- 4.3 Hyperparameter Optimization (HPO) ---
# CRITICAL: HPO_ENABLED.
# Impact: Setting this to True automates the search for optimal hyperparameters. It's the most
# efficient way to fine-tune the model, but it is computationally intensive.
HPO_ENABLED = True # Set to True to enable Keras Tuner Hyperparameter Optimization
HPO_NUM_TRIALS = 30 # Number of HP combinations to try for Keras Tuner (set low for testing)

# --- 4.4 Fixed Training Parameters (when HPO_ENABLED = False) ---
# CRITICAL: DEFAULT_LR_SCHEDULE_TYPE and DEFAULT_INITIAL_LEARNING_RATE.
# New: Default Learning Rate Schedule parameters when HPO is OFF
# Impact: These are arguably the most important training parameters for a fixed run. 
# Cosine decay is generally a good choice over a constant LR. A good starting LR is crucial.
DEFAULT_LR_SCHEDULE_TYPE = 'cosine_decay' # 'constant' or 'cosine_decay', or 'one_cycle'
DEFAULT_INITIAL_LEARNING_RATE = 1e-3 # 0.0005 # Recommended range for DEFAULT_INITIAL_LEARNING_RATE: 1e-4, 5e-4, 1e-3 (current).
DEFAULT_COSINE_DECAY_EPOCHS = 100 # Number of epochs over which to decay for fixed schedule

# ==========================================================
# SECTION 5: STATIC DEFINITIONS
# ==========================================================

# Target variable mapping - Will be populated dynamically
GESTURE_LABELS = {}
GESTURE_LABELS_INV = {} # Inverse mapping for convenience
NUM_CLASSES = 0

# --- Sensor Column Definitions ---
ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
ROT_COLS = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
THM_COLS = [f'thm_{i}' for i in range(1, 6)]
TOF_COLS = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]

# Combined list of all raw sensor columns
ALL_RAW_SENSOR_COLS = ACC_COLS + ROT_COLS + THM_COLS + TOF_COLS

# --- Analysis & Saving ---
# Columns to exclude from sensor data reading
EXCLUDE_COLS_FROM_READING = ['row_id', 'orientation', 'behavior', 'phase', 'sequence_type']

# Demographics columns
DEMO_CATEGORICAL_COLS = ['adult_child', 'sex', 'handedness']
DEMO_NUMERICAL_COLS = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']

# NEW: Control for OOF (Out-of-Fold) Prediction Saving (B.5)
SAVE_OOF_PREDICTIONS = True # Set to True to collect OOF predictions for ensembling, False to skip.

# Control for OOF (Out-of-Fold) Prediction Saving. Set to True to collect OOF predictions for ensembling, False to skip.
SAVE_OOF_PREDICTIONS = True

# 'deep': Fast, but may have version compatibility issues.
# 'kernel': Slow, but model-agnostic and more robust.
SHAP_EXPLAINER_TYPE = 'deep' # Choose which SHAP explainer to use

# ==========================================================
# List of keys to monitor for feature regeneration
# ==========================================================
# The function in pipeline_stages.py will use this list to check for changes.
FEATURE_CONFIG_KEYS = [
    # --- Master Data Switches ---
    'ENABLE_ACC_FEATURES', 'ENABLE_ROT_FEATURES', 'ENABLE_THM_FEATURES', 'ENABLE_TOF_FEATURES',

    # --- Per-Row Derived Feature Switches ---
    'ENABLE_GRAVITY_REMOVAL', 'ENABLE_IMU_DERIVED_FEATURES',

    # --- Static Intra-Sequence Feature Switches ---
    'ENABLE_BASIC_STATS', 'ENABLE_DOMINANT_FFT_FEATURES', 'ENABLE_COMPREHENSIVE_AUTOCORR',
    'ENABLE_COMPREHENSIVE_MISSING_INDICATORS', 'ENABLE_NOLDS_ENTROPY',
    'ENABLE_COMPREHENSIVE_WINDOWED_STATS', 'ENABLE_COMPREHENSIVE_FFT_SPECTRAL',
    'ENABLE_COMPREHENSIVE_CORRELATIONS', 'ENABLE_COMPREHENSIVE_PCA',

    # --- Inter-Sequence Feature Switch ---
    'ENABLE_INTER_SEQUENCE_FEATURES',

    # --- Specific Feature Parameters ---
    'ENABLE_TOF_DIMENSIONALITY_REDUCTION', 'TOF_DR_METHOD', 'TOF_DR_COMPONENTS',
    'ENABLE_TOF_SHAPE_FEATURES', 'ENABLE_THM_SPATIAL_TEMPORAL_FEATURES',
    'ENABLE_INTERACTION_FEATURES', 'DROP_RAW_TOF_FEATURES',

    # --- Data Shape Parameters ---
    'MAX_SEQUENCE_LENGTH'

    # Wavelet Transforms
    'ENABLE_WAVELET_FEATURES'
]