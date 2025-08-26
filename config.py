# config.py
import os
import tensorflow as tf

# --- Environment and Execution Control ---
ENV_PARAM = 0  # 1=Kaggle, 0=Local
TRAIN = True   # Set to False for submission/inference

# --- Framework Control Flags (for development) ---
FORCE_RERUN_SETUP = True
FORCE_RERUN_FEATURE_ENGINEERING = False
FORCE_RERUN_INTER_SEQUENCE = False
FORCE_RERUN_TRAINING = False
FORCE_RERUN_ANALYSIS = True

# --- Path Configuration: Define paths based on environment
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

# Model and Data Parameters
# CRITICAL: MAX_SEQUENCE_LENGTH.
# Impact: Too short may truncate a gesture; too long may add noise/padding.
# Recommended range: 64, 128 (current), 256.
MAX_SEQUENCE_LENGTH = 128 # Maximum length for sensor sequences

N_SPLITS = 5 # Number of folds for cross-validation
RANDOM_STATE = 42 # Seed for reproducibility

# CRITICAL: GLOBAL_BATCH_SIZE.
# Impact: Affects training speed, memory usage, and convergence stability.
# Recommended range: 32, 64 (current), 128.
GLOBAL_BATCH_SIZE = 32 # Batch size for training

WORKERS_AUTOTUNE = tf.data.AUTOTUNE # For tf.data.Dataset prefetching

# Granular Sensor Group Controls for Feature Generation
# These flags control which *base* sensor columns are included in derived features and static feature calculations.
# CRITICAL: ENABLE_TOF_FEATURES.
# Impact: Enabling this drastically increases data dimensionality. It can improve performance
# but also significantly increases computational cost and and memory. Use with caution.
ENABLE_DEMOGRAPHICS = True
ENABLE_ACC_FEATURES = True
ENABLE_ROT_FEATURES = True
ENABLE_THM_FEATURES = True
ENABLE_TOF_FEATURES = True

# Feature Engineering Control Variables - Organized by Level and Granularity
# Level 1: Basic/Fundamental Per-Row Features
ENABLE_GRAVITY_REMOVAL = True # Applies rolling mean-based gravity removal from acc
ENABLE_IMU_DERIVED_FEATURES = True # Calculates acc/rot magnitude, jerk, quat-based ang_vel/dist

# Level 2: Data Augmentation (applied during data generation for training)
# CRITICAL: ENABLE_DATA_AUGMENTATION.
# Impact: Augmentation is crucial for a robust model on limited data. Keep it True.
ENABLE_DATA_AUGMENTATION = True # Add noise, time warping, magnitude scaling during training

# Level 3: Comprehensive/Advanced Intra-Sequence Static Features
# Master switches for feature types (if False, all sub-features of this type are disabled)
# CRITICAL: These flags control which advanced features are computed. Enable them selectively.
# Impact: They introduce highly predictive features but increase computational cost.
# Recommended strategy: Start with basic stats, then enable advanced features one by one.
ENABLE_BASIC_STATS = True # Controls global statistical features (mean, std, min, max, median, range, skew, kurtosis)
ENABLE_DOMINANT_FFT_FEATURES = True # Controls dominant FFT amplitude and frequency features
ENABLE_COMPREHENSIVE_AUTOCORR = True # Autocorrelation at lag 1
ENABLE_COMPREHENSIVE_MISSING_INDICATORS = True # Inactivity indicators for TOF/THM
ENABLE_NOLDS_ENTROPY = False # Sample Entropy (requires nolds library)
ENABLE_COMPREHENSIVE_WINDOWED_STATS = False # Aggregated statistics over sliding windows
ENABLE_COMPREHENSIVE_FFT_SPECTRAL = False # Enhanced frequency domain features (energy bands, centroids, etc.)
ENABLE_COMPREHENSIVE_CORRELATIONS = True # Cross-axis correlations
ENABLE_COMPREHENSIVE_PCA = False # PCA explained variance ratios

# Level 4: Inter-Sequence Features
# CRITICAL: ENABLE_INTER_SEQUENCE_FEATURES.
# Impact: This feature set is powerful for subject-specific learning and can significantly
# improve performance by normalizing a sequence relative to a subject's behavior.
ENABLE_INTER_SEQUENCE_FEATURES = True # Deviations from subject-level means/stds, sequence position

# NEW Feature Engineering Control Variables (from A.1-A.4)
# CRITICAL: TOF_DR_COMPONENTS.
# Impact: Determines the dimensionality of the TOF data after reduction.
# Recommended range: 2 to 10.
ENABLE_TOF_DIMENSIONALITY_REDUCTION = False # A.1. Apply PCA/UMAP on 64 TOF bins
TOF_DR_METHOD = 'umap' # 'pca' or 'umap' (if umap-learn is installed)
TOF_DR_COMPONENTS = 3 # Number of components for TOF DR
ENABLE_TOF_SHAPE_FEATURES = False # A.2. Extract features describing the "shape" of TOF signal
ENABLE_THM_SPATIAL_TEMPORAL_FEATURES = False # A.3. Spatial/temporal features for THM sensors
ENABLE_INTERACTION_FEATURES = False # A.4. Interaction features between sensor types

# NEW: Control for dropping raw TOF features after derivation
# Set this to True to use ONLY derived TOF features (like PCA, shape, etc.)
# and exclude the 320 raw TOF columns from the final model input.
DROP_RAW_TOF_FEATURES = False

# Granular Feature Type per Sensor Group Controls
# These flags allow fine-grained control over which Level 3 features are applied to which sensor types.
# Only effective if the corresponding master ENABLE_COMPREHENSIVE_* flag (Level 3) is also True.
# IMU refers to Accelerometer and Rotation (gyroscope/quaternion) data and their derivatives.
# THM refers to Thermistor data.
# TOF refers to Time-of-Flight data.

# Basic Statistics
ENABLE_BASIC_STATS_IMU = True
ENABLE_BASIC_STATS_THM = False
ENABLE_BASIC_STATS_TOF = False

# Dominant FFT Features
ENABLE_DOMINANT_FFT_FEATURES_IMU = False
ENABLE_DOMINANT_FFT_FEATURES_THM = False
ENABLE_DOMINANT_FFT_FEATURES_TOF = False

# Comprehensive FFT Spectral Features
ENABLE_COMPREHENSIVE_FFT_SPECTRAL_IMU = False
ENABLE_COMPREHENSIVE_FFT_SPECTRAL_THM = False
ENABLE_COMPREHENSIVE_FFT_SPECTRAL_TOF = False

# Autocorrelation
ENABLE_COMPREHENSIVE_AUTOCORR_IMU = False
ENABLE_COMPREHENSIVE_AUTOCORR_THM = False
ENABLE_COMPREHENSIVE_AUTOCORR_TOF = False

# NOLDS Entropy
ENABLE_NOLDS_ENTROPY_IMU = False
ENABLE_NOLDS_ENTROPY_THM = False
ENABLE_NOLDS_ENTROPY_TOF = False

# Windowed Statistics
ENABLE_COMPREHENSIVE_WINDOWED_STATS_IMU = False
ENABLE_COMPREHENSIVE_WINDOWED_STATS_THM = False
ENABLE_COMPREHENSIVE_WINDOWED_STATS_TOF = False

# Wavelet Transforms add features that describe how the frequency of a signal changes over time, which is excellent for complex gestures
ENABLE_WAVELET_FEATURES = True

# Automated Feature Selection: This adds a new pipeline stage to automatically select the most impactful features.
ENABLE_FEATURE_SELECTION = False

# 'gru', 'transformer', or 'hybrid'
MODEL_TYPE = 'gru' 
# 'simple' or 'bahdanau'
ATTENTION_TYPE = 'simple' 

# NEW Model & Training Configuration (from B.1, B.3, B.4)
# CRITICAL: HPO_ENABLED.
# Impact: Setting this to True automates the search for optimal hyperparameters. It's the most
# efficient way to fine-tune the model, but it is computationally intensive.
HPO_ENABLED = True # Set to True to enable Keras Tuner Hyperparameter Optimization
HPO_NUM_TRIALS = 30 # Number of HP combinations to try for Keras Tuner (set low for testing)
# CRITICAL: ENABLE_TRANSFORMER_BLOCK.
# Impact: A major architectural choice. Replaces GRUs with a Transformer Encoder. Can capture
# complex, long-range dependencies but may require more data.
ENABLE_TRANSFORMER_BLOCK = True # Set to True to use Transformer Encoder instead of Bidirectional GRU
TRANSFORMER_NUM_HEADS = 4 # For MultiHeadAttention if Transformer block is enabled
TRANSFORMER_FF_DIM = 128 # Feed-forward dimension for Transformer block

# New: Default Learning Rate Schedule parameters when HPO is OFF
# CRITICAL: DEFAULT_LR_SCHEDULE_TYPE and DEFAULT_INITIAL_LEARNING_RATE.
# Impact: These are arguably the most important training parameters. Cosine decay is generally
# a good choice over a constant LR. A good starting LR is crucial.
# Recommended range for DEFAULT_INITIAL_LEARNING_RATE: 1e-4, 5e-4, 1e-3 (current).
DEFAULT_LR_SCHEDULE_TYPE = 'cosine_decay' # 'constant' or 'cosine_decay', or 'one_cycle'
DEFAULT_INITIAL_LEARNING_RATE = 0.0005
DEFAULT_COSINE_DECAY_EPOCHS = 75 # Number of epochs over which to decay for fixed schedule

# NEW: Control for Normalization Layer in the CNN/Dense blocks
# CRITICAL: NORMALIZATION_LAYER_TYPE
# Impact: Controls which type of normalization is used.
# 'batch_norm' is the standard, 'instance_norm' is a potential alternative for small batches.
#NORMALIZATION_LAYER_TYPE = 'instance_norm' # 'batch_norm' or 'instance_norm'

# Target variable mapping - Will be populated dynamically
GESTURE_LABELS = {}
GESTURE_LABELS_INV = {} # Inverse mapping for convenience
NUM_CLASSES = 0

# Sensor columns (base columns)
ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
ROT_COLS = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
THM_COLS = [f'thm_{i}' for i in range(1, 6)]
TOF_COLS = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]

# Combined list of all raw sensor columns
ALL_RAW_SENSOR_COLS = ACC_COLS + ROT_COLS + THM_COLS + TOF_COLS

# Columns to exclude from sensor data reading
EXCLUDE_COLS_FROM_READING = ['row_id', 'orientation', 'behavior', 'phase', 'sequence_type']

# Demographics columns
DEMO_CATEGORICAL_COLS = ['adult_child', 'sex', 'handedness']
DEMO_NUMERICAL_COLS = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']

# NEW: Control for OOF (Out-of-Fold) Prediction Saving (B.5)
SAVE_OOF_PREDICTIONS = True # Set to True to collect OOF predictions for ensembling, False to skip.

# Choose which SHAP explainer to use:
# 'deep': Fast, but may have version compatibility issues.
# 'kernel': Slow, but model-agnostic and more robust.
SHAP_EXPLAINER_TYPE = 'deep'

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

    # --- Specific Feature Parameters (from v1.6.6) ---
    'ENABLE_TOF_DIMENSIONALITY_REDUCTION', 'TOF_DR_METHOD', 'TOF_DR_COMPONENTS',
    'ENABLE_TOF_SHAPE_FEATURES', 'ENABLE_THM_SPATIAL_TEMPORAL_FEATURES',
    'ENABLE_INTERACTION_FEATURES', 'DROP_RAW_TOF_FEATURES',

    # --- Data Shape Parameters ---
    'MAX_SEQUENCE_LENGTH'
    
    # Wavelet Transforms
    'ENABLE_WAVELET_FEATURES'
]