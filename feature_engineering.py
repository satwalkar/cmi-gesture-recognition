# feature_engineering.py
import os
import gc
import numpy as np
import polars as pl
import warnings
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, sosfilt, find_peaks
from statsmodels.tsa.stattools import acf
from sklearn.decomposition import PCA
import pandas as pd
import umap

# Import project-specific configuration
import config

# --- NOLDS Library Availability Check ---
# This check is local to the feature engineering module.
try:
    import nolds
    _NOLDS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: 'nolds' library not found. Entropy features will be skipped.")
    _NOLDS_AVAILABLE = False
    # Define dummy functions to prevent NameError if nolds is not installed
    def sample_entropy(*args, **kwargs): return np.nan
    def approx_entropy(*args, **kwargs): return np.nan

def get_pl_data_as_np(df_or_series: pl.DataFrame, col_name: str = None) -> np.ndarray:
    """Safely extracts a column from a Polars DataFrame as a NumPy array, handling potential NaN/Inf."""
    if col_name:
        if col_name not in df_or_series.columns:
            return np.array([], dtype=np.float32)
        # Use .to_numpy(allow_copy=True) for a safe, mutable copy
        data = df_or_series.select(pl.col(col_name)).to_series().to_numpy(allow_copy=True)
    else:
        data = df_or_series.to_numpy(allow_copy=True)

    # Replace non-finite values with 0.0
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

# -------------------------------------------------------------------
# All feature calculation helper functions from the original script
# are placed here.
# -------------------------------------------------------------------

def _calculate_per_row_derived_features(df_seq: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates per-row (time-step) derived features like linear acceleration, jerk,
    and quaternion-based angular velocity/distance, based on granular control flags.
    """
    # Ensure 'row_id' is present for rolling operations if not already
    if 'row_id' not in df_seq.columns:
        # Replaced the deprecated `with_row_count` with `with_row_index`
        # The new method defaults to a column name of 'index', so we explicitly name it 'row_id'
        df_seq = df_seq.with_row_index(name="row_id")

    # Initialize lists to track which columns are actually present and used
    active_acc_cols = [col for col in config.ACC_COLS if col in df_seq.columns]
    active_rot_cols = [col for col in config.ROT_COLS if col in df_seq.columns]

    # Basic IMU derived features (magnitudes)
    if config.ENABLE_IMU_DERIVED_FEATURES:
        if config.ENABLE_ACC_FEATURES and active_acc_cols:
            df_seq = df_seq.with_columns([
                (pl.col('acc_x').pow(2) + pl.col('acc_y').pow(2) + pl.col('acc_z').pow(2)).sqrt().alias('acc_mag'),
            ])
        if config.ENABLE_ROT_FEATURES and active_rot_cols:
            df_seq = df_seq.with_columns([
                (pl.col('rot_x').pow(2) + pl.col('rot_y').pow(2) + pl.col('rot_z').pow(2)).sqrt().alias('rot_mag'), # Assuming rot_x,y,z are vector part, rot_w is scalar
            ])

    # Gravity Removal (using rolling mean as a low-pass filter) for Linear Acceleration
    gravity_cols_to_drop = []
    if config.ENABLE_GRAVITY_REMOVAL and config.ENABLE_ACC_FEATURES and active_acc_cols:
        df_seq = df_seq.with_columns([
            pl.col('acc_x').rolling_mean(window_size=50, min_samples=1, center=True).over('sequence_id').alias('gravity_acc_x'),
            pl.col('acc_y').rolling_mean(window_size=50, min_samples=1, center=True).over('sequence_id').alias('gravity_acc_y'),
            pl.col('acc_z').rolling_mean(window_size=50, min_samples=1, center=True).over('sequence_id').alias('gravity_acc_z')
        ])
        # Add linear acceleration as new columns
        df_seq = df_seq.with_columns([
            (pl.col('acc_x') - pl.col('gravity_acc_x')).alias('lin_acc_x'),
            (pl.col('acc_y') - pl.col('gravity_acc_y')).alias('lin_acc_y'),
            (pl.col('acc_z') - pl.col('gravity_acc_z')).alias('lin_acc_z'),
        ])
        gravity_cols_to_drop = ['gravity_acc_x', 'gravity_acc_y', 'gravity_acc_z']

    # Jerk (derivative of linear acceleration)
    if config.ENABLE_IMU_DERIVED_FEATURES:
        if config.ENABLE_ACC_FEATURES and 'lin_acc_x' in df_seq.columns: # Jerk needs linear acceleration
            df_seq = df_seq.with_columns([
                pl.col('lin_acc_x').diff().over('sequence_id').fill_null(0).alias('jerk_x'),
                pl.col('lin_acc_y').diff().over('sequence_id').fill_null(0).alias('jerk_y'),
                pl.col('lin_acc_z').diff().over('sequence_id').fill_null(0).alias('jerk_z'),
            ])
            df_seq = df_seq.with_columns([
                (pl.col('jerk_x').pow(2) + pl.col('jerk_y').pow(2) + pl.col('jerk_z').pow(2)).sqrt().alias('jerk_mag')
            ])

        # Initialize angular velocity and distance arrays to empty/zeros
        # They will be populated if the conditions in the if-statement below are met.
        # Otherwise, they remain zero-filled, ensuring the pl.Series creation doesn't fail.
        # The size should be df_seq.height, which can be 0 or 1, resulting in empty arrays.
        # This prevents UnboundLocalError.
        ang_vel_quat_x_vals = np.zeros(df_seq.height, dtype=np.float32)
        ang_vel_quat_y_vals = np.zeros(df_seq.height, dtype=np.float32)
        ang_vel_quat_z_vals = np.zeros(df_seq.height, dtype=np.float32)
        ang_dist_quat_vals = np.zeros(df_seq.height, dtype=np.float32)

        # Quaternion-based Angular Velocity and Distance (requires scipy.spatial.transform.Rotation)
        if config.ENABLE_ROT_FEATURES and all(col in df_seq.columns for col in config.ROT_COLS) and df_seq.height > 1:
            quat_cols = ['rot_x', 'rot_y', 'rot_z', 'rot_w'] # Ensure rot_w is included for quaternions

            # Convert to numpy for scipy operations
            quat_values = df_seq.select(quat_cols).to_numpy()
            sampling_rate = 50 # Assuming 50Hz for CMI data
            time_delta = 1.0 / sampling_rate

            for i in range(df_seq.height - 1):
                q_t = quat_values[i]
                q_t_plus_dt = quat_values[i+1]

                # Skip if quaternions are invalid or near zero magnitude
                if not (np.all(np.isfinite(q_t)) and np.linalg.norm(q_t) > 1e-6 and
                        np.all(np.isfinite(q_t_plus_dt)) and np.linalg.norm(q_t_plus_dt) > 1e-6):
                    continue

                try:
                    rot_t = R.from_quat(q_t[[1,2,3,0]]) # scipy expects [x,y,z,w]
                    rot_t_plus_dt = R.from_quat(q_t_plus_dt[[1,2,3,0]]) # scipy expects [x,y,z,w]
                    delta_rot = rot_t.inv() * rot_t_plus_dt
                    # as_rotvec returns [angle_x, angle_y, angle_z] representing a rotation vector
                    ang_vel_quat_x_vals[i] = delta_rot.as_rotvec()[0] / time_delta
                    ang_vel_quat_y_vals[i] = delta_rot.as_rotvec()[1] / time_delta
                    ang_vel_quat_z_vals[i] = delta_rot.as_rotvec()[2] / time_delta
                    ang_dist_quat_vals[i] = np.linalg.norm(delta_rot.as_rotvec()) # Magnitude of rotation vector
                except ValueError: # Catch errors from invalid quaternion input to scipy
                    pass
            del quat_values # Free memory

        # These lines are now safe because the _vals arrays are always initialized.
        df_seq = df_seq.with_columns([
            pl.Series("ang_vel_quat_x", ang_vel_quat_x_vals), pl.Series("ang_vel_quat_y", ang_vel_quat_y_vals),
            pl.Series("ang_vel_quat_z", ang_vel_quat_z_vals), pl.Series("ang_dist_quat", ang_dist_quat_vals),
        ])

        # --- CORRECTED ANGULAR ACCELERATION CALCULATION START ---
        # Removed the old, physically incorrect ang_acc_x/y/z from rot_x/y/z.
        # Now calculating angular acceleration as the derivative of the
        # quaternion-derived angular velocity components.
        if config.ENABLE_ROT_FEATURES and all(col in df_seq.columns for col in ['ang_vel_quat_x', 'ang_vel_quat_y', 'ang_vel_quat_z']):
            df_seq = df_seq.with_columns([
                pl.col('ang_vel_quat_x').diff().over('sequence_id').fill_null(0).alias('ang_acc_quat_x'),
                pl.col('ang_vel_quat_y').diff().over('sequence_id').fill_null(0).alias('ang_acc_quat_y'),
                pl.col('ang_vel_quat_z').diff().over('sequence_id').fill_null(0).alias('ang_acc_quat_z'),
            ])
            # Optional: Add magnitude of angular acceleration
            df_seq = df_seq.with_columns([
                (pl.col('ang_acc_quat_x').pow(2) + pl.col('ang_acc_quat_y').pow(2) + pl.col('ang_acc_quat_z').pow(2)).sqrt().alias('ang_acc_quat_mag')
            ])
        # --- CORRECTED ANGULAR ACCELERATION CALCULATION END ---

    # Drop gravity columns after all per-row features are calculated
    if gravity_cols_to_drop:
        df_seq = df_seq.drop(gravity_cols_to_drop)

    # Ensure no NaNs/Nulls from new columns
    return df_seq.fill_nan(0.0).fill_null(0.0)

def _get_basic_stats_features(df_seq: pl.DataFrame, cols: list) -> dict:
    """# --- Calculates Global Statistical Features (Mean, Std, Min, Max, Median, Range, Skew, Kurtosis)."""
    features = {}
    for col in cols:
        if df_seq.select(pl.col(col).is_null().all()).item() or df_seq.height == 0:
            for s in ['mean', 'std', 'min', 'max', 'median', 'range', 'skew', 'kurtosis']:
                features[f'{col}_{s}'] = 0.0
        else:
            data_np = get_pl_data_as_np(df_seq, col)
            features[f'{col}_mean'] = (df_seq.select(pl.col(col).mean()).item() or 0.0)
            features[f'{col}_std'] = (df_seq.select(pl.col(col).std()).fill_null(0.0).item() or 0.0)
            features[f'{col}_min'] = (df_seq.select(pl.col(col).min()).item() or 0.0)
            features[f'{col}_max'] = (df_seq.select(pl.col(col).max()).item() or 0.0)
            features[f'{col}_median'] = (df_seq.select(pl.col(col).median()).item() or 0.0)
            features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
            features[f'{col}_skew'] = skew(data_np) if len(data_np) > 1 and np.std(data_np) > 1e-9 else 0.0
            features[f'{col}_kurtosis'] = kurtosis(data_np) if len(data_np) > 1 and np.std(data_np) > 1e-9 else 0.0
            del data_np
    return features

def _get_fft_features(df_seq: pl.DataFrame, cols: list, sampling_rate: int) -> dict:
    """Calculates dominant FFT amplitude and frequency."""
    features = {}
    for col in cols:
        data_for_fft = get_pl_data_as_np(df_seq, col)
        if len(data_for_fft) == 0 or np.all(data_for_fft == 0.0):
            features[f'fft_amp_{col}'] = 0.0
            features[f'fft_freq_{col}'] = 0.0
        else:
            fft_vals = rfft(data_for_fft)
            fft_freq = rfftfreq(len(data_for_fft), 1.0 / sampling_rate)
            if len(fft_vals) > 0 and np.any(np.abs(fft_vals) > 0):
                dominant_freq_idx = np.argmax(np.abs(fft_vals))
                features[f'fft_amp_{col}'] = np.abs(fft_vals)[dominant_freq_idx]
                features[f'fft_freq_{col}'] = fft_freq[dominant_freq_idx]
            else:
                features[f'fft_amp_{col}'] = 0.0
                features[f'fft_freq_{col}'] = 0.0
        del data_for_fft
    return features

def _get_autocorrelation_features(df_seq: pl.DataFrame, cols: list) -> dict:
    """Calculates autocorrelation at lag 1."""
    features = {}
    for col in cols:
        data_for_autocorr = get_pl_data_as_np(df_seq, col)
        if len(data_for_autocorr) > 1 and np.std(data_for_autocorr) > 1e-9:
            try:
                autocorr_val = acf(data_for_autocorr, nlags=1, fft=True)[1]
                features[f'{col}_autocorr_lag1'] = autocorr_val if not np.isnan(autocorr_val) else 0.0
            except Exception:
                features[f'{col}_autocorr_lag1'] = 0.0
        else:
            features[f'{col}_autocorr_lag1'] = 0.0
        del data_for_autocorr
    return features

def _get_entropy_features(df_seq: pl.DataFrame, cols: list) -> dict:
    """Calculates sample entropy using the nolds library."""
    # First, check if the library is available at all.
    if not _NOLDS_AVAILABLE:
        return {}

    features = {}
    for col in cols:
        data = get_pl_data_as_np(df_seq, col)
        # Then, perform the regular calculation logic.
        if len(data) > 50 and np.std(data) > 1e-9:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Zero vectors are within tolerance', category=RuntimeWarning)
                    val = nolds.sampen(data, emb_dim=2)
                    features[f'{col}_sample_entropy'] = val if np.isfinite(val) else 0.0
            except Exception:
                features[f'{col}_sample_entropy'] = 0.0
        else:
            features[f'{col}_sample_entropy'] = 0.0
        del data
    return features

def _get_cross_axis_correlation_features(df_seq: pl.DataFrame) -> dict:
    """Calculates cross-axis correlations for IMU-like data."""
    features = {}
    correlation_base_channels = []
    if config.ENABLE_ACC_FEATURES: correlation_base_channels.extend(['acc_x', 'acc_y', 'acc_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z', 'jerk_x', 'jerk_y', 'jerk_z'])
    if config.ENABLE_ROT_FEATURES: correlation_base_channels.extend(['rot_x', 'rot_y', 'rot_z', 'ang_vel_quat_x', 'ang_vel_quat_y', 'ang_vel_quat_z'])

    correlation_base_channels = [col for col in correlation_base_channels if col in df_seq.columns and df_seq[col].dtype.is_numeric()]

    for i in range(len(correlation_base_channels)):
        for j in range(i + 1, len(correlation_base_channels)):
            col1_name = correlation_base_channels[i]
            col2_name = correlation_base_channels[j]
            col1_data_np = get_pl_data_as_np(df_seq, col1_name)
            col2_data_np = get_pl_data_as_np(df_seq, col2_name)
            if len(col1_data_np) > 1 and len(col2_data_np) > 1:
                temp_df_for_corr = pd.DataFrame({col1_name: col1_data_np, col2_name: col2_data_np}).dropna()
                if temp_df_for_corr.shape[0] > 1:
                    corr_val = temp_df_for_corr[col1_name].corr(temp_df_for_corr[col2_name])
                    features[f'{col1_name}_{col2_name}_corr'] = corr_val if not np.isnan(corr_val) else 0.0
                else:
                    features[f'{col1_name}_{col2_name}_corr'] = 0.0
                del temp_df_for_corr
            else:
                features[f'{col1_name}_{col2_name}_corr'] = 0.0
            del col1_data_np, col2_data_np
    return features

def _get_missing_indicator_features(df_seq: pl.DataFrame) -> dict:
    """
    Calculates inactivity indicators for TOF and THM sensor groups.
    An indicator is True if a sensor group's values are all null or near-zero.
    """
    features = {}
    sensor_groups_to_check = {}

    # Define which sensor groups to check for inactivity
    if config.ENABLE_TOF_FEATURES:
        for i in range(1, 6): # TOF sensors 1-5
            tof_group_cols = [col for col in df_seq.columns if col.startswith(f'tof_{i}_v')]
            if tof_group_cols:
                sensor_groups_to_check[f'tof_{i}'] = tof_group_cols
    if config.ENABLE_THM_FEATURES:
        for i in range(1, 6): # THM sensors 1-5
            thm_col = f'thm_{i}'
            if thm_col in df_seq.columns:
                sensor_groups_to_check[f'thm_{i}'] = [thm_col]

    # Calculate the inactivity feature for each defined group
    for group_name, cols_in_group in sensor_groups_to_check.items():
        existing_cols = [col for col in cols_in_group if col in df_seq.columns and df_seq[col].dtype.is_numeric()]
        if existing_cols:
            # Check if sum of absolute values is near zero (constant zero readings)
            sum_of_abs_vals = (df_seq.select(pl.sum_horizontal([pl.col(c).abs() for c in existing_cols]).sum()).item() or 0.0)
            is_abs_sum_near_zero = sum_of_abs_vals < 1e-6

            # Check if all columns in the group are entirely null
            is_any_col_entirely_null = any([df_seq.select(pl.col(c).is_null().all()).item() for c in existing_cols])

            is_inactive = is_abs_sum_near_zero or is_any_col_entirely_null
            features[f'{group_name}_is_inactive'] = float(is_inactive)
        else:
            features[f'{group_name}_is_inactive'] = 0.0

    return features

def _get_windowed_stats_features(df_seq: pl.DataFrame, cols: list) -> dict:
    """
    Calculates aggregated statistics (mean, std, min, max, median)
    over various sliding windows for a given list of columns.
    """
    features = {}
    window_sizes = [10, 25, 50, 75]

    for col in cols:
        # Pre-check if the sequence is long enough and has variance
        if df_seq.height > max(window_sizes) and (df_seq.select(pl.col(col).std()).item() or 0.0) > 1e-9:
            for win_size in window_sizes:
                if df_seq.height >= win_size:
                    # Use Polars' rolling operations
                    rolling_means = df_seq.select(pl.col(col).rolling_mean(window_size=win_size, min_samples=1)).to_series().drop_nulls()
                    rolling_stds = df_seq.select(pl.col(col).rolling_std(window_size=win_size, min_samples=1)).to_series().drop_nulls()
                    rolling_mins = df_seq.select(pl.col(col).rolling_min(window_size=win_size, min_samples=1)).to_series().drop_nulls()
                    rolling_maxs = df_seq.select(pl.col(col).rolling_max(window_size=win_size, min_samples=1)).to_series().drop_nulls()
                    rolling_medians = df_seq.select(pl.col(col).rolling_median(window_size=win_size, min_samples=1)).to_series().drop_nulls()

                    # Aggregate the rolling statistics
                    features[f'{col}_w{win_size}_mean_agg'] = (rolling_means.mean() or 0.0) if rolling_means.len() > 0 else 0.0
                    features[f'{col}_w{win_size}_std_agg'] = (rolling_stds.mean() or 0.0) if rolling_stds.len() > 0 else 0.0
                    features[f'{col}_w{win_size}_min_agg'] = (rolling_mins.min() or 0.0) if rolling_mins.len() > 0 else 0.0
                    features[f'{col}_w{win_size}_max_agg'] = (rolling_maxs.max() or 0.0) if rolling_maxs.len() > 0 else 0.0
                    features[f'{col}_w{win_size}_median_agg'] = (rolling_medians.median() or 0.0) if rolling_medians.len() > 0 else 0.0
                else:
                    # Sequence is too short for this window size, fill with zeros
                    for suffix in ['mean_agg', 'std_agg', 'min_agg', 'max_agg', 'median_agg']:
                        features[f'{col}_w{win_size}_{suffix}'] = 0.0
        else:
            # Sequence is too short or has no variance, fill all window features with zeros
            for win_size in window_sizes:
                for suffix in ['mean_agg', 'std_agg', 'min_agg', 'max_agg', 'median_agg']:
                    features[f'{col}_w{win_size}_{suffix}'] = 0.0
    return features


def _get_tof_dimensionality_reduction_features(df_seq: pl.DataFrame) -> dict:
    """
    Applies dimensionality reduction (PCA or UMAP) to the average TOF signal shape.
    This version includes a full, working UMAP implementation.
    """
    features = {}
    tof_sensor_groups = {f'tof_{i}': [f'tof_{i}_v{j}' for j in range(64)] for i in range(1, 6)}

    for group_name, cols_in_group in tof_sensor_groups.items():
        existing_tof_bins = [col for col in cols_in_group if col in df_seq.columns]

        # Check if there's enough data to process
        if len(existing_tof_bins) == 64 and df_seq.height > 1:
            try:
                # CRITICAL STEP: We compute the average signal shape across time FIRST.
                # This gives us a single (1, 64) vector representing the gesture's TOF profile.
                tof_signal_shape = df_seq.select(existing_tof_bins).mean().to_numpy(allow_copy=True)
                
                # Check for constant data (no variance) which would cause DR to fail
                if np.std(tof_signal_shape) < 1e-9:
                    raise ValueError("TOF signal has no variance.")

                if config.TOF_DR_METHOD == 'pca':
                    reducer = PCA(n_components=config.TOF_DR_COMPONENTS, random_state=config.RANDOM_STATE)
                    reduced_components = reducer.fit_transform(tof_signal_shape)
                
                elif config.TOF_DR_METHOD == 'umap':
                    # This is the new, complete UMAP implementation
                    reducer = umap.UMAP(
                        n_components=config.TOF_DR_COMPONENTS, 
                        n_neighbors=min(15, df_seq.height - 1), # n_neighbors must be smaller than sample size
                        min_dist=0.1,
                        metric='euclidean',
                        random_state=config.RANDOM_STATE
                    )
                    # UMAP, like PCA, works on the average signal shape
                    full_tof_data = df_seq.select(existing_tof_bins).to_numpy(allow_copy=True)
                    reduced_components = reducer.fit_transform(full_tof_data)
                    # We take the mean of the components over time to get a static feature
                    reduced_components = np.mean(reduced_components, axis=0, keepdims=True)

                # Store the resulting components as features
                for i in range(config.TOF_DR_COMPONENTS):
                    features[f"{group_name}_{config.TOF_DR_METHOD}_component_{i+1}"] = reduced_components[0, i]

            except Exception as e:
                # If DR fails for any reason, create zero-filled features to maintain schema
                # print(f"Warning: DR for {group_name} failed: {e}. Filling with zeros.")
                for i in range(config.TOF_DR_COMPONENTS):
                    features[f"{group_name}_{config.TOF_DR_METHOD}_component_{i+1}"] = 0.0
        else:
            # If there isn't enough data, create zero-filled features
            for i in range(config.TOF_DR_COMPONENTS):
                features[f"{group_name}_{config.TOF_DR_METHOD}_component_{i+1}"] = 0.0
                
    return features

def _get_tof_shape_features(df_seq: pl.DataFrame) -> dict:
    """
    Extracts features describing the shape of the TOF signal for each of the 5 sensors.
    Calculates number of peaks, peak height/width stats, signal energy, and zero-crossing rate.
    """
    if not config.ENABLE_TOF_FEATURES:
        return {}

    features = {}
    tof_sensor_groups = {f'tof_{i}': [f'tof_{i}_v{j}' for j in range(64)] for i in range(1, 6)}

    for group_name, cols_in_group in tof_sensor_groups.items():
        existing_tof_bins = [col for col in cols_in_group if col in df_seq.columns and df_seq[col].dtype.is_numeric()]

        if existing_tof_bins and df_seq.height > 1:
            # Average the 64 TOF bins across all time steps to get a single 1D signal
            tof_signal_1d = df_seq.select(existing_tof_bins).mean().to_numpy().flatten()
            tof_signal_1d = np.nan_to_num(tof_signal_1d, nan=0.0, posinf=0.0, neginf=0.0)

            if len(tof_signal_1d) > 0 and np.std(tof_signal_1d) > 1e-9:
                peaks, properties = find_peaks(tof_signal_1d, prominence=0.1, width=1)
                features[f'{group_name}_num_peaks'] = len(peaks)

                if len(peaks) > 0:
                    peak_heights = properties.get('peak_heights', np.array([]))
                    peak_widths = properties.get('widths', np.array([]))
                    features[f'{group_name}_peak_heights_mean'] = np.mean(peak_heights) if len(peak_heights) > 0 else 0.0
                    features[f'{group_name}_peak_heights_std'] = np.std(peak_heights) if len(peak_heights) > 0 else 0.0
                    features[f'{group_name}_peak_widths_mean'] = np.mean(peak_widths) if len(peak_widths) > 0 else 0.0
                    features[f'{group_name}_peak_widths_std'] = np.std(peak_widths) if len(peak_widths) > 0 else 0.0
                else:
                    for s in ['peak_heights_mean', 'peak_heights_std', 'peak_widths_mean', 'peak_widths_std']:
                        features[f'{group_name}_{s}'] = 0.0

                features[f'{group_name}_total_signal_energy'] = np.sum(tof_signal_1d**2)
                features[f'{group_name}_zero_crossing_rate'] = ((tof_signal_1d[:-1] * tof_signal_1d[1:]) < 0).sum()
            else:
                # Data is constant or too short, fill with zeros
                for s in ['num_peaks', 'peak_heights_mean', 'peak_heights_std', 'peak_widths_mean', 'peak_widths_std', 'total_signal_energy', 'zero_crossing_rate']:
                    features[f'{group_name}_{s}'] = 0.0
            del tof_signal_1d
        else:
            # No TOF bins or sequence too short, fill with zeros
            for s in ['num_peaks', 'peak_heights_mean', 'peak_heights_std', 'peak_widths_mean', 'peak_widths_std', 'total_signal_energy', 'zero_crossing_rate']:
                features[f'{group_name}_{s}'] = 0.0
    return features

def _get_thm_spatial_temporal_features(df_seq: pl.DataFrame) -> dict:
    """
    Calculates spatial and temporal features for Thermistor sensors,
    including differences between adjacent sensors and cross-correlations.
    """
    if not config.ENABLE_THM_FEATURES:
        return {}

    features = {}
    valid_thm_cols = [col for col in config.THM_COLS if col in df_seq.columns and df_seq[col].dtype.is_numeric()]

    if len(valid_thm_cols) >= 2:
        # Spatial Differences (between adjacent sensors)
        spatial_diffs = []
        for i in range(len(valid_thm_cols) - 1):
            col1 = valid_thm_cols[i]
            col2 = valid_thm_cols[i+1]
            diff_mean = (df_seq.select((pl.col(col1) - pl.col(col2)).abs().mean()).item() or 0.0)
            features[f'{col1}_{col2}_spatial_diff_mean'] = diff_mean
            spatial_diffs.append(diff_mean)

        if spatial_diffs:
            features['thm_spatial_diff_mean_overall'] = np.mean(spatial_diffs)
            features['thm_spatial_diff_std_overall'] = np.std(spatial_diffs)
        else:
            features['thm_spatial_diff_mean_overall'] = 0.0
            features['thm_spatial_diff_std_overall'] = 0.0

        # Cross-correlations between THM sensors
        if df_seq.height > 1:
            thm_df_pd = df_seq.select(valid_thm_cols).to_pandas()
            for i in range(len(valid_thm_cols)):
                for j in range(i + 1, len(valid_thm_cols)):
                    col1_name = valid_thm_cols[i]
                    col2_name = valid_thm_cols[j]
                    corr_val = thm_df_pd[col1_name].corr(thm_df_pd[col2_name])
                    features[f'{col1_name}_{col2_name}_cross_corr'] = corr_val if not np.isnan(corr_val) else 0.0
            del thm_df_pd
        else:
            for i in range(len(valid_thm_cols)):
                for j in range(i + 1, len(valid_thm_cols)):
                    features[f'{valid_thm_cols[i]}_{valid_thm_cols[j]}_cross_corr'] = 0.0
    return features

def _get_interaction_features(df_seq: pl.DataFrame) -> dict:
    """
    Creates interaction features between different sensor types, such as
    cross-correlations (e.g., Acc vs. Thm) and ratios of magnitudes.
    """
    features = {}

    # Cross-sensor correlations
    # Example 1: Acc magnitude vs. Thm1
    if 'acc_mag' in df_seq.columns and 'thm_1' in df_seq.columns and df_seq.height > 1:
        acc_mag_data = get_pl_data_as_np(df_seq, 'acc_mag')
        thm1_data = get_pl_data_as_np(df_seq, 'thm_1')
        temp_df = pd.DataFrame({'acc_mag': acc_mag_data, 'thm_1': thm1_data}).dropna()
        if temp_df.shape[0] > 1:
            corr_val = temp_df['acc_mag'].corr(temp_df['thm_1'])
            features['acc_mag_x_thm1_corr'] = corr_val if not np.isnan(corr_val) else 0.0
        else:
            features['acc_mag_x_thm1_corr'] = 0.0
        del acc_mag_data, thm1_data, temp_df

    # Example 2: Rot magnitude vs. a specific TOF bin
    if 'rot_mag' in df_seq.columns and 'tof_1_v0' in df_seq.columns and df_seq.height > 1:
        rot_mag_data = get_pl_data_as_np(df_seq, 'rot_mag')
        tof_1_v0_data = get_pl_data_as_np(df_seq, 'tof_1_v0')
        temp_df = pd.DataFrame({'rot_mag': rot_mag_data, 'tof_1_v0': tof_1_v0_data}).dropna()
        if temp_df.shape[0] > 1:
            corr_val = temp_df['rot_mag'].corr(temp_df['tof_1_v0'])
            features['rot_mag_x_tof_1_v0_corr'] = corr_val if not np.isnan(corr_val) else 0.0
        else:
            features['rot_mag_x_tof_1_v0_corr'] = 0.0
        del rot_mag_data, tof_1_v0_data, temp_df

    # Ratios of magnitudes
    if 'acc_mag' in df_seq.columns and 'jerk_mag' in df_seq.columns:
        acc_mag_mean = df_seq.select(pl.col('acc_mag').mean()).item() or 0.0
        jerk_mag_mean = df_seq.select(pl.col('jerk_mag').mean()).item() or 0.0
        if jerk_mag_mean > 1e-9: # Avoid division by zero
            features['acc_mag_mean_div_jerk_mag_mean_ratio'] = acc_mag_mean / jerk_mag_mean
        else:
            features['acc_mag_mean_div_jerk_mag_mean_ratio'] = 0.0

    return features

def _get_spectral_features(df_seq: pl.DataFrame, cols: list, sampling_rate: int) -> dict:
    """
    Calculates enhanced frequency domain features like energy in different bands,
    spectral centroid, bandwidth, and rolloff.
    """
    features = {}
    for col in cols:
        data_for_spectral = get_pl_data_as_np(df_seq, col)

        if len(data_for_spectral) == 0 or np.all(data_for_spectral == 0.0):
            for s in ['energy_low', 'energy_mid', 'energy_high', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
                features[f'{col}_{s}'] = 0.0
            continue

        # --- Energy Bands Calculation ---
        try:
            low_pass_cutoff, mid_band_low, mid_band_high, high_pass_cutoff = 5.0, 5.0, 15.0, 15.0
            sos_low = butter(2, low_pass_cutoff, btype='lowpass', output='sos', fs=sampling_rate)
            sos_mid = butter(2, [mid_band_low, mid_band_high], btype='bandpass', output='sos', fs=sampling_rate)
            sos_high = butter(2, high_pass_cutoff, btype='highpass', output='sos', fs=sampling_rate)

            low_filtered = sosfilt(sos_low, data_for_spectral)
            mid_filtered = sosfilt(sos_mid, data_for_spectral)
            high_filtered = sosfilt(sos_high, data_for_spectral)

            features[f'{col}_energy_low'] = np.sum(low_filtered**2) / len(low_filtered) if len(low_filtered) > 0 else 0.0
            features[f'{col}_energy_mid'] = np.sum(mid_filtered**2) / len(mid_filtered) if len(mid_filtered) > 0 else 0.0
            features[f'{col}_energy_high'] = np.sum(high_filtered**2) / len(high_filtered) if len(high_filtered) > 0 else 0.0
        except ValueError: # Catches filter design errors
            for s in ['energy_low', 'energy_mid', 'energy_high']:
                features[f'{col}_{s}'] = 0.0

        # --- Spectral Shape Descriptors ---
        yf = rfft(data_for_spectral)
        xf = rfftfreq(len(data_for_spectral), 1.0 / sampling_rate)
        magnitude_spectrum = np.abs(yf)

        if len(yf) > 0 and np.sum(magnitude_spectrum) > 1e-9:
            features[f'{col}_spectral_centroid'] = np.sum(xf * magnitude_spectrum) / np.sum(magnitude_spectrum)
            centroid = features[f'{col}_spectral_centroid']
            features[f'{col}_spectral_bandwidth'] = np.sqrt(np.sum(((xf - centroid)**2) * magnitude_spectrum) / np.sum(magnitude_spectrum))

            cumulative_energy = np.cumsum(magnitude_spectrum**2)
            total_energy = cumulative_energy[-1]
            if total_energy > 1e-9:
                roll_off_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0][0]
                features[f'{col}_spectral_rolloff'] = xf[roll_off_idx]
            else:
                features[f'{col}_spectral_rolloff'] = 0.0
        else:
            for s in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
                features[f'{col}_{s}'] = 0.0

        del data_for_spectral
    return features

def _get_pca_features(df_seq: pl.DataFrame) -> dict:
    """
    Calculates explained variance ratios from Principal Component Analysis (PCA)
    for predefined groups of IMU sensor data.
    """
    features = {}
    pca_groups = {
        'acc_pca': ['acc_x', 'acc_y', 'acc_z'],
        'lin_acc_pca': ['lin_acc_x', 'lin_acc_y', 'lin_acc_z'],
        'rot_pca': ['rot_x', 'rot_y', 'rot_z'],
        'jerk_pca': ['jerk_x', 'jerk_y', 'jerk_z'],
        'ang_vel_quat_pca': ['ang_vel_quat_x', 'ang_vel_quat_y', 'ang_vel_quat_z']
    }

    for prefix, cols in pca_groups.items():
        # Check if the base sensor type for this PCA group is enabled
        is_acc_group = any(s in prefix for s in ['acc', 'jerk', 'lin_acc'])
        is_rot_group = any(s in prefix for s in ['rot', 'ang_vel_quat'])

        if (is_acc_group and not config.ENABLE_ACC_FEATURES) or (is_rot_group and not config.ENABLE_ROT_FEATURES):
            continue

        existing_cols = [c for c in cols if c in df_seq.columns and df_seq[c].dtype.is_numeric()]

        if len(existing_cols) >= 2: # Need at least 2 dimensions for PCA
            data_for_pca = df_seq.select(existing_cols).to_numpy()
            data_for_pca = np.nan_to_num(data_for_pca, nan=0.0, posinf=0.0, neginf=0.0)

            # Check for constant data, which causes PCA to fail
            if data_for_pca.shape[0] > 1 and not np.all(data_for_pca == data_for_pca[0, :]):
                try:
                    pca = PCA(n_components=min(2, data_for_pca.shape[1]), random_state=config.RANDOM_STATE)
                    pca.fit(data_for_pca)

                    features[f'{prefix}_pca1_var_ratio'] = pca.explained_variance_ratio_[0] if len(pca.explained_variance_ratio_) > 0 else 0.0
                    features[f'{prefix}_pca2_var_ratio'] = pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0.0
                except ValueError:
                    features[f'{prefix}_pca1_var_ratio'] = 0.0
                    features[f'{prefix}_pca2_var_ratio'] = 0.0
            else:
                features[f'{prefix}_pca1_var_ratio'] = 0.0
                features[f'{prefix}_pca2_var_ratio'] = 0.0
            del data_for_pca
        else:
            features[f'{prefix}_pca1_var_ratio'] = 0.0
            features[f'{prefix}_pca2_var_ratio'] = 0.0

    return features

# -------------------------------------------------------------------
# Orchestration functions for feature engineering
# -------------------------------------------------------------------

def _calculate_per_sequence_static_features(df_seq: pl.DataFrame) -> dict:
    """
    Orchestrator that calls all specialized helper functions to generate static features for a sequence.

    Calculates all static (per-sequence) features by calling specialized helper functions.
    This function orchestrates the feature generation process based on global control flags.

    # It correctly reads flags from the config module to decide which helper functions to call.
    """
    new_static_features_dict = {}
    sampling_rate = 50

    # 1. Dynamically determine active time-series channels based on group flags
    imu_channels = []
    if config.ENABLE_ACC_FEATURES:
        imu_channels.extend(config.ACC_COLS)
        if config.ENABLE_GRAVITY_REMOVAL: imu_channels.extend(['lin_acc_x', 'lin_acc_y', 'lin_acc_z'])
        if config.ENABLE_IMU_DERIVED_FEATURES: imu_channels.extend(['acc_mag', 'jerk_x', 'jerk_y', 'jerk_z', 'jerk_mag', 'ang_acc_quat_x', 'ang_acc_quat_y', 'ang_acc_quat_z', 'ang_acc_quat_mag'])

    if config.ENABLE_ROT_FEATURES:
        imu_channels.extend(config.ROT_COLS)
        if config.ENABLE_IMU_DERIVED_FEATURES: imu_channels.extend(['rot_mag', 'ang_vel_quat_x', 'ang_vel_quat_y', 'ang_vel_quat_z', 'ang_dist_quat'])

    thm_channels = [col for col in config.THM_COLS if col in df_seq.columns]
    tof_channels = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64) if f'tof_{i}_v{j}' in df_seq.columns]

    imu_numeric_cols = [col for col in imu_channels if col in df_seq.columns and df_seq[col].dtype.is_numeric()]
    thm_numeric_cols = [col for col in thm_channels if col in df_seq.columns and df_seq[col].dtype.is_numeric()]
    tof_numeric_cols = [col for col in tof_channels if col in df_seq.columns and df_seq[col].dtype.is_numeric()]

    # 2. Call helper functions based on control flags and update the main dictionary
    if config.ENABLE_BASIC_STATS:
        if config.ENABLE_BASIC_STATS_IMU: new_static_features_dict.update(_get_basic_stats_features(df_seq, imu_numeric_cols))
        if config.ENABLE_BASIC_STATS_THM: new_static_features_dict.update(_get_basic_stats_features(df_seq, thm_numeric_cols))
        if config.ENABLE_BASIC_STATS_TOF: new_static_features_dict.update(_get_basic_stats_features(df_seq, tof_numeric_cols))

    if config.ENABLE_DOMINANT_FFT_FEATURES:
        fft_cols = []
        if config.ENABLE_DOMINANT_FFT_FEATURES_IMU: fft_cols.extend([col for col in imu_numeric_cols if col not in config.ROT_COLS])
        if config.ENABLE_DOMINANT_FFT_FEATURES_THM: fft_cols.extend(thm_numeric_cols)
        if config.ENABLE_DOMINANT_FFT_FEATURES_TOF: fft_cols.extend(tof_numeric_cols)
        new_static_features_dict.update(_get_fft_features(df_seq, fft_cols, sampling_rate))

    if config.ENABLE_COMPREHENSIVE_AUTOCORR:
        autocorr_cols = []
        if config.ENABLE_COMPREHENSIVE_AUTOCORR_IMU: autocorr_cols.extend([col for col in imu_numeric_cols if col not in config.ROT_COLS])
        if config.ENABLE_COMPREHENSIVE_AUTOCORR_THM: autocorr_cols.extend(thm_numeric_cols)
        if config.ENABLE_COMPREHENSIVE_AUTOCORR_TOF: autocorr_cols.extend(tof_numeric_cols)
        new_static_features_dict.update(_get_autocorrelation_features(df_seq, autocorr_cols))

    if config.ENABLE_NOLDS_ENTROPY:
        entropy_cols = []
        if config.ENABLE_NOLDS_ENTROPY_IMU: entropy_cols.extend([col for col in imu_numeric_cols if col not in config.ROT_COLS])
        if config.ENABLE_NOLDS_ENTROPY_THM: entropy_cols.extend(thm_numeric_cols)
        if config.ENABLE_NOLDS_ENTROPY_TOF: entropy_cols.extend(tof_numeric_cols)
        new_static_features_dict.update(_get_entropy_features(df_seq, entropy_cols))

    if config.ENABLE_COMPREHENSIVE_FFT_SPECTRAL:
        spectral_cols = []
        if config.ENABLE_COMPREHENSIVE_FFT_SPECTRAL_IMU: spectral_cols.extend([col for col in imu_numeric_cols if col not in config.ROT_COLS])
        if config.ENABLE_COMPREHENSIVE_FFT_SPECTRAL_THM: spectral_cols.extend(thm_numeric_cols)
        if config.ENABLE_COMPREHENSIVE_FFT_SPECTRAL_TOF: spectral_cols.extend(tof_numeric_cols)
        new_static_features_dict.update(_get_spectral_features(df_seq, spectral_cols, sampling_rate))

    if config.ENABLE_COMPREHENSIVE_CORRELATIONS:
        new_static_features_dict.update(_get_cross_axis_correlation_features(df_seq))

    if config.ENABLE_COMPREHENSIVE_PCA:
        new_static_features_dict.update(_get_pca_features(df_seq))

    # --- Missing Data / Inactivity Indicators (for TOF and THM) ---
    if config.ENABLE_COMPREHENSIVE_MISSING_INDICATORS:
        new_static_features_dict.update(_get_missing_indicator_features(df_seq))

    # --- Windowed Statistical Features ---
    if config.ENABLE_COMPREHENSIVE_WINDOWED_STATS:
        window_cols = []
        if config.ENABLE_COMPREHENSIVE_WINDOWED_STATS_IMU: window_cols.extend([col for col in imu_numeric_cols if col not in config.ROT_COLS])
        if config.ENABLE_COMPREHENSIVE_WINDOWED_STATS_THM: window_cols.extend(thm_numeric_cols)
        if config.ENABLE_COMPREHENSIVE_WINDOWED_STATS_TOF: window_cols.extend(tof_numeric_cols)
        new_static_features_dict.update(_get_windowed_stats_features(df_seq, window_cols))

    # --- Advanced Sensor-Specific Features ---
    if config.ENABLE_TOF_DIMENSIONALITY_REDUCTION:
        new_static_features_dict.update(_get_tof_dimensionality_reduction_features(df_seq))

    if config.ENABLE_TOF_SHAPE_FEATURES:
        new_static_features_dict.update(_get_tof_shape_features(df_seq))

    if config.ENABLE_THM_SPATIAL_TEMPORAL_FEATURES:
        new_static_features_dict.update(_get_thm_spatial_temporal_features(df_seq))

    if config.ENABLE_INTERACTION_FEATURES:
        new_static_features_dict.update(_get_interaction_features(df_seq))

    return new_static_features_dict

def _process_single_sequence_for_fe(df_seq_raw: pl.DataFrame, output_dir: str) -> str:
    """
    Applies per-row and per-sequence static feature engineering to a single raw Polars DataFrame sequence.
    Saves the processed sequence to a Parquet file in output_dir and returns the file path.
    """
    current_seq_id = df_seq_raw.select('sequence_id').unique().item()
    # print(f"DEBUG: Worker {os.getpid()} processing sequence_id {current_seq_id} (rows: {df_seq_raw.height})") # Debugging line

    # Validate ID format
    if not isinstance(current_seq_id, str):
        current_seq_id = str(current_seq_id)

    # Ensure proper SEQ_XXXXX format
    if not current_seq_id.startswith('SEQ_'):
        raise ValueError(f"Invalid sequence ID format: {current_seq_id}")

    # ====== CONDITIONAL CLEANING ======
    if isinstance(current_seq_id, str) and current_seq_id.startswith("('") and current_seq_id.endswith("',)"):
        current_seq_id = current_seq_id[2:-3]  # Clean tuple-strings
    # ====== END CLEANING ======

    # Strict format validation
    if not current_seq_id.startswith('SEQ_'):
        raise ValueError(f"Missing SEQ_ prefix in: {current_seq_id}")
    if len(current_seq_id.split('_')) != 2 or not current_seq_id.split('_')[1].isdigit():
        raise ValueError(f"Malformed sequence ID: {current_seq_id}")

    # Skip empty sequences
    if df_seq_raw.height == 0:
        print(f"WARNING: Empty sequence {current_seq_id}")
        return None

    #print(f"Processing sequence_id {current_seq_id} (rows: {df_seq_raw.height})")

    # 1. Calculate per-row derived features
    df_seq_derived = _calculate_per_row_derived_features(df_seq_raw.clone()) # Clone to avoid modifying original in place

    # 2. Calculate per-sequence static features
    static_features_dict = _calculate_per_sequence_static_features(df_seq_derived)

    # Convert static features dictionary to a single-row Polars DataFrame
    static_df_pl_single_row = pl.DataFrame([static_features_dict])

    # Broadcast static features to all rows of the sequence
    # This creates a DataFrame where each row has the same static features
    static_df_pl_broadcasted = pl.concat([static_df_pl_single_row] * df_seq_derived.height, how="vertical") # Use vertical concat for single row broadcast

    # Combine time-series data with static features
    df_seq_final = pl.concat([df_seq_derived, static_df_pl_broadcasted], how="horizontal")

    # --- START OF CHANGE ---
    # Explicitly cast ALL numerical columns to Float32 BEFORE writing.
    # This prevents schema mismatches (Float32 vs Float64) that can happen
    # due to implicit type inference in Polars or differences in calculation outputs.
    # We iterate over all columns, and if they are numeric, we cast them to Float32.
    expressions_to_cast = []
    for col in df_seq_final.columns:
        if df_seq_final[col].dtype.is_numeric(): # Check if it's any numeric type (int, float)
            expressions_to_cast.append(pl.col(col).cast(pl.Float32))
        else:
            expressions_to_cast.append(pl.col(col)) # Keep non-numeric columns as they are

    df_seq_final = df_seq_final.with_columns(expressions_to_cast)
    # --- END OF CHANGE ---

    # Ensure any remaining NaNs/Nulls are filled AFTER explicit casting
    # This is important as some operations might introduce them, or the cast itself.
    df_seq_final = df_seq_final.fill_nan(0.0).fill_null(0.0)

    # At the very end, check if we should drop the raw TOF columns
    if config.DROP_RAW_TOF_FEATURES:
        # Get a list of raw TOF columns that exist in the dataframe
        raw_tof_cols_to_drop = [col for col in df_seq_final.columns if col in config.TOF_COLS]
        if raw_tof_cols_to_drop:
            df_seq_final = df_seq_final.drop(raw_tof_cols_to_drop)

    # Save the processed sequence to a permanent Parquet file and return its path
    output_file_path = os.path.join(output_dir, f"seq_{current_seq_id}_features.parquet")
    df_seq_final.write_parquet(output_file_path)

    del df_seq_raw, df_seq_derived, static_features_dict, static_df_pl_single_row, static_df_pl_broadcasted # Free memory
    gc.collect()

    return output_file_path

def get_all_feature_columns(df_pl: pl.DataFrame) -> list[str]:
    """Dynamically gets all numerical feature columns from the DataFrame, excluding identifiers."""
    excluded_cols = ['sequence_id', 'subject', 'gesture', 'gesture_encoded', 'row_id', 'sequence_counter', 'step']
    feature_cols = [col for col in df_pl.columns if df_pl[col].dtype.is_numeric() and col not in excluded_cols]
    feature_cols.sort()
    return feature_cols