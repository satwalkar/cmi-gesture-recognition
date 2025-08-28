# analysis_tools.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import shap
import config
import tensorflow as tf

def generate_and_save_confusion_matrix(oof_true_labels, oof_preds_labels, class_names, output_dir):
    """
    Computes, plots, and saves a normalized confusion matrix.

    Args:
        oof_true_labels (np.ndarray): 1D array of true integer labels.
        oof_preds_labels (np.ndarray): 1D array of predicted integer labels.
        class_names (list): List of string names for the classes, in order.
        output_dir (str): Directory to save the output plot and Excel file.
    """
    print("Generating and saving confusion matrix...")
    cm = confusion_matrix(oof_true_labels, oof_preds_labels, normalize='true')

    # Plotting
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", cbar=True,
                     xticklabels=class_names, yticklabels=class_names)
    ax.set_title('Normalized Confusion Matrix (OOF Predictions)', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Confusion matrix plot saved to: {plot_path}")
    # Saving to Excel
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    excel_path = os.path.join(output_dir, "confusion_matrix.xlsx")
    try:
        df_cm.to_excel(excel_path)
        print(f"Normalized confusion matrix saved to: {excel_path}")
    except ImportError:
        print("\nWarning: `openpyxl` is not installed. Skipping save to Excel.")
    except Exception as e:
        print(f"An unexpected error occurred while saving confusion matrix to Excel: {e}")

def calculate_permutation_importance(model, ts_data, demo_data, y_true, ts_feature_names, demo_feature_names):
    """
    Calculates permutation importance for a Keras model that can be single or multi-input.
    """
    print("\nStarting granular permutation importance calculation...")
    importance = {}

    # --- 1. Get baseline accuracy ---
    # Conditionally define the inputs for the model based on whether demo_data exists
    if demo_data.size > 0:
        baseline_inputs = {'time_series_input': ts_data, 'demographics_input': demo_data}
    else:
        baseline_inputs = ts_data

    baseline_preds = model.predict(baseline_inputs, verbose=0)
    baseline_accuracy = np.mean(np.argmax(baseline_preds, axis=1) == np.argmax(y_true, axis=1))
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")

    # --- 2. Permute each time-series feature ---
    for i, feature_name in enumerate(ts_feature_names):
        shuffled_ts_data = ts_data.copy()
        np.random.shuffle(shuffled_ts_data[:, :, i].reshape(-1))

        # Use the same conditional logic for the predict call
        if demo_data.size > 0:
            shuffled_inputs = {'time_series_input': shuffled_ts_data, 'demographics_input': demo_data}
        else:
            shuffled_inputs = shuffled_ts_data

        shuffled_preds = model.predict(shuffled_inputs, verbose=0)
        shuffled_accuracy = np.mean(np.argmax(shuffled_preds, axis=1) == np.argmax(y_true, axis=1))
        importance[feature_name] = baseline_accuracy - shuffled_accuracy

    # --- 3. Permute each demographics feature (if they exist) ---
    if demo_data.size > 0:
        for i, feature_name in enumerate(demo_feature_names):
            shuffled_demo_data = demo_data.copy()
            np.random.shuffle(shuffled_demo_data[:, i])

            # Use the same conditional logic for the predict call
            shuffled_inputs = {'time_series_input': ts_data, 'demographics_input': shuffled_demo_data}

            shuffled_preds = model.predict(shuffled_inputs, verbose=0)
            shuffled_accuracy = np.mean(np.argmax(shuffled_preds, axis=1) == np.argmax(y_true, axis=1))
            importance[feature_name] = baseline_accuracy - shuffled_accuracy

    # --- 4. Convert results to a DataFrame ---
    importance_df = pd.DataFrame(
        list(importance.items()),
        columns=['Feature', 'Importance (Accuracy Drop)']
    ).sort_values(by='Importance (Accuracy Drop)', ascending=False)

    return importance_df

def calculate_and_plot_permutation_importance(model, val_data, ts_feature_names, demo_feature_names, output_dir):
    """
    Calculates and plots permutation importance for a multi-input model.

    Args:
        model (tf.keras.Model): The trained Keras model.
        val_data (tuple): A tuple containing (sample_ts_data, sample_demo_data, sample_labels).
        ts_feature_names (list): List of names for the time-series features.
        demo_feature_names (list): List of names for the demographic features.
        output_dir (str): Directory to save the output plot.
    """
    print("\n" + "="*50)
    print("--- Permutation Importance Analysis ---")
    print("="*50)

    sample_ts_data, sample_demo_data, sample_labels = val_data

    importance_df = calculate_permutation_importance(
        model, sample_ts_data, sample_demo_data, sample_labels, ts_feature_names, demo_feature_names
    )

    print("\nTop 40 Permutation Importance Scores (Accuracy Drop):")
    print(importance_df.head(40))

    # Plotting the results
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance (Accuracy Drop)', y='Feature', data=importance_df.head(40))
    plt.title('Permutation Importance - Top 40 Features')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "permutation_importance.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Permutation importance plot saved to: {plot_path}")

def calculate_and_plot_shap_values(model, train_data, val_data, ts_feature_names, demo_feature_names, output_dir):
    """
    Calculates and plots SHAP summary plots.
    This version includes the corrected reshaping logic for KernelExplainer.
    """
    print("\n" + "="*50)
    print(f"--- SHAP Values Analysis (Using '{config.SHAP_EXPLAINER_TYPE}' explainer) ---")
    print("="*50)

    background_ts, background_demo = train_data
    explain_ts, explain_demo = val_data

    try:
        shap_values_ts = None
        shap_values_demo = None

        if config.SHAP_EXPLAINER_TYPE == 'deep':
            if background_demo.size > 0:
                explainer_inputs = [tf.convert_to_tensor(d, dtype=tf.float32) for d in [background_ts, background_demo]]
                explain_inputs = {'time_series_input': tf.convert_to_tensor(explain_ts, dtype=tf.float32),
                                  'demographics_input': tf.convert_to_tensor(explain_demo, dtype=tf.float32)}
            else:
                explainer_inputs = tf.convert_to_tensor(background_ts, dtype=tf.float32)
                explain_inputs = tf.convert_to_tensor(explain_ts, dtype=tf.float32)

            explainer = shap.DeepExplainer(model, explainer_inputs)
            shap_values = explainer.shap_values(explain_inputs)

            # --- THIS IS THE FIX ---
            # Correctly unpack the results from DeepExplainer
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values_ts, shap_values_demo = shap_values[0], shap_values[1]
            else:
                shap_values_ts = shap_values
            # ------------------------

        elif config.SHAP_EXPLAINER_TYPE == 'kernel':
            def predict_wrapper(X_flattened):
                X_reshaped = X_flattened.reshape(-1, config.MAX_SEQUENCE_LENGTH, len(ts_feature_names))
                # The model's predict function should be passed to the explainer
                # so we need a predict function that matches the model's output shape
                return model(X_reshaped)

            background_ts_2d = background_ts.reshape(background_ts.shape[0], -1)
            explain_ts_2d = explain_ts.reshape(explain_ts.shape[0], -1)

            summary_data_2d = shap.kmeans(background_ts_2d, 10)
            explainer = shap.KernelExplainer(predict_wrapper, summary_data_2d)
            shap_values_flat = explainer.shap_values(explain_ts_2d[:10])

            # --- THIS IS THE FIX ---
            shap_values_np = np.array(shap_values_flat)
            # shap_values_np shape is (num_classes, num_samples, flattened_features)
            # The KernelExplainer for multi-output models returns a list of arrays,
            # so the shape should be (num_classes, num_samples, flattened_features)

            # Correctly reshape and transpose the array
            num_classes, num_samples, _ = shap_values_np.shape

            shap_values_ts = shap_values_np.reshape(
                num_classes,
                num_samples,
                config.MAX_SEQUENCE_LENGTH,
                len(ts_feature_names)
            )

        # --- Plotting Logic (now works for both explainers) ---
        if demo_feature_names and shap_values_demo is not None:
            # ... (demographics plotting logic) ...
            avg_abs_shap = np.mean(np.abs(shap_values_demo), axis=(0, 1))
            shap_importance_df = pd.DataFrame({
                'Feature': demo_feature_names,
                'Mean Absolute SHAP': avg_abs_shap
            }).sort_values(by='Mean Absolute SHAP', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Mean Absolute SHAP', y='Feature', data=shap_importance_df)
            plt.title('Overall Feature Importance (SHAP) - Demographics')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_demographics_importance.png"))
            plt.show()

        if ts_feature_names and shap_values_ts is not None:
            avg_abs_shap_ts = np.mean(np.abs(shap_values_ts), axis=(0, 1, 2))
            shap_importance_df_ts = pd.DataFrame({
                'Feature': ts_feature_names,
                'Mean Absolute SHAP': avg_abs_shap_ts
            }).sort_values(by='Mean Absolute SHAP', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Mean Absolute SHAP', y='Feature', data=shap_importance_df_ts.head(40))
            plt.title('Overall Feature Importance (SHAP) - Top 40 Time-Series')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_timeseries_importance.png"))
            plt.show()

    except Exception as e:
        print(f"Error during SHAP value calculation: {e}")
        print("SHAP analysis skipped.")