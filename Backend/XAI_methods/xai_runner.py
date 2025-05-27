import datetime
import json
import sys
import warnings
import pandas as pd
import numpy as np
import os
import time
from typing import List, Optional, Dict, Any
import traceback

# Custom / Third-party imports needed for XAI logic
from ML_models.model_wrapper import ModelWrapperForXAI
from XAI_methods.timeseriesExplainer import TimeSeriesExplainer
from XAI_methods import xai_visualizations as x
import utils as ut

# Constants or Configurations
OUTPUT_DIR = "/data"
MAX_BG_SAMPLES = 250000 # Example default, can be overridden

class XAIRunner:
    """
    Handles the execution of Explainable AI (XAI) methods for time series models.
    """
    def __init__(
        self,
        xai_settings: Dict[str, Any],
        model_wrapper: ModelWrapperForXAI,
        sequence_length: int,
        feature_columns: List[str],
        actual_label_col: str,
        continuous_features_list: List[str],
        job_name: str,
        mode: str = 'classification', # Default mode
        output_dir: str = OUTPUT_DIR,
        # --- Arguments for NDCG ---
        inj_params: Optional[List[List[Dict[str, Any]]]] = None, # Injected anomaly parameters
        timestamp_col_name: str = 'timestamp', # Name of the timestamp column
    ):
        """
        Initializes the XAIRunner.

        Args:
            xai_params (List[Dict[str, Any]]): List of configurations for each XAI method.
            model_wrapper (ModelWrapperForXAI): Wrapped model instance ready for XAI.
            sequence_length (int): The sequence length used by the model and for XAI.
            feature_columns (List[str]): List of base feature names.
            actual_label_col (str): Name of the column containing the true labels.
            continuous_features_list (List[str]): List of continuous feature names (for DiCE).
            job_name (str): Identifier for the current job (used for saving outputs).
            mode (str): Type of problem ('classification' or 'regression'). Defaults to 'classification'.
            output_dir (str): Directory to save XAI plots and results.
        """
        if not isinstance(xai_settings, dict):
            raise TypeError("xai_settings must be a dictionary wit settings and list of dictionaries.")
        if not model_wrapper:
            raise ValueError("model_wrapper cannot be None.")
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            # Even non-sequential models need a sequence context for XAI framework
            raise ValueError("sequence_length must be a positive integer for XAI processing.")

        self.xai_params = xai_settings.get("xai_params", None)
        self.model_wrapper = model_wrapper
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        self.actual_label_col = actual_label_col
        self.continuous_features_list = continuous_features_list
        self.job_name = job_name
        self.mode = mode
        self.output_dir = output_dir
        
        self.xai_settings_global = xai_settings.get("xai_settings")
        self.xai_sampling_strategy = xai_settings.get("xai_sampling_strategy", "random")
        self.xai_sample_seed = xai_settings.get("xai_sample_seed", None)
        
        self.all_aggregated_scores_for_ndcg: Dict[str, List[Dict[str, float]]] = {} # Method -> List of [Scores_Dict_for_Instance_1, Scores_Dict_for_Instance_2, ...]

        # --- For NDCG ---
        self.inj_params = inj_params
        self.timestamp_col_name = timestamp_col_name
        self.ndcg_k_values = xai_settings.get("ndcg_k_values", [3, 5, 10])
        self.ndcg_results: Dict[str, Dict[int, List[float]]] = {}
        self.ndcg_ground_truth_found_count = 0
        self.ndcg_anomalies_explained_count = 0
        
        # --- For Individual XAI Method Timings ---
        self.xai_method_timings: Dict[str, float] = {}
        
        # # print(f"DEBUG XAIRunner __init__: Job '{self.job_name}'")
        # # print(f"DEBUG XAIRunner __init__: inj_params received: {bool(self.inj_params)}")
        # if self.inj_params:
            # # print(f"DEBUG XAIRunner __init__: First inj_param group (sample): {self.inj_params[0][:1] if self.inj_params and self.inj_params[0] else 'Empty group'}")
        # # print(f"DEBUG XAIRunner __init__: timestamp_col_name: '{self.timestamp_col_name}'")
        # # print(f"DEBUG XAIRunner __init__: actual_label_col: '{self.actual_label_col}'")
        # # print(f"DEBUG XAIRunner __init__: feature_columns: {self.feature_columns[:5]}...") # Print first 5 features
        
        os.makedirs(self.output_dir, exist_ok=True)
        # print(f"XAIRunner initialized for job '{self.job_name}'. Output directory: '{self.output_dir}'")

    def _get_ground_truth_features_for_instance(
        self,
        instance_original_df_idx: int, # Index in source_df
        source_df: pd.DataFrame        # This is data_source_for_explanation
    ) -> List[str]:
        # print(f"DEBUG NDCG_GT: Called for instance_orig_df_idx: {instance_original_df_idx}"

        if self.inj_params is None or not self.inj_params:
            # # print(f"DEBUG NDCG_GT: Exiting - self.inj_params is None or empty.")
            return []

        try:
            if instance_original_df_idx >= len(source_df):
                # print(f"DEBUG NDCG_GT: Exiting - instance_original_df_idx {instance_original_df_idx} is out of bounds for source_df (len {len(source_df)}).")
                return []

            instance_label = source_df.iloc[instance_original_df_idx][self.actual_label_col]
            is_anomaly = (str(instance_label) == '1' or instance_label is True or instance_label == 1)
            
            # print(f"DEBUG NDCG_GT: Instance_idx={instance_original_df_idx}, Label='{instance_label}', IsAnomaly={is_anomaly}")

            if not is_anomaly:
                # # # print(f"DEBUG NDCG_GT: Instance_idx={instance_original_df_idx} is NOT an anomaly. Skipping GT search.") # Less verbose for non-anomalies
                return []
            
            self.ndcg_anomalies_explained_count +=1

            if self.timestamp_col_name not in source_df.columns:
                # print(f"DEBUG NDCG_GT: Exiting - Timestamp column '{self.timestamp_col_name}' not in source_df. Columns: {source_df.columns.tolist()}")
                return []
            
            instance_timestamp_val = source_df.iloc[instance_original_df_idx][self.timestamp_col_name]
            # print(f"DEBUG NDCG_GT: Instance_idx={instance_original_df_idx}, Raw TS Value='{instance_timestamp_val}', Type={type(instance_timestamp_val)}")

            if not isinstance(instance_timestamp_val, pd.Timestamp):
                # print(f"DEBUG NDCG_GT: Exiting - Instance timestamp is not pd.Timestamp.")
                return []
            
            current_instance_ts_utc = instance_timestamp_val.tz_convert('UTC') if instance_timestamp_val.tzinfo else instance_timestamp_val.tz_localize('UTC')

            if source_df.empty:
                # print(f"DEBUG NDCG_GT: Exiting - source_df is empty.")
                return []
            
            base_timestamp_from_data = source_df[self.timestamp_col_name].min()
            if not isinstance(base_timestamp_from_data, pd.Timestamp):
                 # print(f"DEBUG NDCG_GT: Exiting - Base timestamp from data is not pd.Timestamp. Type: {type(base_timestamp_from_data)}")
                 return []
            base_ts_utc = base_timestamp_from_data.tz_convert('UTC') if base_timestamp_from_data.tzinfo else base_timestamp_from_data.tz_localize('UTC')
            
            # print(f"DEBUG NDCG_GT: Instance_TS_UTC={current_instance_ts_utc}, Base_TS_UTC={base_ts_utc}")

            for group_idx, anomaly_group in enumerate(self.inj_params):
                # print(f"DEBUG NDCG_GT: Processing inj_params group {group_idx}")
                for setting_idx, inj_setting in enumerate(anomaly_group):
                    # print(f"DEBUG NDCG_GT:  Inj_setting {setting_idx}: {inj_setting}")
                    inj_ts_offset_str = inj_setting.get("timestamp")
                    inj_duration_str = inj_setting.get("duration")
                    inj_columns = inj_setting.get("columns")

                    if inj_ts_offset_str is None or inj_duration_str is None or inj_columns is None:
                        # print(f"DEBUG NDCG_GT:   Skipping inj_setting {setting_idx} due to missing fields.")
                        continue
                    
                    try:
                        offset_seconds = float(inj_ts_offset_str)
                        duration_seconds = ut.parse_duration_to_seconds(inj_duration_str)
                        
                        if duration_seconds is None: 
                            # print(f"DEBUG NDCG_GT:   Could not parse duration '{inj_duration_str}' for inj_setting {setting_idx}. Skipping.")
                             continue
                        
                        inj_ts_start_actual = base_ts_utc + pd.Timedelta(seconds=offset_seconds)
                        inj_ts_end_actual = inj_ts_start_actual + pd.Timedelta(seconds=duration_seconds)
                        # print(f"DEBUG NDCG_GT:   Calculated Inj Window: Start={inj_ts_start_actual}, End={inj_ts_end_actual}")

                        if inj_ts_start_actual <= current_instance_ts_utc < inj_ts_end_actual:
                            # # print(f"DEBUG NDCG_GT:   MATCH FOUND for instance {instance_original_df_idx}! Inj Cols={inj_columns}")
                            self.ndcg_ground_truth_found_count +=1
                            return inj_columns
                        # else:
                            # print(f"DEBUG NDCG_GT:   No match for instance {instance_original_df_idx} with this inj_setting.")

                    except ValueError as ve:
                        # print(f"DEBUG NDCG_GT:   ValueError processing inj_setting {setting_idx} timestamp_offset or duration: {ve}. Skipping.")
                        continue
                    except Exception as e:
                        # print(f"DEBUG NDCG_GT:   Error processing inj_setting {setting_idx}: {e}. Skipping.")
                        continue
            # print(f"DEBUG NDCG_GT: No matching injection found for instance {instance_original_df_idx} after checking all inj_params.")
            return []
        except Exception as e:
            # print(f"DEBUG NDCG_GT: CRITICAL ERROR in _get_ground_truth_features_for_instance for index {instance_original_df_idx}: {e}")
            traceback.print_exc()
            return []
        
    def _extract_aggregated_feature_scores(
        self,
        xai_result_for_one_instance: Any, 
        method_name: str,
        original_instance_sequence_np: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Aggregates feature importance scores from XAI result for one instance
        to base feature names.
        """
        aggregated_scores: Dict[str, float] = {}
        if xai_result_for_one_instance is None:
            return aggregated_scores

        if method_name == "ShapExplainer":
            # xai_result_for_one_instance is np.ndarray of shape (seq_len, n_features)
            # Or if multi-class, it might be one of the elements of a list.
            # We assume it's already the correct slice for the class of interest.
            if isinstance(xai_result_for_one_instance, np.ndarray) and \
               xai_result_for_one_instance.ndim == 2 and \
               xai_result_for_one_instance.shape[0] == self.sequence_length and \
               xai_result_for_one_instance.shape[1] == len(self.feature_columns):
                
                abs_shap_values = np.abs(xai_result_for_one_instance)
                # Sum importances across time steps for each base feature
                for i, base_feat_name in enumerate(self.feature_columns):
                    aggregated_scores[base_feat_name] = np.sum(abs_shap_values[:, i])
            else:
                warnings.warn(f"Unexpected SHAP result format for instance. Cannot aggregate for NDCG. Shape: {getattr(xai_result_for_one_instance, 'shape', 'N/A')}", RuntimeWarning)

        elif method_name == "LimeExplainer":
            # xai_result_for_one_instance is a lime.explanation.Explanation object
            try:
                # .as_list() gives (flat_feature_name, weight)
                for flat_feat_name, weight in xai_result_for_one_instance.as_list():
                    # Extract base feature name (e.g., from "FeatureA_t-1")
                    base_feat_name = flat_feat_name.split('_t-')[0]
                    if base_feat_name in self.feature_columns:
                        aggregated_scores.setdefault(base_feat_name, 0.0)
                        aggregated_scores[base_feat_name] += abs(weight)
            except Exception as e:
                warnings.warn(f"Error processing LIME explanation for NDCG: {e}", RuntimeWarning)
        
        elif method_name == "DiceExplainer":
            # xai_result_for_one_instance is a dice_ml.counterfactual_explanations.CounterfactualExplanations object
            # It contains CFs for ONE original instance.
            # original_instance_sequence_np should be the (1, seq_len, n_features) numpy array for that instance
            
            if original_instance_sequence_np is None or original_instance_sequence_np.shape[0] != 1:
                warnings.warn("DiCE NDCG: original_instance_sequence_np not provided or incorrect shape. Skipping score extraction.", RuntimeWarning)
                return aggregated_scores

            if not hasattr(xai_result_for_one_instance, 'cf_examples_list') or \
               not xai_result_for_one_instance.cf_examples_list:
                # warnings.warn("DiCE NDCG: No counterfactual examples found in the result. Skipping score extraction.", RuntimeWarning)
                # # print(f"DEBUG DiCE NDCG: No CFs found for this instance.") # Less noisy if common
                return aggregated_scores

            # Flatten original instance for comparison: (1, seq_len * n_features)
            original_flat = original_instance_sequence_np.reshape(1, -1)

            feature_change_counts: Dict[str, int] = {feat_name: 0 for feat_name in self.feature_columns}

            for cf_example in xai_result_for_one_instance.cf_examples_list:
                # cf_example.final_cfs_df contains the counterfactuals as a DataFrame
                # It has flattened feature names (e.g., V1_t-0, V1_t-1, ..., V_N_t-M)
                if cf_example.final_cfs_df is None or cf_example.final_cfs_df.empty:
                    continue

                for _, cf_row_series in cf_example.final_cfs_df.iterrows():
                    
                    # Regenerate flat feature names used by DiCE
                    # This assumes DiceExplainer.py uses this convention:
                    dice_flat_feature_names = []
                    for t in range(self.sequence_length):
                        time_suffix = f"_t-{self.sequence_length - 1 - t}" # e.g. _t-9 ... _t-0
                        for base_feat in self.feature_columns:
                            dice_flat_feature_names.append(base_feat + time_suffix)
                    
                    # Compare each *base feature* by checking if any of its time steps changed
                    for base_feat_idx, base_feat_name in enumerate(self.feature_columns):
                        base_feature_changed_in_cf = False
                        for t_step in range(self.sequence_length):
                            original_idx_flat = base_feat_idx + (t_step * len(self.feature_columns)) # Index in original_flat
                            
                            # Construct the flat feature name for DiCE output matching this time step and base feature
                            # Example: base_feat_name='V1', t_step=0, seq_len=3 -> 'V1_t-2' (if t-0 is most recent)
                            # Let's use the dice_flat_feature_names generated above for lookup
                            flat_feat_name_in_dice = self.feature_columns[base_feat_idx] + f"_t-{self.sequence_length - 1 - t_step}"

                            if flat_feat_name_in_dice not in cf_row_series.index:
                                # This can happen if features_to_vary was used and this feature wasn't varied
                                # Or if the DiCE output columns don't perfectly match our generated flat names
                                # warnings.warn(f"DiCE NDCG: Flat feature {flat_feat_name_in_dice} not in CF. This is okay if it wasn't varied.")
                                continue # Cannot compare if not in CF

                            original_val = original_flat[0, original_idx_flat]
                            cf_val = cf_row_series[flat_feat_name_in_dice]
                            
                            # Check for significant change (handles potential float precision issues)
                            if not np.isclose(original_val, cf_val):
                                base_feature_changed_in_cf = True
                                break # This base feature changed, no need to check other time steps for it
                        
                        if base_feature_changed_in_cf:
                            feature_change_counts[base_feat_name] += 1
            
            # Convert counts to scores (higher count = higher importance)
            for feat_name, count in feature_change_counts.items():
                aggregated_scores[feat_name] = float(count)
        
        return aggregated_scores

    def run_explanations(
        self,
        training_features_df: pd.DataFrame, # Used for background data
        training_df_with_labels: pd.DataFrame, # Used for background outcomes and DiCE context
        data_source_for_explanation: pd.DataFrame # Data to explain (e.g., anomalies or all data)
    ):
        """
        Executes the configured XAI methods on the provided data.

        Args:
            training_features_df (pd.DataFrame): DataFrame with features used for training (for background data).
            training_df_with_labels (pd.DataFrame): Training DataFrame including the label column (for background outcomes).
            data_source_for_explanation (pd.DataFrame): DataFrame containing the instances to be explained.
        """
        # print(f"\n--- Starting XAI Execution via XAIRunner for job '{self.job_name}' ---")
        # print(f"Processing {len(self.xai_params)} XAI method(s)...")
        
        self.xai_method_timings.clear() # Clear previous timings if any

        # --- Check Prerequisites ---
        if (TimeSeriesExplainer is None or ut.dataframe_to_sequences is None or
                x.process_and_plot_shap is None or x.process_and_plot_lime is None or x.process_and_plot_dice is None):
            # print("XAI components not available (import failed). Skipping XAI.")
            return
        
        # # print(f"DEBUG run_explanations: START. Job '{self.job_name}'")
        # # print(f"DEBUG run_explanations: data_source_for_explanation.shape: {data_source_for_explanation.shape}, .head(2): \n{data_source_for_explanation.head(2)}")
        # # print(f"DEBUG run_explanations: Timestamp col in source_df: '{self.timestamp_col_name}', Dtype: {data_source_for_explanation[self.timestamp_col_name].dtype if self.timestamp_col_name in data_source_for_explanation else 'Not found'}")
        # # print(f"DEBUG run_explanations: Label col in source_df: '{self.actual_label_col}', Values sample: {data_source_for_explanation[self.actual_label_col].unique()[:5] if self.actual_label_col in data_source_for_explanation else 'Not found'}")

        try:
            # --- 1. Prepare Common Background Data ---
            # print(f"Preparing background data using sequence length {self.sequence_length}...")
            background_data_np = ut.dataframe_to_sequences(
                df=training_features_df,
                sequence_length=self.sequence_length,
                feature_cols=self.feature_columns
            )

            shap_method_for_init = 'kernel' # Default for ts_explainer init
            for config_check in self.xai_params:
                if config_check.get("method") == 'ShapExplainer':
                    settings_check = config_check.get('settings', {})
                    shap_method_for_init = settings_check.get('shap_method', 'kernel')
                    break 

            max_bg_samples = MAX_BG_SAMPLES
            if shap_method_for_init == 'tree' and background_data_np.shape[0] > 0:
                max_bg_samples = len(background_data_np)
                # print(f"Using all {max_bg_samples} background samples for TreeSHAP initialization.")
            elif len(background_data_np) > max_bg_samples:
                # print(f"Sampling background data down from {len(background_data_np)} to {max_bg_samples} instances.")
                indices_bg_sample = np.random.choice(len(background_data_np), max_bg_samples, replace=False)
                background_data_np = background_data_np[indices_bg_sample]

            if background_data_np.size == 0:
                # print("Warning: Background data generation resulted in empty array. Skipping XAI.")
                return # Stop if background fails

            num_bg_sequences = background_data_np.shape[0]
            # Correctly get end indices for background labels
            # If background_data_np was sampled, indices_bg_sample contains the original starting indices of these sequences
            if 'indices_bg_sample' in locals():
                end_indices_bg_labels = [start_idx + self.sequence_length - 1 for start_idx in indices_bg_sample]
            else: # Background data was not sampled, use all
                end_indices_bg_labels = [i + self.sequence_length - 1 for i in range(num_bg_sequences)]
            
            valid_end_indices_bg_labels = [idx for idx in end_indices_bg_labels if idx < len(training_df_with_labels)]
            if len(valid_end_indices_bg_labels) != num_bg_sequences:
                mismatch_warning = (
                    f"Potential mismatch for background labels: {num_bg_sequences} sequences, "
                    f"but {len(valid_end_indices_bg_labels)} valid indices. Trimming sequences."
                )
                warnings.warn(mismatch_warning, RuntimeWarning)
            try:
                background_outcomes_np = training_df_with_labels[self.actual_label_col].iloc[valid_end_indices_bg_labels ].values
                # print(f"Extracted {len(background_outcomes_np)} background labels corresponding to sequences.")
                # If lengths still mismatch after extraction, trim the longer one
                if len(background_outcomes_np) != background_data_np.shape[0]:
                    min_len = min(len(background_outcomes_np), background_data_np.shape[0])
                    # print(f"Adjusting background data/outcomes to matched length: {min_len}")
                    background_data_np = background_data_np[:min_len]
                    background_outcomes_np = background_outcomes_np[:min_len]

            except KeyError:
                raise KeyError(f"Label column '{self.actual_label_col}' not found in training_df_with_labels.")
            except IndexError as e:
                # print(f"Detailed IndexError during label extraction: {e}")
                raise IndexError("Error accessing labels using calculated end indices. Check sequence alignment and sampling logic.")
            except Exception as e:
                # print(f"Unexpected error during background label extraction: {e}")
                raise

            # --- 2. Initialize TimeSeriesExplainer ---
            try:
                # print(f"Initializing TimeSeriesExplainer with mode '{self.mode}', SHAP method '{shap_method_for_init}'...")
                ts_explainer = TimeSeriesExplainer(
                    model=self.model_wrapper,
                    background_data=background_data_np,
                    background_outcomes=background_outcomes_np,
                    feature_names=self.feature_columns,
                    mode=self.mode,
                    # --- Pass DiCE specific context as kwargs ---
                    training_df_for_dice=training_df_with_labels,
                    outcome_name_for_dice=self.actual_label_col,
                    continuous_features_for_dice=self.continuous_features_list,
                    # --- Shap Explainer Method ---
                    shap_method=shap_method_for_init
                )
                # print("TimeSeriesExplainer initialized successfully.")
            except Exception as e:
                # print(f"Failed to initialize TimeSeriesExplainer: {e}")
                traceback.print_exc()
                return # Stop if explainer fails

            # --- 3. Prepare ALL Instances and Labels from Explanation Source ---
            all_instances_to_explain_np = np.array([])
            all_original_labels_for_exp_source = None
            num_instances_available = 0
            try:
                # print(f"Generating all possible sequences from explanation data source...")
                # Features only for sequence generation
                exp_source_features_df = data_source_for_explanation[self.feature_columns]
                all_instances_to_explain_np = ut.dataframe_to_sequences(
                    df=exp_source_features_df, 
                    sequence_length=self.sequence_length, 
                    feature_cols=self.feature_columns
                )
                num_instances_available = all_instances_to_explain_np.shape[0]
                # if num_instances_available == 0: # print("Warning: No sequences generated for explanation."); return
                # print(f"Generated {num_instances_available} total sequences for potential explanation.")

                # Extract Labels corresponding to the END of each sequence from data_source_for_explanation
                end_indices_exp_source = list(range(self.sequence_length - 1, len(data_source_for_explanation)))
                # Ensure we don't go out of bounds for all_instances_to_explain_np
                valid_end_indices_for_labels = end_indices_exp_source[:num_instances_available]


                if self.actual_label_col in data_source_for_explanation.columns:
                    all_original_labels_for_exp_source = data_source_for_explanation[self.actual_label_col].iloc[valid_end_indices_for_labels].values
                    # print(f"Extracted {len(all_original_labels_for_exp_source)} labels for the {num_instances_available} generated sequences.")
                    if len(all_original_labels_for_exp_source) != num_instances_available: # Final length check
                        min_len = min(len(all_original_labels_for_exp_source), num_instances_available)
                        warnings.warn(f"Final label/instance mismatch for explanation source. Adjusting to {min_len}.", RuntimeWarning)
                        all_instances_to_explain_np = all_instances_to_explain_np[:min_len]
                        all_original_labels_for_exp_source = all_original_labels_for_exp_source[:min_len]; num_instances_available = min_len
                else: warnings.warn(f"Label column '{self.actual_label_col}' not found in explanation source.")

            except Exception as prep_err: 
                print(f"ERROR preparing instances/labels from source: {prep_err}"); traceback.print_exc(); return
            
            # # print(f"DEBUG run_explanations: num_instances_available (sequences for explanation): {num_instances_available}")
            # if all_original_labels_for_exp_source is not None:
                # # print(f"DEBUG run_explanations: all_original_labels_for_exp_source shape: {all_original_labels_for_exp_source.shape}, sample: {all_original_labels_for_exp_source[:5]}")
            
            # --- 4. Define Plot Handlers Dictionary ---
            plot_handlers = {
                "ShapExplainer": x.process_and_plot_shap,
                "LimeExplainer": x.process_and_plot_lime,
                "DiceExplainer": x.process_and_plot_dice
            }

            # --- 5. Loop Through XAI Methods from Config ---
            for xai_config in self.xai_params:
                method_name = xai_config.get("method")
                settings = xai_config.get("settings", {})
                method_start_time = time.perf_counter() # Start timer for this method

                if not method_name or method_name == "none": 
                    if method_name: self.xai_method_timings[method_name] = 0.0 # Record 0 time if skipped
                    continue
                # print(f"\n===== Running Method: {method_name.upper()} =====")

                try:
                    # --- Determine Indices TO Explain for THIS method ---
                    final_sequence_indices = np.array([], dtype=int)
                    method_specific_orig_indices = settings.get('explain_indices') # Check for override list

                    if method_specific_orig_indices is not None and isinstance(method_specific_orig_indices, list):
                        # print(f"Using method-specific indices from config: {method_specific_orig_indices[:10]}...")
                        # Ensure indices are integers and valid for the source DF
                        selected_original_indices = np.array([int(i) for i in method_specific_orig_indices if i in data_source_for_explanation.index])
                        if len(selected_original_indices) != len(method_specific_orig_indices):
                            warnings.warn("Some method-specific indices were out of bounds for the source DataFrame.", RuntimeWarning)
                    else:
                        # Use global strategy defined in __init__
                        current_strategy = settings.get('sampling_strategy', self.xai_sampling_strategy) # Allow override of strategy per method? No, use global.
                        current_n_samples = settings.get('n_explain_max', 10)
                        # print(f"Using global sampling strategy '{self.xai_sampling_strategy}' with n={current_n_samples}.")
                        selected_original_indices = ut.select_explanation_indices(
                            data_source_for_explanation, # Sample from the full source
                            current_strategy,
                            current_n_samples,
                            label_col=self.actual_label_col,
                            random_state=self.xai_sample_seed
                        )

                    if len(selected_original_indices) == 0:
                        # print(f"WARNING: No original indices selected based on strategy/override for {method_name}. Skipping method.")
                        self.xai_method_timings[method_name] = time.perf_counter() - method_start_time # Record time even if skipped early
                        continue

                    # BEGIN: Index selection logic from original, adapted slightly
                    final_sequence_indices_in_full_set = np.array([], dtype=int) # Indices relative to all_instances_to_explain_np
                    method_specific_orig_df_indices = settings.get('explain_indices') 

                    selected_original_df_indices = [] # Indices relative to data_source_for_explanation

                    if method_specific_orig_df_indices is not None and isinstance(method_specific_orig_df_indices, list):
                        # print(f"Using method-specific indices from config: {method_specific_orig_df_indices[:10]}...")
                        selected_original_df_indices = np.array([int(i) for i in method_specific_orig_df_indices if i in data_source_for_explanation.index])
                        if len(selected_original_df_indices) != len(method_specific_orig_df_indices):
                            warnings.warn("Some method-specific indices were out of bounds for the source DataFrame.", RuntimeWarning)
                    else:
                        current_n_samples_for_exp = settings.get('n_explain_max', 10) # n_explain_max in config or default
                        # print(f"Using global sampling strategy '{self.xai_sampling_strategy}' with n={current_n_samples_for_exp} for method {method_name}.")
                        selected_original_df_indices = ut.select_explanation_indices(
                            data_source_for_explanation, self.xai_sampling_strategy,
                            current_n_samples_for_exp, label_col=self.actual_label_col,
                            random_state=self.xai_sample_seed
                        )
                    
                    if len(selected_original_df_indices) == 0:
                        # print(f"WARNING: No original DataFrame indices selected for {method_name}. Skipping method.")
                        self.xai_method_timings[method_name] = time.perf_counter() - method_start_time
                        continue

                    # Map selected original DF indices to sequence array indices
                    offset = self.sequence_length - 1
                    # sequence_indices_potential are indices into all_instances_to_explain_np
                    sequence_indices_potential = selected_original_df_indices - offset 
                    valid_mask = (sequence_indices_potential >= 0) & (sequence_indices_potential < num_instances_available)
                    final_sequence_indices_in_full_set = sequence_indices_potential[valid_mask].astype(int)

                    if len(final_sequence_indices_in_full_set) < len(selected_original_df_indices):
                        warnings.warn(f"Could not map all selected original DF indices to valid sequence indices for {method_name}.", RuntimeWarning)
                    
                    if len(final_sequence_indices_in_full_set) == 0:
                        # print(f"WARNING: No valid sequence indices derived for {method_name}. Skipping method.")
                        self.xai_method_timings[method_name] = time.perf_counter() - method_start_time
                        continue
                    
                    # Apply per-method instance limit (n_explain_max from settings) AFTER initial selection
                    n_explain_max_setting = settings.get("n_explain_max") # This was already used for sampling if no override
                    if n_explain_max_setting is not None: # Re-apply if it's a hard limit different from sampling N
                        try:
                            n_explain_max_val = int(n_explain_max_setting)
                            if n_explain_max_val > 0 and n_explain_max_val < len(final_sequence_indices_in_full_set):
                                # print(f"Applying n_explain_max={n_explain_max_val} (selected {len(final_sequence_indices_in_full_set)} initially). Taking first N.")
                                final_sequence_indices_in_full_set = final_sequence_indices_in_full_set[:n_explain_max_val]
                        except (ValueError, TypeError):
                            warnings.warn(f"Invalid n_explain_max value '{n_explain_max_setting}'. Ignoring limit.", RuntimeWarning)
                    
                    num_sequences_to_process_for_method = len(final_sequence_indices_in_full_set)
                    if num_sequences_to_process_for_method == 0:
                        print(f"No instances left to explain for {method_name} after limiting/mapping. Skipping."); 
                        self.xai_method_timings[method_name] = time.perf_counter() - method_start_time
                        continue
                    
                    # print(f"Final number of sequences to explain for {method_name}: {num_sequences_to_process_for_method}")

                    current_instances_np = all_instances_to_explain_np[final_sequence_indices_in_full_set]
                    current_original_labels_for_method = None
                    if all_original_labels_for_exp_source is not None:
                        try:
                            current_original_labels_for_method = all_original_labels_for_exp_source[final_sequence_indices_in_full_set]
                        except IndexError as slice_err:
                            print(f"ERROR slicing labels with final sequence indices for {method_name}: {slice_err}. Proceeding without labels for this method.")
                    # END: Index selection logic from original

                    explainer_object = ts_explainer._get_or_initialize_explainer(method_name)
                    if explainer_object is None: 
                        print(f"Could not get explainer for {method_name}. Skipping.")
                        self.xai_method_timings[method_name] = time.perf_counter() - method_start_time
                        continue

                    # print(f"Using configuration for {method_name}: {settings}")
                    handler_args = {
                        "explainer_object": explainer_object, "feature_names": self.feature_columns,
                        "sequence_length": self.sequence_length, "output_dir": self.output_dir,
                        "mode": self.mode, "job_name": self.job_name,
                        "original_labels": current_original_labels_for_method # Pass the specific labels for these instances
                    }
                    
                    xai_results_batch = None # To store results from ts_explainer.explain

                    # --- Method Specific Logic ---
                    if method_name == "DiceExplainer":
                        features_to_vary = settings.get('features_to_vary', self.feature_columns)
                        dice_runtime_kwargs = {'total_CFs': settings.get('total_CFs', 4), 
                                            'desired_class': settings.get('desired_class', 'opposite'), 
                                            'features_to_vary': features_to_vary}
                        # print(f"DiCE Runtime Params: {dice_runtime_kwargs}")

                        # For DiCE, we will process one instance at a time for NDCG
                        # # print(f"DEBUG DiCE: Processing {num_sequences_to_process_for_method} instances for DiCE explanations and NDCG.")
                        for loop_idx in range(num_sequences_to_process_for_method):
                            instance_orig_df_idx = final_sequence_indices_in_full_set[loop_idx] + self.sequence_length - 1
                            # # print(f"DEBUG DiCE Loop: loop_idx={loop_idx}, instance_orig_df_idx={instance_orig_df_idx}")
                            if instance_orig_df_idx >= len(data_source_for_explanation):
                                # # print(f"DEBUG DiCE Loop: SKIPPING instance_orig_df_idx {instance_orig_df_idx} (OOB)")
                                continue

                            current_instance_single_np_for_dice = current_instances_np[loop_idx : loop_idx + 1] # Shape (1, seq, feat)
                            
                            current_label_single_for_dice = None
                            if current_original_labels_for_method is not None:
                                current_label_single_for_dice = [current_original_labels_for_method[loop_idx]]

                            try:
                                dice_explanation_object_for_instance = ts_explainer.explain(
                                    instances_to_explain=current_instance_single_np_for_dice, 
                                    method_name=method_name, 
                                    **dice_runtime_kwargs
                                )
                                # If explain returns a list even for one instance:
                                if isinstance(dice_explanation_object_for_instance, list) and dice_explanation_object_for_instance:
                                    dice_explanation_object_for_instance = dice_explanation_object_for_instance[0]
                                
                                # Plotting for this DiCE instance
                                handler_args_dice_instance = handler_args.copy()
                                handler_args_dice_instance.update({
                                    "results": dice_explanation_object_for_instance, 
                                    "instances_explained": current_instance_single_np_for_dice,
                                    "original_labels": current_label_single_for_dice, 
                                    "instance_index": loop_idx # instance_index might be useful for saving plots if DiceExplainer doesn't handle multiple instances in one plot
                                })
                                plot_func = plot_handlers.get(method_name)
                                if plot_func:
                                    # print(f"Calling plot handler for DiCE instance {loop_idx}...")
                                    plot_func(**handler_args_dice_instance)

                                # --- NDCG Calculation for DiCE instance ---
                                if self.inj_params:
                                    # # print(f"DEBUG DiCE NDCG: Attempting to get GT for instance_orig_df_idx={instance_orig_df_idx}")
                                    true_relevant_feats = self._get_ground_truth_features_for_instance(instance_orig_df_idx, data_source_for_explanation)
                                    if true_relevant_feats:
                                        # # print(f"DEBUG DiCE NDCG: GT found for {instance_orig_df_idx}: {true_relevant_feats}")
                                        # Pass the original instance data to _extract_aggregated_feature_scores
                                        xai_scores = self._extract_aggregated_feature_scores(
                                            dice_explanation_object_for_instance, 
                                            method_name,
                                            original_instance_sequence_np=current_instance_single_np_for_dice
                                        )
                                        if xai_scores:
                                            self.all_aggregated_scores_for_ndcg.setdefault(method_name, []).append(xai_scores)
                                            for k_val in self.ndcg_k_values:
                                                ndcg_score = ut.calculate_ndcg_at_k(xai_scores, true_relevant_feats, self.feature_columns, k_val)
                                                self.ndcg_results.setdefault(method_name, {}).setdefault(k_val, []).append(ndcg_score)
                                                # print(f"  DiCE NDCG@{k_val} for instance {loop_idx} (OrigDFIdx {instance_orig_df_idx}): {ndcg_score:.4f}")
                                        #else:
                                            # print(f"DEBUG DiCE NDCG: No XAI scores extracted for instance {loop_idx}")
                                    #else:
                                        # print(f"DEBUG DiCE NDCG: No GT for {instance_orig_df_idx}")
                            except Exception as dice_instance_err:
                                # print(f"ERROR during DiCE explanation/plotting/NDCG for instance {loop_idx} (OrigDFIdx: {instance_orig_df_idx}): {dice_instance_err}")
                                traceback.print_exc()
                        # print(f"--- Finished DiCE Explanations ---")
                    elif method_name == "ShapExplainer":
                        current_shap_method_run = settings.get('shap_method', shap_method_for_init) # Use method from settings or the one used for init
                        
                        # Re-initialize explainer if shap_method for this run is different than during ts_explainer init
                        if current_shap_method_run != ts_explainer.shap_method: # Assuming ts_explainer stores its shap_method
                            # print(f"Re-initializing SHAP explainer for method: {current_shap_method_run} (was {ts_explainer.shap_method})")
                            ts_explainer._initialize_shap_explainer(current_shap_method_run, background_data_np, background_outcomes_np) # Pass BGs again
                            explainer_object = ts_explainer.shap_explainer # Update explainer_object
                            handler_args["explainer_object"] = explainer_object


                        n_samples_shap = settings.get('nsamples', 50); k_features = settings.get('l1_reg_k_features', 10 if self.sequence_length * len(self.feature_columns) > 10 else self.sequence_length * len(self.feature_columns) )
                        l1_reg_shap = 'auto'
                        if current_shap_method_run != 'tree':
                            n_flat_features = self.sequence_length * len(self.feature_columns)
                            l1_reg_shap = f'num_features({k_features})' if n_samples_shap < n_flat_features and k_features > 0 else 'auto'
                        
                        shap_runtime_kwargs = {'nsamples': n_samples_shap, 'l1_reg': l1_reg_shap}
                        # print(f"SHAP Runtime Params ({current_shap_method_run}): {shap_runtime_kwargs}")
                        xai_results_batch = ts_explainer.explain(instances_to_explain=current_instances_np, method_name=method_name, **shap_runtime_kwargs)
                        handler_args.update({"results": xai_results_batch, "instances_explained": current_instances_np})
                        # Plotting
                        plot_func = plot_handlers.get(method_name)
                        if plot_func: print(f"Calling plot handler for SHAP..."); plot_func(**handler_args)

                        if xai_results_batch is not None and self.inj_params:
                            shap_values_to_process = xai_results_batch
                            if isinstance(xai_results_batch, list) and len(xai_results_batch) > 0: # Check for list and non-empty
                                class_idx_for_ndcg = 1 if len(xai_results_batch) > 1 else 0 # Choose class 1 if available, else 0
                                if class_idx_for_ndcg < len(xai_results_batch):
                                    shap_values_to_process = xai_results_batch[class_idx_for_ndcg]
                                else:
                                    warnings.warn(f"SHAP results list length {len(xai_results_batch)} invalid for class index {class_idx_for_ndcg}. Using class 0.", RuntimeWarning)
                                    shap_values_to_process = xai_results_batch[0]
                            
                            if not isinstance(shap_values_to_process, np.ndarray) or \
                            shap_values_to_process.shape[0] != num_sequences_to_process_for_method:
                                warnings.warn(f"SHAP results for NDCG have unexpected format/shape. Expected {num_sequences_to_process_for_method} instances. Got shape: {getattr(shap_values_to_process,'shape','N/A')}. Skipping NDCG for SHAP.", RuntimeWarning)
                            else:
                                # print(f"DEBUG run_explanations SHAP: Processing {num_sequences_to_process_for_method} instances for NDCG.")
                                for loop_idx in range(num_sequences_to_process_for_method):
                                    instance_orig_df_idx = final_sequence_indices_in_full_set[loop_idx] + self.sequence_length - 1
                                    # print(f"DEBUG run_explanations SHAP Loop: loop_idx={loop_idx}, instance_orig_df_idx={instance_orig_df_idx}")
                                    if instance_orig_df_idx >= len(data_source_for_explanation):
                                        # print(f"DEBUG run_explanations SHAP Loop: SKIPPING instance_orig_df_idx {instance_orig_df_idx} (out of bounds for source_df len {len(data_source_for_explanation)})")
                                        continue
                                    
                                    # print(f"DEBUG run_explanations SHAP: Attempting to get GT for instance_orig_df_idx={instance_orig_df_idx}")
                                    true_relevant_feats = self._get_ground_truth_features_for_instance(instance_orig_df_idx, data_source_for_explanation)
                                    if true_relevant_feats:
                                        # print(f"DEBUG run_explanations SHAP: GT found for {instance_orig_df_idx}: {true_relevant_feats}")
                                        shap_slice_for_instance = shap_values_to_process[loop_idx] # This is (seq_len, n_features)
                                        xai_scores = self._extract_aggregated_feature_scores(shap_slice_for_instance, method_name)
                                        if xai_scores:
                                            self.all_aggregated_scores_for_ndcg.setdefault(method_name, []).append(xai_scores)
                                            for k_val in self.ndcg_k_values:
                                                ndcg_score = ut.calculate_ndcg_at_k(xai_scores, true_relevant_feats, self.feature_columns, k_val)
                                                self.ndcg_results.setdefault(method_name, {}).setdefault(k_val, []).append(ndcg_score)
                                                # print(f"  SHAP NDCG@{k_val} for instance {loop_idx} (OrigDFIdx {instance_orig_df_idx}): {ndcg_score:.4f}") # Original print
                                        # else:
                                            # print(f"DEBUG run_explanations SHAP: No XAI scores extracted for instance {loop_idx} (OrigDFIdx {instance_orig_df_idx}).")
                                    # else:
                                        # print(f"DEBUG run_explanations SHAP: No GT for {instance_orig_df_idx}")
                        # --- END NDCG FOR SHAP ---

                    elif method_name == "LimeExplainer":
                        # LIME is per-instance, so loop here for explain and NDCG
                        num_features_lime = settings.get('num_features', 10); num_samples_lime = settings.get('num_samples', 1000)
                        lime_runtime_kwargs = {'num_features': num_features_lime, 'num_samples': num_samples_lime}
                        # print(f"LIME Runtime Params: {lime_runtime_kwargs}")
                        # print(f"Executing LIME for {num_sequences_to_process_for_method} selected instance(s)...")

                        for loop_idx in range(num_sequences_to_process_for_method):
                            # Get original DF index for this instance (point of explanation)
                            # final_sequence_indices_in_full_set[loop_idx] is index in all_instances_to_explain_np
                            # This corresponds to original DF row: final_sequence_indices_in_full_set[loop_idx] + sequence_length - 1
                            instance_orig_df_idx = final_sequence_indices_in_full_set[loop_idx] + self.sequence_length - 1
                            if instance_orig_df_idx >= len(data_source_for_explanation): # Boundary check
                                warnings.warn(f"LIME loop: instance_orig_df_idx {instance_orig_df_idx} out of bounds for data_source_for_explanation. Skipping.")
                                continue

                            # print(f"--- Explaining Instance LoopIdx={loop_idx} (OrigDFIdx={instance_orig_df_idx}) with LIME ---")
                            current_instance_single_np = current_instances_np[loop_idx : loop_idx + 1] # (1, seq, feat)
                            
                            current_label_single = None
                            if current_original_labels_for_method is not None:
                                current_label_single = [current_original_labels_for_method[loop_idx]]

                            try:
                                lime_explanation_obj = ts_explainer.explain(instances_to_explain=current_instance_single_np, method_name='LimeExplainer', **lime_runtime_kwargs)
                                
                                # Plotting for LIME
                                handler_args_lime_instance = handler_args.copy()
                                handler_args_lime_instance.update({
                                    "results": lime_explanation_obj, "instances_explained": current_instance_single_np,
                                    "instance_index": loop_idx, "original_labels": current_label_single
                                })
                                plot_func = plot_handlers.get(method_name)
                                if plot_func: print(f"Calling plot handler for LIME instance {loop_idx}..."); plot_func(**handler_args_lime_instance)
                                
                                # --- NDCG Calculation for LIME instance ---
                                if self.inj_params: # Only if ground truth is available
                                    # # print(f"DEBUG run_explanations LIME: Attempting to get GT for instance_orig_df_idx={instance_orig_df_idx}")
                                    true_relevant_feats = self._get_ground_truth_features_for_instance(instance_orig_df_idx, data_source_for_explanation)
                                    # if true_relevant_feats:
                                        # # print(f"DEBUG run_explanations LIME: GT found for {instance_orig_df_idx}: {true_relevant_feats}")
                                    
                                    # print("Got inj_params. Ground Truth found")
                                    true_relevant_feats = self._get_ground_truth_features_for_instance(instance_orig_df_idx, data_source_for_explanation)
                                    if true_relevant_feats:
                                        xai_scores = self._extract_aggregated_feature_scores(lime_explanation_obj, method_name)
                                        if xai_scores:
                                            self.all_aggregated_scores_for_ndcg.setdefault(method_name, []).append(xai_scores)
                                            for k_val in self.ndcg_k_values:
                                                ndcg_score = ut.calculate_ndcg_at_k(xai_scores, true_relevant_feats, self.feature_columns, k_val)
                                                self.ndcg_results.setdefault(method_name, {}).setdefault(k_val, []).append(ndcg_score)
                                                # print(f"  LIME NDCG@{k_val} for instance {loop_idx} (OrigDFIdx {instance_orig_df_idx}): {ndcg_score:.4f}")
                            except Exception as lime_instance_err:
                                # print(f"ERROR during LIME explanation/plotting/NDCG for instance {loop_idx} (OrigDFIdx: {instance_orig_df_idx}): {lime_instance_err}")
                                traceback.print_exc()
                        # print(f"--- Finished LIME Explanations ---")
                    else:
                        # print(f"Skipping unknown or unhandled XAI method: {method_name}")
                        self.xai_method_timings[method_name] = time.perf_counter() - method_start_time
                        continue # Skip NDCG for unhandled

                    # --- NDCG Calculation for BATCHED methods (SHAP, potentially DiCE if adapted) ---
                    if method_name in ["ShapExplainer"] and xai_results_batch is not None and self.inj_params:
                        # xai_results_batch for SHAP is ndarray (n_instances, seq_len, n_features) or list of these
                        # current_instances_np is (n_instances, seq_len, n_features)
                        
                        # Determine which part of SHAP results to use (e.g., for class 1 if binary classification)
                        shap_values_to_process = xai_results_batch
                        if isinstance(xai_results_batch, list) and len(xai_results_batch) > 1:
                             # Assuming class 1 (anomaly) is of interest if binary classification
                             # This might need refinement based on model output / interpretation
                             class_idx_for_ndcg = 1 
                             if class_idx_for_ndcg < len(xai_results_batch):
                                 shap_values_to_process = xai_results_batch[class_idx_for_ndcg]
                             else: # Fallback to first class if index is out of bounds
                                 warnings.warn(f"SHAP results list length {len(xai_results_batch)} too short for class index {class_idx_for_ndcg}. Using class 0 for NDCG.", RuntimeWarning)
                                 shap_values_to_process = xai_results_batch[0]
                        
                        if not isinstance(shap_values_to_process, np.ndarray) or \
                           shap_values_to_process.shape[0] != num_sequences_to_process_for_method:
                            warnings.warn(f"SHAP results for NDCG have unexpected format/shape. Skipping NDCG. Shape: {getattr(shap_values_to_process,'shape','N/A')}", RuntimeWarning)
                        else:
                            for loop_idx in range(num_sequences_to_process_for_method):
                                instance_orig_df_idx = final_sequence_indices_in_full_set[loop_idx] + self.sequence_length - 1
                                if instance_orig_df_idx >= len(data_source_for_explanation): continue # Boundary check

                                true_relevant_feats = self._get_ground_truth_features_for_instance(instance_orig_df_idx, data_source_for_explanation)
                                if true_relevant_feats:
                                    # shap_values_to_process is (n_instances, seq_len, n_features)
                                    # So, shap_values_to_process[loop_idx] is (seq_len, n_features)
                                    xai_scores = self._extract_aggregated_feature_scores(shap_values_to_process[loop_idx], method_name)
                                    if xai_scores:
                                        self.all_aggregated_scores_for_ndcg.setdefault(method_name, []).append(xai_scores)
                                        for k_val in self.ndcg_k_values:
                                            ndcg_score = ut.calculate_ndcg_at_k(xai_scores, true_relevant_feats, self.feature_columns, k_val)
                                            self.ndcg_results.setdefault(method_name, {}).setdefault(k_val, []).append(ndcg_score)
                                            # print(f"  {method_name} NDCG@{k_val} for instance {loop_idx} (OrigDFIdx {instance_orig_df_idx}): {ndcg_score:.4f}")
                
                except Exception as explain_err:
                    # print(f"ERROR during explanation/plotting/NDCG setup for method '{method_name}': {explain_err}")
                    traceback.print_exc()
                finally:
                    self.xai_method_timings[method_name] = round(time.perf_counter() - method_start_time, 4)
                    
                # --- After the loop, aggregate and save the summary feature importances ---
                if self.all_aggregated_scores_for_ndcg:
                    summary_feature_importances = {}
                    for method, scores_list in self.all_aggregated_scores_for_ndcg.items():
                        if not scores_list:
                            continue
                        # Convert list of dicts to DataFrame for easy averaging
                        df_scores = pd.DataFrame(scores_list)
                        # Average scores for each feature across all explained instances for this method
                        # It's possible not all features appear in all xai_scores dicts (e.g., if LIME/DiCE don't report 0 for some)
                        # So, df_scores.mean() will handle NaNs correctly by skipping them for mean calculation.
                        # We want to ensure all self.feature_columns are present, with 0 if they never got a score.
                        avg_scores = df_scores.mean().fillna(0) # Fill NaN with 0 if a feature was never scored by any instance
                        
                        # Ensure all base features are in the series, even if they had no scores from any instance
                        for base_feat in self.feature_columns:
                            if base_feat not in avg_scores:
                                avg_scores[base_feat] = 0.0
                                
                        summary_feature_importances[method] = avg_scores.to_dict()

                    if summary_feature_importances:
                        # Save this summary to a file in the job's main XAI output directory
                        # (not under a specific method's subdir)
                        summary_fi_path = os.path.join(self.output_dir, self.job_name, "aggregated_feature_importances.json")
                        try:
                            os.makedirs(os.path.dirname(summary_fi_path), exist_ok=True)
                            with open(summary_fi_path, 'w') as f:
                                json.dump(summary_feature_importances, f, indent=4)
                            # print(f"DEBUG XAIRunner: Saved aggregated feature importances to {summary_fi_path}")
                        except Exception as e:
                            print(f"ERROR XAIRunner: Failed to save aggregated feature importances: {e}")

        except Exception as xai_general_err:
            # print(f"ERROR during XAI processing in XAIRunner: {xai_general_err}")
            traceback.print_exc()

        # print(f"--- Finished XAI Execution via XAIRunner for job '{self.job_name}' ---")
        # print(f"DEBUG run_explanations: NDCG Summary: Found ground truth for {self.ndcg_ground_truth_found_count} instances out of {self.ndcg_anomalies_explained_count} explained anomalous instances.") # Added
        # if self.ndcg_anomalies_explained_count == 0 and self.inj_params: # Added
            # print("DEBUG run_explanations: NDCG Warning: No anomalous instances were selected for explanation OR label mismatch. NDCG cannot be calculated meaningfully.") # Added
        # elif self.ndcg_ground_truth_found_count == 0 and self.ndcg_anomalies_explained_count > 0: # Added
            # print("DEBUG run_explanations: NDCG Warning: Ground truth features could not be matched for any explained anomalous instances. Check timestamp/duration/label alignment in inj_params and data.") # Added
        # print(f"DEBUG run_explanations: FINAL ndcg_results dictionary: {self.ndcg_results}") # Added
        
        
    def get_xai_method_timings(self) -> Dict[str, float]:
        """Returns the execution times for each XAI method."""
        return self.xai_method_timings