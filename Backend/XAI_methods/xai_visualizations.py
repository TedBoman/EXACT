# File: xai_visualizations.py

import traceback
import shap
import dice_ml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from typing import Any, List, Optional, Union

# --- SHAP Handler ---
def process_and_plot_shap(
    results: Any,                      # Expect np.ndarray or list[np.ndarray] from ShapExplainer.explain
    explainer_object: Any,             # The specific ShapExplainer instance (to get expected_value)
    instances_explained: np.ndarray,   # 3D numpy array (n, seq, feat) that was explained
    original_labels: Union[np.ndarray, pd.Series], # Unused in this snippet, but kept for signature consistency
    feature_names: List[str],          # Base feature names (e.g., ['Temp', 'Pressure'])
    sequence_length: int,
    output_dir: str,
    mode: str,                         # 'classification' or 'regression'
    class_index_to_plot: int = 0,      # Default class to plot for classification
    max_display_features: int = 20,
    job_name='none',
):
    """
    Processes SHAP results and generates standard static plots (.png)
    and interactive HTML plots (.html) suitable for frontend display.
    Files are saved to output_dir/job_name/SHAP/
    """    
    # print(f"--- Processing and Plotting SHAP Results (Class Index: {class_index_to_plot if mode=='classification' else 'N/A'}) ---")

    if results is None:
        # print("SHAP results are None. Skipping plotting.")
        return

    # Construct the full output directory path
    specific_output_dir = os.path.join(output_dir, job_name, 'SHAP')
    os.makedirs(specific_output_dir, exist_ok=True) # Ensure directory exists

    # --- Prepare Data for Standard SHAP Plots ---
    n_instances_explained = instances_explained.shape[0]
    n_features = len(feature_names)
    n_flat_features = sequence_length * n_features

    feature_names_flat = [f"{feat}_t-{i}" for feat in feature_names for i in range(sequence_length - 1, -1, -1)]
    features_flat_np = instances_explained.reshape(n_instances_explained, -1)

    is_classification = isinstance(results, list)
    shap_values_3d = None
    expected_value_for_plot = None
    base_expected_value = getattr(explainer_object, 'expected_value', None)

    if is_classification:
        if not results or not isinstance(results, list) or class_index_to_plot >= len(results):
            # print(f"SHAP results list is empty, not a list, or invalid class index {class_index_to_plot}. Skipping plotting.")
            return
        shap_values_3d = results[class_index_to_plot]
        if base_expected_value is not None:
            if hasattr(base_expected_value, '__len__') and not isinstance(base_expected_value, str) and len(base_expected_value) > class_index_to_plot:
                expected_value_for_plot = base_expected_value[class_index_to_plot]
            elif not hasattr(base_expected_value, '__len__') or isinstance(base_expected_value, (int, float)): # scalar for binary
                 expected_value_for_plot = base_expected_value
            else:
                warnings.warn(f"Could not get expected value for class {class_index_to_plot}. Expected value type: {type(base_expected_value)}", RuntimeWarning)
        else:
            warnings.warn(f"Base expected value not found in explainer object.", RuntimeWarning)
    else: # Regression or Binary assumed to return single array
        shap_values_3d = results
        expected_value_for_plot = base_expected_value

    if shap_values_3d is None or shap_values_3d.size == 0:
        # print("No valid SHAP values found for the selected class/output. Skipping plotting.")
        return

    # --- Final Validation and Reshaping for Plotting ---
    expected_shape_3d = (n_instances_explained, sequence_length, n_features)
    if shap_values_3d.shape != expected_shape_3d:
        # print(f"Warning: SHAP values shape {shap_values_3d.shape} mismatch expected {expected_shape_3d}. Check explainer's output.")
        try:
            shap_values_flat = shap_values_3d.reshape(n_instances_explained, -1)
            if shap_values_flat.shape[1] != n_flat_features:
                raise ValueError(f"Flattened SHAP values shape {shap_values_flat.shape[1]} mismatch expected flat features {n_flat_features}.")
        except ValueError as e:
            # print(f"Cannot proceed with plotting due to shape mismatch: {e}")
            return
    else:
        shap_values_flat = shap_values_3d.reshape(n_instances_explained, -1)

    # --- Generate and Save Plots ---
    # print(f"Saving SHAP plots to {specific_output_dir}...")
    plot_suffix = f"_c{class_index_to_plot}" if is_classification else ""

    shap_explanation = None
    if expected_value_for_plot is not None:
        try:
            # For shap.Explanation, base_values should ideally be an array if values is 2D,
            # or a scalar if it's the same for all instances.
            # If expected_value_for_plot is scalar, it will be broadcasted.
            base_values_for_explanation = np.full(n_instances_explained, expected_value_for_plot) \
                                          if isinstance(expected_value_for_plot, (float, int)) \
                                          else expected_value_for_plot

            shap_explanation = shap.Explanation(
                values=shap_values_flat,
                base_values=base_values_for_explanation,
                data=features_flat_np,
                feature_names=feature_names_flat
            )
        except Exception as e:
            # print(f"Could not create shap.Explanation object: {e}. Some plots may require it or might not be accurate.")
            # Fallback for summary_plot if explanation fails but we have shap_values_flat
            if shap_values_flat is None:
                 # print("shap_values_flat is also None. Cannot proceed with summary plot either.")
                 return # Cannot make any plots if this fails and shap_values_flat is also bad
    elif shap_values_flat is None: # If expected_value is None and shap_values_flat is also None
        # print("SHAP values (shap_values_flat) are None and expected_value_for_plot is None. Cannot create SHAP Explanation or plots.")
        return
    #else: # expected_value_for_plot is None, but we might still make some plots like summary
        #print("expected_value_for_plot is None. Waterfall and Force plots requiring base values will be skipped or may error.")


    # --- Summary Plot (Dot) ---
    if shap_values_flat is not None and features_flat_np is not None:
        try:
            plt.figure()
            shap.summary_plot(shap_values_flat, features=features_flat_np, feature_names=feature_names_flat, show=False, plot_type='dot')
            plt.title(f"SHAP Summary Plot (Dot{plot_suffix})")
            plt.savefig(os.path.join(specific_output_dir, f"{job_name}_shap_summary_dot{plot_suffix}.png"), bbox_inches='tight')
            # print("Saved Summary Plot (Dot).")
        except Exception as e: print(f"Failed Summary Plot (Dot): {e}")
        finally: plt.close()
    # else:
        # print("Skipping Summary Plot (Dot): Missing shap_values_flat or features_flat_np.")

    # --- Bar Plot ---
    if shap_explanation:
        try:
            plt.figure()
            shap.plots.bar(shap_explanation, max_display=max_display_features, show=False)
            plt.title(f"SHAP Feature Importance (Bar{plot_suffix})")
            plt.savefig(os.path.join(specific_output_dir, f"{job_name}_shap_summary_bar{plot_suffix}.png"), bbox_inches='tight')
            # print("Saved Summary Plot (Bar).")
        except Exception as e: print(f"Failed Summary Plot (Bar): {e}")
        finally: plt.close()
    # else: print("Skipping Bar plot: shap.Explanation object not available.")

    # --- Waterfall Plot (First Instance) ---
    if shap_explanation and expected_value_for_plot is not None and n_instances_explained > 0:
        instance_idx_to_plot = 0
        try:
            plt.figure()
            # shap.plots.waterfall needs a single instance from the Explanation object
            shap.plots.waterfall(shap_explanation[instance_idx_to_plot], max_display=max_display_features, show=False)
            plt.savefig(os.path.join(specific_output_dir, f"{job_name}_shap_waterfall_inst{instance_idx_to_plot}{plot_suffix}.png"), bbox_inches='tight')
            # print(f"Saved Waterfall Plot for Instance {instance_idx_to_plot}.")
        except Exception as e: print(f"Failed Waterfall Plot: {e}")
        finally: plt.close()
    # elif expected_value_for_plot is None: # print("Skipping Waterfall plot: expected_value not available.")
    # elif not shap_explanation : # print("Skipping Waterfall plot: shap.Explanation object not available.")
    # elif n_instances_explained == 0: # print("Skipping Waterfall plot: no instances to plot.")


    # --- Heatmap Plot ---
    if shap_explanation:
        try:
            # Heatmap often needs more vertical space
            fig_height = max(6, min(n_instances_explained, 50) * 0.4) # Limit max instances for height calc to avoid huge figs
            plt.figure(figsize=(10, fig_height)) # Set figure size before plotting
            shap.plots.heatmap(shap_explanation, max_display=min(max_display_features, n_flat_features), show=False)
            plt.savefig(os.path.join(specific_output_dir, f"{job_name}_shap_heatmap{plot_suffix}.png"), bbox_inches='tight')
            # print("Saved Heatmap Plot.")
        except Exception as e: print(f"Failed Heatmap Plot: {e}")
        finally: plt.close() # Close the figure created for heatmap
    # else: print("Skipping Heatmap plot: shap.Explanation object not available.")

    # --- Interactive Force Plot (First Instance) ---
    if shap_explanation and expected_value_for_plot is not None and n_instances_explained > 0:
        instance_idx_to_plot = 0
        try:
            # shap.plots.force for a single instance from the Explanation object
            # This returns an AdditiveForceVisualizer object
            force_plot_instance_obj = shap.plots.force(shap_explanation[instance_idx_to_plot], show=False)
            if force_plot_instance_obj:
                save_path = os.path.join(specific_output_dir, f"{job_name}_shap_force_interactive_inst{instance_idx_to_plot}{plot_suffix}.html")
                shap.save_html(save_path, force_plot_instance_obj)
                # print(f"Saved Interactive Force Plot for Instance {instance_idx_to_plot} to {save_path}")
            # else:
                # print(f"Failed to generate Interactive Force Plot object for Instance {instance_idx_to_plot}.")
        except Exception as e:
            print(f"Failed Interactive Force Plot (Instance {instance_idx_to_plot}): {e}")
    # elif expected_value_for_plot is None: # print("Skipping Interactive Force Plot (Instance): expected_value not available.")
    # elif not shap_explanation : # print("Skipping Interactive Force Plot (Instance): shap.Explanation object not available.")
    # elif n_instances_explained == 0: # print("Skipping Interactive Force Plot (Instance): no instances to plot.")


    # --- Interactive Force Plot (All Instances - Global Summary) ---
    if shap_explanation and expected_value_for_plot is not None:
        try:
            # shap.plots.force for all instances in the Explanation object
            # This also returns an AdditiveForceVisualizer object, for a global summary
            force_plot_all_obj = shap.plots.force(shap_explanation, show=False)
            if force_plot_all_obj:
                save_path = os.path.join(specific_output_dir, f"{job_name}_shap_force_interactive_all_instances{plot_suffix}.html")
                shap.save_html(save_path, force_plot_all_obj)
                # print(f"Saved Interactive Force Plot (All Instances) to {save_path}")
            # else:
                # print(f"Failed to generate Interactive Force Plot object for All Instances.")
        except Exception as e:
            print(f"Failed Interactive Force Plot (All Instances): {e}")
    # elif expected_value_for_plot is None: # print("Skipping Interactive Force Plot (All Instances): expected_value not available.")
    # elif not shap_explanation: # print("Skipping Interactive Force Plot (All Instances): shap.Explanation object not available.")

    # print(f"--- Finished SHAP Plotting in {specific_output_dir} ---")

# --- LIME Handler ---
def process_and_plot_lime(
     results: Any,                       # Expect LIME Explanation object
     explainer_object: Any,              # The specific LimeExplainer instance
     instances_explained: np.ndarray,    # Should be shape (1, seq, feat) for LIME
     original_labels: Union[np.ndarray, pd.Series],
     feature_names: List[str],           # Base feature names
     sequence_length: int,
     output_dir: str,
     mode: str,
     instance_index: int = 0,            # Index if looping outside
     job_name='none',
     **kwargs):
     """Processes LIME results and generates standard plots/output."""
     # print(f"--- Processing and Plotting LIME Results for Instance Index {instance_index} ---")

     if results is None:
         # print("LIME results object is None. Skipping.")
         return
     
     output_dir = output_dir+'/'+job_name+'/LIME'

     # LIME Explanation object usually comes from explaining one instance
     lime_explanation = results

     try:
        # Save as HTML file (most common way to save LIME plots)
        os.makedirs(output_dir, exist_ok=True) # Ensure dir exists
        html_file = os.path.join(output_dir, f"{job_name}_lime_explanation_inst{instance_index}.html")
        lime_explanation.save_to_file(html_file)
        # print(f"Saved LIME explanation HTML to {html_file}")

        # Print top features to console
        # print(f"Top LIME features for instance {instance_index}:")
        # print(lime_explanation.as_list())

     except Exception as e:
        print(f"Failed to process/save LIME results for instance {instance_index}: {e}")

     # print("--- Finished LIME Plotting ---")

def process_and_plot_dice(
    results: Any, # Expect dice_ml.CounterfactualExplanations object for ONE original instance
    explainer_object: Any, 
    instances_explained: np.ndarray, # 3D numpy array (1, seq_len, features) for the ONE original instance
    original_labels: Union[np.ndarray, pd.Series, List], # Label for the ONE original instance
    feature_names: List[str], 
    sequence_length: int,
    output_dir: str, # This is base output dir, will append /job_name/DiCE
    mode: str,
    job_name: str = "job",
    instance_index: int = 0, # This is the index from XAIRunner's loop, used for unique filenames
    **kwargs
):
    # print(f"--- Processing DiCE for instance_index (from XAIRunner loop): {instance_index} ---")

    # Construct full output path
    dice_output_dir = os.path.join(output_dir, job_name, 'DiCE')
    os.makedirs(dice_output_dir, exist_ok=True)

    if results is None:
        # print("DiCE results object is None. Skipping.")
        return
    if instances_explained is None or instances_explained.shape[0] != 1:
        # print(f"DiCE instances_explained is None or not for a single instance (shape: {getattr(instances_explained, 'shape', 'N/A')}). Skipping.")
        return

    try:
        # results should be a CounterfactualExplanations object.
        # Its cf_examples_list should contain ONE DiceCFExamples object
        # if it's the result of explaining ONE original instance.
        
        if not hasattr(results, 'cf_examples_list') or not results.cf_examples_list:
            # print(f"    No cf_examples_list found or empty in DiCE results for instance_index {instance_index}. Skipping CSV save.")
            return
        
        # We expect only one DiceCFExamples object in this list because XAIRunner calls .explain for one instance.
        if len(results.cf_examples_list) > 1:
            warnings.warn(f"DiCE results.cf_examples_list has {len(results.cf_examples_list)} items. Expected 1 for single instance explanation. Using the first one.", RuntimeWarning)
        
        cf_example_for_this_instance = results.cf_examples_list[0] # Get the (only) DiceCFExamples

        # --- Get Counterfactuals DataFrame ---
        cfs_df = None
        if hasattr(cf_example_for_this_instance, 'final_cfs_df') and cf_example_for_this_instance.final_cfs_df is not None:
            cfs_df = cf_example_for_this_instance.final_cfs_df.copy()
        
        if cfs_df is None or cfs_df.empty:
            # print(f"    No final_cfs_df found or empty for instance_index {instance_index}. Skipping CSV save.")
            return
        if not isinstance(cfs_df, pd.DataFrame):
            # print(f"    Warning: Expected pandas DataFrame for counterfactuals, got {type(cfs_df)}. Skipping CSV save for instance_index {instance_index}.")
            return

        cfs_df['type'] = 'counterfactual'

        # --- Prepare Original Instance Row ---
        # Get necessary info from explainer_object (which is the DiceExplainer instance from TimeSeriesExplainer)
        if not hasattr(explainer_object, 'flat_feature_names') or not hasattr(explainer_object, 'outcome_name'):
             raise ValueError("Explainer object (DiceExplainer) is missing 'flat_feature_names' or 'outcome_name' attributes.")
        
        expected_flat_feature_names = explainer_object.flat_feature_names
        outcome_name = explainer_object.outcome_name

        original_instance_3d_single = instances_explained[0] # instances_explained is (1, seq, feat)
        original_flat_np = original_instance_3d_single.flatten()

        if len(original_flat_np) != len(expected_flat_feature_names):
            raise ValueError(f"Length mismatch: Flattened original data ({len(original_flat_np)}) vs "
                             f"expected flat feature names ({len(expected_flat_feature_names)}).")

        original_series = pd.Series(original_flat_np, index=expected_flat_feature_names)
        
        # Assign original label
        try:
            # original_labels should be a list/array with one element for this single instance
            if isinstance(original_labels, (list, np.ndarray, pd.Series)) and len(original_labels) > 0:
                true_label = original_labels[0]
                original_series[outcome_name] = true_label
                # print(f"      Instance_index {instance_index}: Assigned original label '{true_label}' to column '{outcome_name}'.")
            else:
                raise IndexError("original_labels is empty or not in expected format.")
        except IndexError:
            # print(f"      Warning: Could not get label for instance_index {instance_index} from original_labels (data: {original_labels}). Setting to NA.")
            original_series[outcome_name] = pd.NA
        except Exception as e_label:
            # print(f"      Warning: Error retrieving label for instance_index {instance_index}: {e_label}. Setting to NA.")
            original_series[outcome_name] = pd.NA
            
        original_series['type'] = 'original'
        original_df_row = original_series.to_frame().T

        # --- Align columns and types ---
        target_cols = cfs_df.columns.tolist() # Use columns from CFs as the master list
        # Ensure 'outcome_name' is in target_cols if not already (can happen if CFs don't include it)
        if outcome_name not in target_cols:
            target_cols.append(outcome_name)
            if outcome_name not in cfs_df.columns: # Add to cfs_df if missing
                cfs_df[outcome_name] = pd.NA # Or predicted outcome from CFs if available
        
        # Ensure 'type' is in target_cols
        if 'type' not in target_cols: # Should not happen as we add it
            target_cols.insert(0, 'type')


        final_original_df_row = pd.DataFrame(columns=target_cols)
        for col in target_cols:
            if col in original_df_row.columns:
                final_original_df_row[col] = original_df_row[col]
            else:
                final_original_df_row[col] = pd.NA
        
        final_cfs_df = pd.DataFrame(columns=target_cols)
        for col in target_cols:
            if col in cfs_df.columns:
                final_cfs_df[col] = cfs_df[col]
            else:
                final_cfs_df[col] = pd.NA

        # --- Combine Original and Counterfactuals ---
        # Reorder 'type' to be first if not already
        if 'type' in final_original_df_row.columns:
            final_original_df_row = final_original_df_row[['type'] + [c for c in final_original_df_row.columns if c != 'type']]
        if 'type' in final_cfs_df.columns:
            final_cfs_df = final_cfs_df[['type'] + [c for c in final_cfs_df.columns if c != 'type']]
        
        combined_df = pd.concat([final_original_df_row, final_cfs_df], ignore_index=True)

        # --- Save Combined CSV ---
        # Use the instance_index from XAIRunner for unique filenames
        filename = f"{job_name}_instance_{instance_index}_original_and_counterfactuals.csv"
        filepath = os.path.join(dice_output_dir, filename)
        combined_df.to_csv(filepath, index=False)
        # print(f"    Saved original and counterfactuals for instance_index {instance_index} to: {filepath}")

    except ValueError as ve: # Catch ValueErrors from our checks
        # print(f"ValueError processing DiCE for instance_index {instance_index}: {ve}")
        traceback.print_exc()
    except AttributeError as ae:
         # print(f"AttributeError processing DiCE for instance_index {instance_index}: {ae}")
         traceback.print_exc()
    except Exception as e:
         # print(f"Generic error processing DiCE for instance_index {instance_index}: {e}")
         traceback.print_exc()

    # print(f"--- Finished DiCE processing for instance_index {instance_index} ---")

# --- Add handlers for other XAI methods as needed ---