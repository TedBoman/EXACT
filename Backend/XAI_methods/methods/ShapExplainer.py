import shap
import numpy as np
import pandas as pd
from typing import Any, Union, Dict, List, Optional
import warnings
import logging 
from XAI_methods.explainer_method_api import ExplainerMethodAPI
# logger = logging.get# logger(__name__)

class ShapExplainer(ExplainerMethodAPI):
    """
    SHAP implementation conforming to the ExplainerMethodAPI.

    Uses appropriate SHAP explainers (Tree, Kernel, etc.) based on configuration.
    Assumes the model wrapper handles input preparation.
    """

    def __init__(self, model: Any, background_data: np.ndarray, **params: Any):
        """
        Initializes the ShapExplainer.

        Args:
            model (Any): The ModelWrapperForXAI instance.
            background_data (np.ndarray): 3D NumPy array (samples, seq_len, features) for background data.
            **params (Any): Additional parameters including:
                - mode (str): 'classification' or 'regression'.
                - shap_method (str): 'kernel', 'tree', 'linear', 'partition'.
        """
        # logger.info("Initializing ShapExplainer...")
        # logger.debug(f"Shap __init__ received params: {params}")
        self.model = model # This is the ModelWrapperForXAI instance
        self.mode = params.get('mode', 'regression').lower()
        self.shap_method = params.get('shap_method', 'kernel').lower()

        # Validate the selected SHAP method
        supported_methods = ['kernel', 'tree', 'linear', 'partition']
        if self.shap_method not in supported_methods:
            raise ValueError(f"Unsupported SHAP method: '{self.shap_method}'. Supported methods are: {supported_methods}")
        # logger.info(f"Selected SHAP method: '{self.shap_method}'")
        # logger.info(f"Prediction mode: '{self.mode}'")

        self.background_data = background_data

        # --- Define sequence shape attributes ---
        if background_data.ndim == 3:
            self._original_sequence_shape = background_data.shape[1:] # (seq_len, n_features)
            self._num_flat_features = np.prod(self._original_sequence_shape)
            # logger.info(f"Derived sequence shape: {self._original_sequence_shape}, Flat features: {self._num_flat_features}")
        else:
            raise ValueError("Background data must be 3D (samples, seq_len, features) to determine sequence shape.")

        # --- Set link argument (used by Kernel/Partition) ---
        if self.mode == 'classification':
            self.link_arg = "logit"
            # logger.info(f"Mode is '{self.mode}', setting link argument to string 'logit'")
        else: # regression
            self.link_arg = "identity"
            # logger.info(f"Mode is '{self.mode}', setting link argument to string 'identity'")

        # --- Define Internal Prediction Function (_predict_fn_shap) ---
        # This function is primarily used by KernelExplainer and PartitionExplainer.
        # TreeExplainer interacts directly with the raw model.
        def _predict_fn_shap(data_2d: np.ndarray) -> np.ndarray:
            """
            Prediction function wrapper for SHAP explainers like Kernel/Partition.
            Reshapes SHAP's 2D input to the model's expected 3D format and
            calls the wrapper's predict_proba (if classification and available)
            or predict method. Returns 2D array (samples, num_outputs).
            """
            num_samples = data_2d.shape[0]
            if num_samples == 0:
                # Determine expected output columns based on mode and proba availability
                n_outputs = 1
                if self.mode == 'classification' and hasattr(self.model, 'predict_proba') and callable(getattr(self.model, 'predict_proba')):
                     # Check if the underlying model's predict_proba likely returns 2 columns
                     # This is a heuristic, might need refinement
                     try:
                         # Predict on a dummy sample (if possible) to infer output shape
                         dummy_input = self.background_data[0:1] # Use first background sample
                         dummy_input_flat = dummy_input.reshape(1, self._num_flat_features)
                         dummy_input_3d = dummy_input_flat.reshape((1,) + self._original_sequence_shape)
                         dummy_proba = self.model.predict_proba(dummy_input_3d)
                         if isinstance(dummy_proba, np.ndarray) and dummy_proba.ndim == 2:
                             n_outputs = dummy_proba.shape[1]
                         else: # Default to 2 for classification if shape check fails
                              n_outputs = 2
                     except Exception:
                         n_outputs = 2 # Default to 2 for classification if check fails
                return np.empty((0, n_outputs), dtype=float)

            try:
                data_reshaped_3d = data_2d.reshape((num_samples,) + self._original_sequence_shape)
            except ValueError as e:
                # logger.error(f"Error reshaping data in _predict_fn_shap. Input shape: {data_2d.shape}, Target sequence shape: {self._original_sequence_shape}. Error: {e}")
                raise ValueError(f"Error reshaping data for model prediction in SHAP wrapper. Input shape: {data_2d.shape}, Target shape: {(num_samples,) + self._original_sequence_shape}. Error: {e}") from e

            predictions = None
            # --- Prioritize predict_proba for classification ---
            if self.mode == 'classification' and hasattr(self.model, 'predict_proba') and callable(getattr(self.model, 'predict_proba')):
                try:
                    # logger.debug(f"_predict_fn_shap calling wrapper.predict_proba for shape {data_reshaped_3d.shape}")
                    # The wrapper's predict_proba should return (N, 2) or similar
                    predictions = self.model.predict_proba(data_reshaped_3d)
                    # logger.debug(f"_predict_fn_shap received predict_proba output shape: {predictions.shape}")
                except Exception as e_proba:
                    # logger.warning(f"_predict_fn_shap: Error calling wrapper.predict_proba: {e_proba}. Falling back to predict().")
                    # Fallback if predict_proba fails unexpectedly
                    if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
                        predictions = self.model.predict(data_reshaped_3d)
                    else:
                        # logger.error("_predict_fn_shap: predict_proba failed and predict is not available.")
                        raise AttributeError("Model wrapper lacks both working predict_proba and predict methods.") from e_proba

            elif hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
                # Use predict for regression or if predict_proba unavailable for classification
                # logger.debug(f"_predict_fn_shap calling wrapper.predict for shape {data_reshaped_3d.shape}")
                predictions = self.model.predict(data_reshaped_3d)
                # logger.debug(f"_predict_fn_shap received predict output shape: {predictions.shape}")
            else:
                 # logger.error("_predict_fn_shap: Neither predict_proba nor predict available on wrapped model.")
                 raise AttributeError("Model wrapper lacks both predict_proba and predict methods.")

            # Ensure output is 2D (n_samples, n_outputs)
            if predictions.ndim == 1:
                return predictions[:, np.newaxis]
            elif predictions.ndim == 0:
                return np.array([[predictions.item()]])
            elif predictions.ndim == 2:
                # Expected output shape (N, 2) from predict_proba or (N, 1) from predict
                return predictions
            else:
                # logger.error(f"_predict_fn_shap: Wrapper's prediction function returned unexpected shape: {predictions.shape}. Expected 1D or 2D.")
                raise ValueError(f"Wrapper's prediction function returned unexpected shape: {predictions.shape}. Expected 1D or 2D.")

        # Assign the function to the instance
        self._predict_fn_shap = _predict_fn_shap
        # logger.info("Internal prediction function (_predict_fn_shap) defined.")

        # --- Prepare Background Data Summary (Flattening) ---
        # logger.info("Preparing background data summary...")
        n_bg_samples = self.background_data.shape[0]
        background_data_flat = self.background_data.reshape(n_bg_samples, self._num_flat_features)
        background_summary_np: Optional[np.ndarray] = None # Initialize

        # --- Conditional Explainer Initialization ---
        self._explainer = None # Initialize explainer attribute

        if self.shap_method == 'kernel':
            # logger.info("Initializing shap.KernelExplainer...")
            # Prepare background data summary for KernelExplainer
            k_summary = min(50, n_bg_samples)
            if n_bg_samples > k_summary * 2:
                # logger.info(f"Summarizing background data using shap.kmeans (k={k_summary})...")
                try:
                    if not np.issubdtype(background_data_flat.dtype, np.floating):
                         background_data_flat = background_data_flat.astype(np.float64)
                    summary_object = shap.kmeans(background_data_flat, k_summary, round_values=False)
                    if hasattr(summary_object, 'data') and isinstance(summary_object.data, np.ndarray):
                        background_summary_np = summary_object.data
                        # logger.info(f"Extracted NumPy array from kmeans summary. Shape: {background_summary_np.shape}")
                    else:
                        warnings.warn("Could not extract NumPy data from shap.kmeans result structure. Falling back to raw data.", RuntimeWarning)
                        background_summary_np = background_data_flat
                except Exception as kmeans_err:
                    # logger.warning(f"shap.kmeans failed ({kmeans_err}). Using raw background data instead. This might be slow.")
                    background_summary_np = background_data_flat
            else:
                # logger.info("Using raw background data for SHAP KernelExplainer (no summarization).")
                background_summary_np = background_data_flat

            # Ensure background_summary_np is assigned
            if background_summary_np is None:
                 # logger.error("Background summary data is None, cannot initialize KernelExplainer.")
                 raise ValueError("Failed to prepare background data summary for KernelExplainer.")

            # logger.info(f"Background data for KernelExplainer prepared. Shape: {background_summary_np.shape}")

            try:
                # Pass the instance method _predict_fn_shap
                self._explainer = shap.KernelExplainer(
                    self._predict_fn_shap,
                    background_summary_np,
                    link=self.link_arg
                 )
                # logger.info(f"Initialized KernelExplainer: {type(self._explainer)}")
            except Exception as e:
                # logger.error(f"Error initializing shap.KernelExplainer: {e}", exc_info=True)
                raise RuntimeError("Failed to initialize KernelExplainer") from e

        elif self.shap_method == 'tree':
            # logger.info("Initializing shap.TreeExplainer...")
            # TreeExplainer needs the raw model object
            try:
                raw_model = self.model.model # Accesses the @property in ModelWrapperForXAI
                if raw_model is None:
                    raise ValueError("Could not retrieve underlying model from wrapper.")
                # logger.info(f"Retrieved raw model of type: {type(raw_model)}")
            except AttributeError as e:
                 # logger.error(f"Failed to get raw model via wrapper property: {e}", exc_info=True)
                 raise AttributeError(
                    "shap_method='tree' requires the 'model' wrapper instance "
                    "to provide access to the raw tree model via its '.model' property."
                 ) from e

            # Set model_output argument for TreeExplainer
            if self.mode == 'classification':
                # Use 'probability' for classification models that output probabilities
                self._shap_model_output = "probability"
                # logger.info(f"Mode is '{self.mode}', setting TreeExplainer model_output to '{self._shap_model_output}'")
                # --- REMOVED the check block that caused warnings ---
            else: # regression
                self._shap_model_output = "raw"
                # logger.info(f"Mode is '{self.mode}', setting TreeExplainer model_output to 'raw'")

            # TreeExplainer uses the flattened background data
            # logger.info("Using raw background data for SHAP TreeExplainer.")
            try:
                self._explainer = shap.TreeExplainer(
                    raw_model,
                    data=background_data_flat,
                    model_output=self._shap_model_output,
                    feature_perturbation='interventional'
                )
                # logger.info(f"Initialized TreeExplainer: {type(self._explainer)}")
            except Exception as e:
                # logger.error(f"Error initializing shap.TreeExplainer. Ensure the raw model "
                            #  f"({type(raw_model)}) is supported (e.g., XGBoost, scikit-learn trees/forests). Error: {e}", exc_info=True)
                raise RuntimeError("Failed to initialize TreeExplainer") from e

        # --- Add elif blocks for 'linear' and 'partition' if needed ---
        # elif self.shap_method == 'linear': ...
        # elif self.shap_method == 'partition': ...

        else:
             # This case should be caught by the initial validation, but as a fallback:
             raise ValueError(f"Internal error: Reached end of __init__ with unsupported shap_method '{self.shap_method}'")

        if self._explainer is None:
            raise RuntimeError(f"Explainer object was not initialized for method '{self.shap_method}'.")

        # logger.info("ShapExplainer initialization complete.")

    @property
    def expected_value(self):
        """Returns the expected value (base value) from the SHAP explainer."""
        if hasattr(self, '_explainer') and hasattr(self._explainer, 'expected_value'):
            return self._explainer.expected_value
        else:
            warnings.warn("Could not retrieve expected_value from internal SHAP explainer.", RuntimeWarning)
            return None

    def explain(self,
                instances_to_explain: np.ndarray,
                **kwargs: Any) -> Union[np.ndarray, List[np.ndarray]]:
        """Calculate SHAP values for the given instances."""
        # logger.info(f"ShapExplainer: Received {instances_to_explain.shape[0]} instances to explain using '{self.shap_method}' method.")

        if not isinstance(instances_to_explain, np.ndarray):
            raise TypeError(f"{type(self).__name__} expects instances_to_explain as a NumPy ndarray.")

        if instances_to_explain.shape[1:] != self._original_sequence_shape:
             raise ValueError(
                f"Instance feature shape {instances_to_explain.shape[1:]} does not match "
                f"background data feature shape {self._original_sequence_shape}."
             )

        n_instances = instances_to_explain.shape[0]
        if n_instances == 0:
            # logger.warning("Explain called with empty instances array.")
            return []

        instances_flat = instances_to_explain.reshape(n_instances, self._num_flat_features)
        # logger.debug(f"Flattened instances for SHAP input to shape: {instances_flat.shape}")

        # logger.info(f"Calling SHAP {type(self._explainer).__name__}.shap_values...")

        explainer_kwargs = kwargs.copy()
        if self.shap_method != 'kernel':
            ns = explainer_kwargs.pop('nsamples', None)
            l1 = explainer_kwargs.pop('l1_reg', None)
            # if ns or l1: # logger.debug(f"Ignoring nsamples/l1_reg for non-Kernel explainer ({self.shap_method})")

        try:
            shap_values_output = self._explainer.shap_values(
                instances_flat,
                **explainer_kwargs
            )
            # logger.info("SHAP calculation finished.")
            # Add detailed logging of output shape/type
            # logger.debug(f"Raw shap_values output type: {type(shap_values_output)}")
            # if isinstance(shap_values_output, np.ndarray):
            #     # logger.debug(f"Raw shap_values output shape: {shap_values_output.shape}")
            # elif isinstance(shap_values_output, list):
            #      # logger.debug(f"Raw shap_values output list length: {len(shap_values_output)}")
            #      if len(shap_values_output) > 0 and isinstance(shap_values_output[0], np.ndarray):
            #           # logger.debug(f"Raw shap_values output list[0] shape: {shap_values_output[0].shape}")


            # --- Reshape the output back to the original sequence feature shape ---
            target_shape = (n_instances,) + self._original_sequence_shape
            expected_flat_features = self._num_flat_features
            reshaped_result = None

            if isinstance(shap_values_output, list):
                # logger.debug("Processing list output (multi-class classification).")
                reshaped_shap_values_list = []
                for i, class_shap_values_flat in enumerate(shap_values_output):
                    if not isinstance(class_shap_values_flat, np.ndarray):
                        # logger.warning(f"Item {i} in SHAP values list is not a NumPy array (type: {type(class_shap_values_flat)}). Skipping.")
                        continue
                    # Validate shape before reshaping
                    if class_shap_values_flat.shape != (n_instances, expected_flat_features):
                        warnings.warn(f"SHAP values for class {i} have unexpected flat shape {class_shap_values_flat.shape}. Expected ({n_instances}, {expected_flat_features}). Skipping reshape for this class.", RuntimeWarning)
                        reshaped_shap_values_list.append(class_shap_values_flat) # Append raw if shape doesn't match
                        continue
                    try:
                        # Reshape only if original features were multi-dimensional (e.g., sequence)
                        if len(self._original_sequence_shape) > 1:
                            reshaped_shap_values_list.append(class_shap_values_flat.reshape(target_shape))
                        else: # If original features were 1D, keep flat
                            reshaped_shap_values_list.append(class_shap_values_flat)
                    except ValueError as e:
                        raise ValueError(f"Failed to reshape SHAP values for class {i}. Flat shape: {class_shap_values_flat.shape}, Target shape: {target_shape}. Error: {e}") from e
                # logger.info(f"Processed SHAP values list (items: {len(reshaped_shap_values_list)})")
                reshaped_result = reshaped_shap_values_list

            elif isinstance(shap_values_output, np.ndarray):
                # logger.debug(f"Processing ndarray output with shape: {shap_values_output.shape}")
                shap_values_flat = shap_values_output # Start with the raw output

                # --- Add the squeeze logic here ---
                if shap_values_flat.ndim == 3 and shap_values_flat.shape[2] == 1:
                    # logger.debug(f"Reshaping SHAP values from {shap_values_flat.shape} to 2D.")
                    shap_values_flat = shap_values_flat.squeeze(axis=2)
                    # logger.debug(f"Shape after squeeze: {shap_values_flat.shape}")
                # --- End squeeze logic ---

                # Now validate the potentially squeezed shape
                selected_shap_values = None
                if shap_values_flat.ndim == 2 and shap_values_flat.shape == (n_instances, expected_flat_features):
                    # logger.debug("Assuming 2D array output is for the target class/output.")
                    selected_shap_values = shap_values_flat
                elif shap_values_flat.ndim == 3 and shap_values_flat.shape[0] == n_instances and \
                    shap_values_flat.shape[1] == expected_flat_features and shap_values_flat.shape[2] > 1:
                    # Handle multi-class output returned as single array
                    class_index_to_use = 1 if shap_values_flat.shape[2] == 2 else 0 # Default to class 1 for binary
                    # logger.warning(f"SHAP values ndarray has shape {shap_values_flat.shape}. Selecting class index {class_index_to_use}.")
                    selected_shap_values = shap_values_flat[:, :, class_index_to_use]
                else:
                    # Unexpected shape
                    raise ValueError(f"SHAP values ndarray has unexpected shape {shap_values_flat.shape} after potential squeeze. Expected ({n_instances}, {expected_flat_features}) or ({n_instances}, {expected_flat_features}, n_classes > 1).")

                if selected_shap_values is None:
                    raise RuntimeError("Internal error: selected_shap_values is None after checks.")

                # --- Reshape the selected (N, F) array ---
                if len(self._original_sequence_shape) > 1: # Reshape only if original features were multi-dimensional
                    # logger.debug(f"Shape of selected SHAP values before reshape: {selected_shap_values.shape}")
                    try:
                        reshaped_single_output = selected_shap_values.reshape(target_shape)
                        # logger.info(f"Reshaped SHAP values (single output/class) to array shape: {reshaped_single_output.shape}")
                        reshaped_result = reshaped_single_output
                    except ValueError as e:
                        raise ValueError(f"Failed to reshape SELECTED SHAP values. Flat shape: {selected_shap_values.shape}, Target shape: {target_shape}. Error: {e}") from e
                else:
                    # If original features were 1D (e.g., sequence_length=1), keep flat
                    # logger.info("Original data features were 1D (sequence_length=1), returning flat SHAP values.")
                    reshaped_result = selected_shap_values # Already (N, Features)

            else:
                raise TypeError(f"Unexpected type returned by explainer.shap_values: {type(shap_values_output)}")

            return reshaped_result

        except Exception as e:
            # logger.error(f"Error during SHAP values calculation or reshaping: {e}", exc_info=True)
            raise
