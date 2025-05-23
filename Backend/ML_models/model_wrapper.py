import traceback
import pandas as pd
import numpy as np
from typing import Any, List, Union, Literal
import warnings
import logging
logging.basicConfig(level=logging.INFO) # Set basic level
logger = logging.getLogger(__name__)

class ModelWrapperForXAI:
    """
    Wraps a model instance to provide standardized `.predict()` and `.predict_proba()`
    interfaces for XAI tools (SHAP, LIME, DiCE).
    """
    def __init__(self,
                 actual_model_instance: Any,
                 feature_names: List[str],
                 score_interpretation: Literal['lower_is_anomaly', 'higher_is_anomaly'] = 'higher_is_anomaly'):
        """
        Args:
            actual_model_instance: Trained model instance. Should have at least 'predict' or 'detect'.
            feature_names (List[str]): Feature names list corresponding to the base features.
            score_interpretation (str): How to interpret raw scores ('lower_is_anomaly' or 'higher_is_anomaly').
        """
        self._model = actual_model_instance
        self._feature_names = feature_names
        self._num_classes = 2 # Assuming binary classification (normal vs anomaly)
        self._score_interpretation = score_interpretation

        # --- MODIFIED: Relaxed Checks ---
        # Check for at least one primary prediction method
        has_predict = hasattr(self._model, 'predict') and callable(getattr(self._model, 'predict'))
        has_detect = hasattr(self._model, 'detect') and callable(getattr(self._model, 'detect'))
        if not (has_predict or has_detect):
             raise AttributeError(f"Provided model {type(self._model).__name__} must have a callable 'predict' or 'detect' method.")

        # Warn if predict_proba is missing, as XAI often needs it
        if not hasattr(self._model, 'predict_proba') or not callable(getattr(self._model, 'predict_proba')):
            warnings.warn(f"Wrapped model {type(self._model).__name__} may lack a callable 'predict_proba' method needed for some XAI techniques (e.g., SHAP with probability output).", RuntimeWarning)

        # Warn if score/decision functions are missing (useful for some models/XAI)
        has_score_func = hasattr(self._model, 'decision_function') or hasattr(self._model, 'score_samples')
        if not has_score_func and not hasattr(self._model, 'predict_proba'):
             warnings.warn(f"Wrapped model {type(self._model).__name__} lacks predict_proba and score/decision functions. XAI might be limited.", RuntimeWarning)
        # --- End Modified Checks ---


        if not feature_names: raise ValueError("Feature names list cannot be empty.")
        if self._score_interpretation not in ['lower_is_anomaly', 'higher_is_anomaly']:
            raise ValueError("score_interpretation must be 'lower_is_anomaly' or 'higher_is_anomaly'")
        if not hasattr(self._model, 'sequence_length'):
            warnings.warn(f"Wrapped model {type(self._model).__name__} lacks 'sequence_length' attribute. Input shape handling might rely on defaults.", RuntimeWarning)


        logger.info(f"ModelWrapperForXAI initialized for model type: {type(self._model).__name__}")
        logger.info(f"Underlying score interpretation: '{self._score_interpretation}'")

    @property
    def sequence_length(self):
        """Returns the sequence length from the underlying model, or None if not defined."""
        # Default to 1 if not present, as many models are non-sequential
        return getattr(self._model, 'sequence_length', 1)

    @property
    def model(self) -> Any:
        """Provides access to the underlying model instance, attempting to get the core estimator."""
        # Try common attributes for nested models
        if hasattr(self._model, 'model') and self._model.model is not None:
            return self._model.model
        elif hasattr(self._model, 'base_estimator') and self._model.base_estimator is not None:
            return self._model.base_estimator
        # Otherwise, return the instance we were given
        return self._model

    def _call_internal_method(self, X_input_data: np.ndarray, internal_method_name: str) -> np.ndarray:
        """Calls the specified method on the wrapped model, handling errors."""
        if not hasattr(self._model, internal_method_name) or not callable(getattr(self._model, internal_method_name)):
            logger.error(f"Internal model {type(self._model).__name__} lacks required method '{internal_method_name}'.")
            n_out = X_input_data.shape[0] if X_input_data.ndim >= 1 else 1
            n_cols = self._num_classes if internal_method_name == 'predict_proba' else 1
            return np.full((n_out, n_cols), np.nan)

        internal_method = getattr(self._model, internal_method_name)
        logger.debug(f"Wrapper: Calling internal '{internal_method_name}' with input shape {X_input_data.shape}...")
        try:
            results = internal_method(X_input_data)
            results_np = np.array(results) # Ensure NumPy output
            logger.debug(f"Wrapper: Internal '{internal_method_name}' returned array shape {results_np.shape}")
            return results_np
        except Exception as e:
            logger.error(f"ERROR in ModelWrapper calling internal model.{internal_method_name} with input shape {X_input_data.shape}: {e}")
            traceback.print_exc()
            n_out = X_input_data.shape[0] if X_input_data.ndim >= 1 else 1
            n_cols = self._num_classes if internal_method_name == 'predict_proba' else 1
            return np.full((n_out, n_cols), np.nan)


    def predict(self, X_input: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Provides the .predict() interface (predicts class labels 0 or 1)."""
        # Convert input to NumPy first
        if isinstance(X_input, pd.DataFrame):
            X_np_input = X_input.to_numpy()
        elif isinstance(X_input, np.ndarray):
            X_np_input = X_input
        else:
            raise TypeError(f"predict expects NumPy array or pandas DataFrame, got {type(X_input)}")

        if X_np_input.size == 0:
            logger.warning("Predict called with empty array.")
            return np.empty((0, 1), dtype=int)

        # Decide which method to call (prefer 'detect' if available)
        method_to_call = 'detect' if hasattr(self._model, 'detect') and callable(getattr(self._model, 'detect')) else 'predict'
        logger.debug(f"Calling internal method '{method_to_call}' for label prediction.")

        raw_predictions = self._call_internal_method(X_np_input, method_to_call)

        # Post-process results to ensure (N, 1) integer output
        results_np = np.array(raw_predictions, dtype=float) # Start as float for NaN checks
        nan_mask = np.isnan(results_np)
        if np.any(nan_mask):
            warnings.warn(f"'{method_to_call}' method returned NaNs. Filling with 0 (normal label).", RuntimeWarning)
            results_np[nan_mask] = 0

        # Convert boolean to int if needed
        if pd.api.types.is_bool_dtype(results_np.dtype):
            results_np = results_np.astype(int)
        elif not pd.api.types.is_numeric_dtype(results_np.dtype):
             try: results_np = results_np.astype(int)
             except ValueError:
                 warnings.warn(f"Could not convert '{method_to_call}' results to int. Filling with 0.", RuntimeWarning)
                 results_np.fill(0)

        # Ensure shape is (N, 1)
        if results_np.ndim == 0: results_np = np.array([[results_np.item()]])
        elif results_np.ndim == 1: results_np = results_np[:, np.newaxis]
        elif results_np.ndim == 2 and results_np.shape[1] > 1:
            warnings.warn(f"'{method_to_call}' returned shape {results_np.shape}. Taking first column as label.", RuntimeWarning)
            results_np = results_np[:, 0:1]
        elif results_np.ndim > 2:
             raise ValueError(f"Internal '{method_to_call}' returned unexpected shape {results_np.shape}")

        logger.debug(f"ModelWrapper.predict: Returning shape {results_np.shape}")
        return results_np.astype(int)


    def predict_proba(self, X_input: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Provides the .predict_proba() interface for XAI tools (SHAP, LIME, DiCE).
        Handles input type (DataFrame/NumPy), calls the internal
        model's 'predict_proba' method, and returns the resulting probabilities.
        Ensures output is (N, 2) for binary classification models supporting predict_proba.
        """
        # --- Input Conversion ---
        if isinstance(X_input, pd.DataFrame):
            X_np_input = X_input.to_numpy()
        elif isinstance(X_input, np.ndarray):
            X_np_input = X_input
        else:
            raise TypeError(f"predict_proba expects NumPy array or pandas DataFrame, got {type(X_input)}")

        num_samples_in = X_np_input.shape[0] if X_np_input.ndim > 0 else (1 if X_np_input.size > 0 else 0)
        if num_samples_in == 0:
            return np.empty((0, self._num_classes), dtype=float)

        # --- Check if predict_proba exists ---
        if not hasattr(self._model, 'predict_proba') or not callable(getattr(self._model, 'predict_proba')):
             logger.error(f"Underlying model {type(self._model).__name__} does not have a callable 'predict_proba' method.")
             return np.full((num_samples_in, self._num_classes), 0.5) # Return neutral probabilities

        # --- Call Internal Model's predict_proba ---
        logger.debug(f"Calling internal 'predict_proba'. Input shape: {X_np_input.shape}")
        # Use the helper method which includes error handling and NumPy conversion
        internal_probabilities = self._call_internal_method(X_np_input, 'predict_proba')
        logger.debug(f"Internal 'predict_proba' returned shape: {internal_probabilities.shape}")

        # --- Process Result (Focus on returning N, 2) ---
        n_samples_out = internal_probabilities.shape[0] if internal_probabilities.ndim >= 1 else 0
        probabilities = np.full((num_samples_in, self._num_classes), 0.5) # Default neutral

        # Handle NaN/Inf in internal result - _call_internal_method should return NaN array on error
        invalid_row_mask = np.isnan(internal_probabilities).any(axis=1) | np.isinf(internal_probabilities).any(axis=1)
        if np.any(invalid_row_mask):
             warnings.warn("NaNs or Infs detected in internal predict_proba output. Returning neutral for affected rows.", RuntimeWarning)

        valid_row_mask = ~invalid_row_mask
        valid_internal_probs = internal_probabilities[valid_row_mask]

        # Check dimensions and column count of VALID results
        reconstruction_needed = False
        if valid_internal_probs.ndim == 2 and valid_internal_probs.shape[1] == self._num_classes:
             if valid_internal_probs.shape[0] == np.sum(valid_row_mask):
                 logger.debug(f"Assigning valid (N, {self._num_classes}) probabilities directly.")
                 probabilities[valid_row_mask] = valid_internal_probs
             else:
                 warnings.warn("Length mismatch after filtering NaNs. Check internal predict_proba consistency. Returning neutral.", RuntimeWarning)
                 probabilities.fill(0.5) # Reset all to neutral if length mismatch

        elif valid_internal_probs.ndim == 2 and valid_internal_probs.shape[1] == 1:
             warnings.warn("Internal predict_proba returned shape (N, 1). Reconstructing to (N, 2).", RuntimeWarning)
             if valid_internal_probs.shape[0] == np.sum(valid_row_mask):
                 reconstruction_needed = True
                 proba_class_1 = valid_internal_probs.flatten()
             else:
                 warnings.warn("Length mismatch after filtering NaNs (N, 1 case). Returning neutral.", RuntimeWarning)
                 probabilities.fill(0.5)

        elif valid_internal_probs.ndim == 1:
             warnings.warn("Internal predict_proba returned shape (N,). Reconstructing to (N, 2).", RuntimeWarning)
             if valid_internal_probs.shape[0] == np.sum(valid_row_mask):
                 reconstruction_needed = True
                 proba_class_1 = valid_internal_probs
             else:
                  warnings.warn("Length mismatch after filtering NaNs (N,) case). Returning neutral.", RuntimeWarning)
                  probabilities.fill(0.5)
        else:
             if np.any(valid_row_mask): # Only warn if there were actually valid rows
                warnings.warn(f"Internal predict_proba returned unexpected shape {valid_internal_probs.shape} for valid rows. Returning neutral.", RuntimeWarning)
                probabilities.fill(0.5)

        # Perform reconstruction if needed
        if reconstruction_needed:
            proba_class_0 = 1.0 - proba_class_1
            reconstructed_probs = np.vstack([proba_class_0, proba_class_1]).T
            probabilities[valid_row_mask] = reconstructed_probs # Assign reconstructed to valid rows

        # --- Final Polish ---
        np.clip(probabilities, 0.0, 1.0, out=probabilities)
        sums = np.sum(probabilities, axis=1, keepdims=True)
        sums[sums < 1e-9] = 1.0
        probabilities /= sums

        # Final check on output shape relative to input
        if probabilities.shape[0] != num_samples_in:
             warnings.warn(f"Final probability array shape {probabilities.shape} doesn't match input samples {num_samples_in}. Adjusting.", RuntimeWarning)
             final_probs_adjusted = np.full((num_samples_in, self._num_classes), 0.5)
             len_to_copy = min(probabilities.shape[0], num_samples_in)
             final_probs_adjusted[:len_to_copy] = probabilities[:len_to_copy]
             probabilities = final_probs_adjusted

        logger.debug(f"Wrapper predict_proba returning shape: {probabilities.shape}")
        return probabilities.astype(np.float32)