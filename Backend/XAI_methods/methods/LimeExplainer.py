import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Any, Union, Dict, List, Optional
import warnings
from XAI_methods.explainer_method_api import ExplainerMethodAPI

class LimeExplainer(ExplainerMethodAPI):
    """
    LIME implementation conforming to the ExplainerMethodAPI.

    Uses lime.lime_tabular.LimeTabularExplainer to provide local explanations
    for models handling time series sequences, treating each time step's
    feature as an independent tabular feature.

    NOTE: This standard LIME approach may not fully capture temporal dependencies.
    It explains ONE instance at a time.
    """

    def __init__(self, model: Any, background_data: np.ndarray, **params: Any):
        """
        Initializes the LimeExplainer.

        Args:
            model (Any): The trained machine learning model wrapper instance. Must have a
                         `.predict()` method (returns 1D array or 2D (samples, 1)) or
                         `.predict_proba()` method (returns 2D array (samples, n_classes)).
                         The wrapper should accept 3D NumPy input.
            background_data (np.ndarray): A representative sample of the input data
                already preprocessed/windowed into the 3D NumPy format
                (n_samples, sequence_length, n_features). Used for LIME statistics.
            **params (Any): Additional parameters. Expected keys:
                - feature_names (List[str]): List of base feature names. REQUIRED.
                - mode (str): 'regression' or 'classification'. REQUIRED.
                - class_names (List[str], optional): List of class names for classification mode.
                # Other potential LimeTabularExplainer init args can be passed here.
        """
        # print("Initializing LimeExplainer...")
        self.model = model 
        self.mode = params.get('mode', None)
        self.feature_names = params.get('feature_names', None)
        self.class_names = params.get('class_names', None)

        # --- Validation ---
        if self.mode not in ['regression', 'classification']:
            raise ValueError("LimeExplainer requires 'mode' ('regression' or 'classification') in params.")
        if self.feature_names is None or not isinstance(self.feature_names, list):
             raise ValueError("LimeExplainer requires 'feature_names' (list) in params.")
        if not isinstance(background_data, np.ndarray) or background_data.ndim != 3:
             raise ValueError("LimeExplainer requires background_data as a 3D NumPy array.")
        if background_data.shape[2] != len(self.feature_names):
            raise ValueError(f"Number of features in background_data ({background_data.shape[2]}) does not match length of feature_names ({len(self.feature_names)}).")

        self._original_sequence_shape = background_data.shape[1:] # (seq_len, n_features)
        self.sequence_length = self._original_sequence_shape[0]
        self._num_flat_features = np.prod(self._original_sequence_shape)

        # --- Prepare data and names for LimeTabularExplainer ---
        n_bg_samples = background_data.shape[0]
        self.background_data_flat = background_data.reshape(n_bg_samples, self._num_flat_features)

        # Create flattened feature names (e.g., F1_t-9, F1_t-8, ..., FN_t-0)
        self.feature_names_flat = [
            f"{feat}_t-{i}"
            for i in range(self.sequence_length -1, -1, -1) # Time descending
            for feat in self.feature_names
        ]
        # Check length consistency
        if len(self.feature_names_flat) != self._num_flat_features:
            raise RuntimeError(f"Internal Error: Generated {len(self.feature_names_flat)} flat feature names, but expected {self._num_flat_features}.")


        # --- Define Prediction Function for LIME ---
        # Takes 2D array (n_perturbations, n_flat_features)
        # Returns 1D array (regression) or 2D array (classification probabilities)
        def _predict_fn_lime(data_flat_2d: np.ndarray) -> np.ndarray:
            num_perturbations = data_flat_2d.shape[0]
            try:
                # Reshape flat perturbations back to 3D sequences
                # Shape: (num_perturbations, seq_len, n_features)
                data_reshaped_3d = data_flat_2d.reshape(
                    (num_perturbations,) + self._original_sequence_shape
                )
            except ValueError as e:
                raise ValueError(f"LIME Reshape Error: Cannot reshape flat data ({data_flat_2d.shape}) to sequence shape ({(num_perturbations,) + self._original_sequence_shape}). Error: {e}") from e

            # Call the *wrapper's* appropriate prediction method
            if self.mode == 'classification':
                if not hasattr(self.model, 'predict_proba'):
                     warnings.warn("LIME: Classification mode but model wrapper lacks 'predict_proba'. Using 'predict'. Output shape might be incorrect for LIME.", RuntimeWarning)
                     # Fallback to predict, but LIME expects probabilities
                     predictions = self.model.predict(data_reshaped_3d)
                     # Attempt to format as pseudo-probabilities if predict returns single class (0 or 1)
                     if predictions.ndim == 2 and predictions.shape[1] == 1:
                          prob_class_1 = predictions.flatten().astype(float)
                          prob_class_0 = 1.0 - prob_class_1
                          return np.vstack([prob_class_0, prob_class_1]).T # Shape (n_samples, 2)
                     else:
                          raise RuntimeError("LIME needs probabilities from predict_proba for classification, but model only has predict with incompatible output.")
                else:
                    # Use predict_proba
                    predictions = self.model.predict_proba(data_reshaped_3d) # Should return (n_samples, n_classes)
                    if predictions.ndim != 2 or predictions.shape[0] != num_perturbations:
                        raise ValueError(f"Model wrapper's predict_proba returned unexpected shape {predictions.shape}. Expected ({num_perturbations}, n_classes).")
                    return predictions # Return probabilities
            else: # Regression
                if not hasattr(self.model, 'predict'):
                    raise AttributeError("LIME: Regression mode but model wrapper lacks 'predict' method.")
                predictions = self.model.predict(data_reshaped_3d) # Should return (n_samples, 1) or (n_samples,)
                if predictions.ndim == 2 and predictions.shape[1] == 1:
                    return predictions.flatten() # Return 1D array (n_samples,)
                elif predictions.ndim == 1 and predictions.shape[0] == num_perturbations:
                    return predictions # Already 1D
                else:
                    raise ValueError(f"Model wrapper's predict returned unexpected shape {predictions.shape} for regression. Expected ({num_perturbations}, 1) or ({num_perturbations},).")

        # Store the prediction function (needed for explain_instance)
        self._predict_fn_lime = _predict_fn_lime

        # --- Initialize LimeTabularExplainer ---
        # print("Initializing lime.lime_tabular.LimeTabularExplainer...")
        # Extract relevant init kwargs from params
        lime_init_kwargs = {
             k: v for k, v in params.items()
             if k in ['kernel_width', 'verbose', 'feature_selection', 'discretize_continuous', 'discretizer', 'sample_around_instance', 'random_state']
             # Exclude params we handled manually: mode, feature_names, class_names
             # 'training_data' and 'feature_names' are positional below
        }
        try:
            self._explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.background_data_flat, # Background data (2D)
                feature_names=self.feature_names_flat,   # Flattened feature names
                class_names=self.class_names,            # List of class names if classification
                mode=self.mode,                          # 'classification' or 'regression'
                **lime_init_kwargs                       # Pass other init args
            )
            # print("LimeExplainer initialization complete.")
        except Exception as e:
            # print(f"Error initializing LimeTabularExplainer: {e}")
            raise RuntimeError("Failed to initialize LIME Tabular Explainer.") from e


    @property
    def expected_value(self):
        """ LIME does not compute a global expected value like SHAP. """
        warnings.warn("LIME does not have the concept of 'expected_value'. Returning None.", UserWarning)
        return None

    def explain(self,
                instances_to_explain: np.ndarray,
                **kwargs: Any) -> lime.explanation.Explanation: # Return type is LIME specific
        """
        Explains a SINGLE instance using LIME.

        Args:
            instances_to_explain (np.ndarray): A NumPy array containing EXACTLY ONE
                instance sequence, with shape (1, sequence_length, n_features).
            **kwargs (Any): Keyword arguments passed directly to the
                            `explainer.explain_instance` method. Common arguments:
                            - num_features (int): Number of features to include in explanation (default 10).
                            - num_samples (int): Number of perturbations LIME generates (default 5000).
                            - top_labels (int, optional): For classification, number of classes
                              to explain (e.g., explain only the top predicted class).
                            - labels (tuple, optional): For classification, specific class indices
                              to explain.

        Returns:
            lime.explanation.Explanation: The LIME explanation object for the instance.

        Raises:
            ValueError: If input does not contain exactly one instance or has wrong dimensions.
            TypeError: If input is not a NumPy array.
            Any exceptions from the underlying LIME `explain_instance` call.
        """
        # print(f"LimeExplainer: Received instance data with shape {instances_to_explain.shape} to explain.")

        # --- Input Validation for LIME (expects single instance) ---
        if not isinstance(instances_to_explain, np.ndarray):
             raise TypeError(f"{type(self).__name__} expects instances_to_explain as NumPy ndarray.")
        if instances_to_explain.ndim != 3:
            raise ValueError(f"LIME explain expects input with 3 dimensions (1, sequence_length, features), got {instances_to_explain.ndim}D.")
        if instances_to_explain.shape[0] != 1:
            raise ValueError(f"LIME explains one instance at a time, but received {instances_to_explain.shape[0]} instances.")
        # Check sequence length and feature count consistency
        if instances_to_explain.shape[1:] != self._original_sequence_shape:
            raise ValueError(f"Instance sequence shape {instances_to_explain.shape[1:]} does not match background data sequence shape {self._original_sequence_shape}.")
        # --- End Validation ---

        # Prepare the single instance: flatten to 1D array (n_flat_features,)
        instance_1d_flat = instances_to_explain.reshape(-1) # Reshape (1, seq, feat) -> (seq*feat,)

        # Extract LIME explain_instance specific arguments from kwargs
        num_features = kwargs.get('num_features', 10)
        num_samples = kwargs.get('num_samples', 5000) # LIME default
        labels = kwargs.get('labels', (1,) if self.mode == 'classification' else None) 
        top_labels = kwargs.get('top_labels', None) 
        other_lime_kwargs = {k: v for k, v in kwargs.items() if k not in ['num_features', 'num_samples', 'labels', 'top_labels']}

        # print(f"Calling LIME explain_instance (num_features={num_features}, num_samples={num_samples})...")
        try:
            explanation = self._explainer.explain_instance(
                data_row=instance_1d_flat,           # The flattened instance data
                predict_fn=self._predict_fn_lime,    # The prediction function defined in init
                num_features=num_features,
                num_samples=num_samples,
                labels=labels,                       # Pass specific labels if provided
                top_labels=top_labels,               # Pass top_labels if provided
                **other_lime_kwargs                  # Pass any other valid LIME kwargs
            )
            # print("LIME explanation finished.")
            return explanation
        except Exception as e:
            # print(f"Error during LIME explain_instance calculation: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging LIME issues
            raise RuntimeError("LIME explanation failed.") from e