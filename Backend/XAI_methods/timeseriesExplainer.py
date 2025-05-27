import numpy as np
import pandas as pd
from typing import List, Any, Dict, Optional
from XAI_methods.xai_factory import xai_factory
from XAI_methods.explainer_method_api import ExplainerMethodAPI

class TimeSeriesExplainer:
    """
    Manages Time Series XAI operations.

    Holds shared resources (model, data) and uses an external method
    (`get_method`) to obtain and manage specific
    explainer objects (like SHAP or LIME wrappers) which perform the
    actual explanation tasks.
    """

    def __init__(self, model: Any, background_data: np.ndarray, 
                background_outcomes: np.ndarray, 
                feature_names: List[str], 
                mode: str = 'regression',
                # --- Optional args for specific explainers like DiCE ---
                training_df_for_dice: Optional[pd.DataFrame] = None,
                outcome_name_for_dice: Optional[str] = None,
                continuous_features_for_dice: Optional[List[str]] = None,
                shap_method: str = None,
                **kwargs # kwargs for future flexibility
                ):
        """
        Initializes the TimeSeriesExplainer manager.

        Args:
            model: Trained model instance or wrapper.
            background_data: 3D NumPy array (samples, seq_len, features) of background features.
            feature_names: List of base feature names.
            mode: 'regression' or 'classification'.
            training_df_for_dice (pd.DataFrame, optional): A representative DataFrame
                (samples, features + outcome) needed ONLY for DiCE initialization.
                Features should be the original ones BEFORE windowing/flattening.
            outcome_name_for_dice (str, optional): Name of the outcome column in
                training_df_for_dice. Needed ONLY for DiCE.
            continuous_features_for_dice (List[str], optional): List of BASE feature names
                that are continuous. Needed ONLY for DiCE. Defaults to all features if None.
            **kwargs: Other potential arguments.
        """

        if not hasattr(model, 'predict'):
            raise TypeError("Model must have a 'predict' method.")
        if mode == 'classification' and not hasattr(model, 'predict_proba'):
            print("Warning: Classification mode but model lacks 'predict_proba'. Some explainers might behave unexpectedly.")

        self._model = model
        self._background_data = background_data
        self._sequence_length = background_data.shape[1]
        self._n_features = background_data.shape[2]
        self.shap_method = shap_method
        self._background_outcomes = background_outcomes

        self._feature_names = feature_names
        if mode not in ['regression', 'classification']:
            raise ValueError("Mode must be 'regression' or 'classification'")
        self._mode = mode

        # --- Store DiCE-specific context if provided ---
        self._training_df_for_dice = training_df_for_dice
        self._outcome_name_for_dice = outcome_name_for_dice
        # Default to all features being continuous if not specified for DiCE
        self._continuous_features_for_dice = continuous_features_for_dice if continuous_features_for_dice is not None else feature_names
        # print(f"TimeSeriesExplainer Init: DiCE context received - training_df: {'Yes' if self._training_df_for_dice is not None else 'No'}, outcome: {self._outcome_name_for_dice}")
        # ---

        # Cache to store initialized explainer objects returned by get_method
        self._explainer_cache: Dict[str, ExplainerMethodAPI] = {}
        # print("TimeSeriesExplainer manager initialized.")

    # --- Public properties to access configuration ---
    @property
    def model(self) -> Any: return self._model
    @property
    def background_data(self) -> np.ndarray: return self._background_data
    @property
    def feature_names(self) -> List[str]: return self._feature_names
    @property
    def mode(self) -> str: return self._mode

    # --- Internal Helper ---
    def _get_or_initialize_explainer(self, method_name: str) -> ExplainerMethodAPI:
        """
        Retrieves a specific explainer object from the cache or initializes
        it by calling the external `get_method` factory function.
        """
        method_key = method_name.lower()
        explainer_params_for_factory = {
            'mode': self._mode,
            'feature_names': self._feature_names,
            'sequence_length': self._sequence_length
            # Add other common params if needed
        }

        # Add DiCE specific params with CORRECT names
        if method_key == "diceexplainer":
            # (e.g., by adding it as an __init__ parameter and passing it from execute_calls.py)
            if not hasattr(self, '_background_outcomes') or self._background_outcomes is None:
                raise ValueError("TimeSeriesExplainer is missing '_background_outcomes' needed for DiceExplainer.")
            if not self._outcome_name_for_dice: # Check if the original suffixed name was provided
                raise ValueError("TimeSeriesExplainer is missing '_outcome_name_for_dice' needed for DiceExplainer.")
            if not self._continuous_features_for_dice: # Check if the original suffixed name was provided
                raise ValueError("TimeSeriesExplainer is missing '_continuous_features_for_dice' needed for DiceExplainer.")

            explainer_params_for_factory.update({
                'background_outcomes': self._background_outcomes, # Pass the actual NumPy array
                'outcome_name': self._outcome_name_for_dice, # Use the stored name, but with the key 'outcome_name'
                'continuous_feature_names': self._continuous_features_for_dice # Use the stored list, but with the key 'continuous_feature_names'
                # Add other DiCE init params if needed, e.g. 'backend': 'TF2'
            })

        if method_key not in self._explainer_cache:
            # print(f"Initializing explainer for '{method_key}' via get_method...")
            try:
                # Call the external factory, passing all necessary context
                # get_method should handle which params are needed for which method
                
                if method_name == 'ShapExplainer':
                    explainer_params_for_factory.update({
                        'shap_method': self.shap_method
                    })
                explainer_object = xai_factory(
                    method_name=method_key,
                    ml_model=self._model,
                    background_data=self._background_data, # 3D Features
                    **explainer_params_for_factory # Pass the constructed dictionary
                )
                self._explainer_cache[method_key] = explainer_object
                # print(f"Initialized and cached explainer for '{method_key}'. Type: {type(explainer_object).__name__}")

            except (ValueError, RuntimeError, TypeError, ImportError) as e:
                # print(f"Error initializing explainer '{method_key}': {e}")
                raise RuntimeError(f"Failed to get/initialize explainer '{method_key}'") from e
        # else:
            # print(f"Using cached explainer for '{method_key}'.")

        return self._explainer_cache[method_key]

    # --- Core Public Method ---
    def explain(self,
                instances_to_explain: np.ndarray, # Emphasize NumPy array input
                method_name: str,
                **kwargs: Any) -> Any:
        """
        Perform explanation using the specified method.

        Retrieves or initializes the appropriate specific explainer object
        (e.g., a SHAP wrapper) using the `get_method` factory and calls its
        `explain` method.

        Args:
            instances_to_explain (np.ndarray): The data instances (one or more)
                to explain. **MUST be a NumPy array** preprocessed into the
                3D sequence format expected by the model, typically
                `(n_instances, sequence_length, n_features)`. The conversion
                from original pandas DataFrames or other formats must happen
                *before* calling this method.
            method_name (str): Name of the explanation method (e.g., 'shap').
            **kwargs (Any): Additional keyword arguments passed directly to the
                            specific explainer object's `explain` method.

        Returns:
            Any: The result from the specific explainer's `explain` method.
                 For 'shap', the result is reshaped to match the input sequence
                 dimensions (e.g., (n_instances, sequence_length, n_features)).
                 Format for other methods depends on their implementation.

        Raises:
            RuntimeError: If the requested explainer cannot be initialized via `get_method`.
            TypeError: If `instances_to_explain` is not a NumPy array.
            ValueError: If instance shapes are inconsistent.
            Any exceptions raised by the specific explainer object's `explain` method.
        """
        # print(f"\n--- TimeSeriesExplainer: Requesting explanation via method '{method_name}' ---")

        # Add explicit type check at the entry point for clarity
        if not isinstance(instances_to_explain, np.ndarray):
            raise TypeError("TimeSeriesExplainer.explain requires instances_to_explain to be a NumPy ndarray.")

        try:
            # Step 1: Get the specific explainer object
            explainer = self._get_or_initialize_explainer(method_name)

            # Step 2: Delegate to the specific object's explain method
            # print(f"Calling '{method_name}' explainer's (.explain) method...")
            result = explainer.explain(instances_to_explain, **kwargs) # Pass the validated NumPy array

            # print(f"--- TimeSeriesExplainer: Explanation finished for '{method_name}' ---")
            return result

        except Exception as e:
            print(f"Error during explanation process for method '{method_name}': {e}")
            raise

    def get_initialized_methods(self) -> List[str]:
        """Returns a list of method names for which explainers are currently initialized."""
        return list(self._explainer_cache.keys())