import dice_ml
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Optional
import warnings
import traceback 
from XAI_methods.explainer_method_api import ExplainerMethodAPI


class DiceExplainer(ExplainerMethodAPI):
    """
    DiCE implementation conforming to the ExplainerMethodAPI for generating
    counterfactual explanations, adapted for time series sequences treated
    as tabular data.

    Uses dice_ml.Dice to generate counterfactuals. Assumes the input model
    wrapper handles the sequence-to-tabular transformation internally if needed
    by the underlying ML model. DiCE itself operates on tabular data.
    """

    def __init__(self, model: Any, background_data: np.ndarray, **params: Any):
        """
        Initializes the DiceExplainer.

        Args:
            model (Any): The trained machine learning model wrapper instance (e.g., ModelWrapperForXAI).
                         Must have `.predict()` and `.predict_proba()` methods.
            background_data (np.ndarray): A representative sample of the input data
                already preprocessed/windowed into the 3D NumPy format
                (n_samples, sequence_length, n_features). Used for initializing DiCE's Data object.
            **params (Any): Additional parameters required for DiCE. Expected keys:
                - feature_names (List[str]): List of BASE feature names (before flattening). REQUIRED.
                - mode (str): 'classification' or 'regression'. REQUIRED.
                - background_outcomes (np.ndarray): 1D NumPy array of outcome labels/values
                    corresponding row-wise to the `background_data`. REQUIRED.
                - outcome_name (str): Name of the outcome/target variable. REQUIRED.
                - continuous_feature_names (List[str]): List of BASE feature names that are continuous. REQUIRED.
                - dice_method (str, optional): The DiCE method ('random', 'kdtree', 'genetic'). Defaults to 'random'.
                - backend (str, optional): Backend for DiCE model ('sklearn', 'TF1', 'TF2', 'PYT'). Defaults to 'sklearn'.
                # Other potential dice_ml init args can be passed.
        """
        # print("Initializing DiceExplainer...")
        self.model_wrapper = model # The wrapper instance (e.g., ModelWrapperForXAI)
        self.mode = params.get('mode', None) # Should be 'classification' or 'regression'
        self.feature_names = params.get('feature_names', None) # Base feature names
        self.background_outcomes = params.get('background_outcomes', None)
        self.outcome_name = params.get('outcome_name', None)
        self.continuous_feature_names_base = params.get('continuous_feature_names', None)
        self.dice_method = params.get('dice_method', 'random') # Default DiCE method
        self.backend = params.get('backend', 'sklearn') # Default backend

        # --- Validation ---
        if self.mode not in ['regression', 'classification']:
            raise ValueError("DiceExplainer requires 'mode' ('regression' or 'classification') in params.")
        if self.feature_names is None or not isinstance(self.feature_names, list):
            raise ValueError("DiceExplainer requires 'feature_names' (list of base names) in params.")
        if self.background_outcomes is None or not isinstance(self.background_outcomes, np.ndarray):
            raise ValueError("DiceExplainer requires 'background_outcomes' (1D NumPy array) in params.")
        if self.outcome_name is None or not isinstance(self.outcome_name, str):
            raise ValueError("DiceExplainer requires 'outcome_name' (string) in params.")
        if self.continuous_feature_names_base is None or not isinstance(self.continuous_feature_names_base, list):
             raise ValueError("DiceExplainer requires 'continuous_feature_names' (list of base names) in params.")
        if not isinstance(background_data, np.ndarray) or background_data.ndim != 3:
             raise ValueError("DiceExplainer requires background_data as a 3D NumPy array (samples, seq_len, features).")
        if background_data.shape[0] != len(self.background_outcomes):
             raise ValueError(f"Number of samples in background_data ({background_data.shape[0]}) does not match length of background_outcomes ({len(self.background_outcomes)}).")
        if background_data.shape[2] != len(self.feature_names):
             raise ValueError(f"Number of features in background_data ({background_data.shape[2]}) does not match length of feature_names ({len(self.feature_names)}).")

        # --- Store sequence properties and create flattened names ---
        self._original_sequence_shape = background_data.shape[1:] # (seq_len, n_features)
        self.sequence_length = self._original_sequence_shape[0]
        self._num_flat_features = np.prod(self._original_sequence_shape)

        # Create flattened feature names
        self.flat_feature_names = []
        for i in range(self.sequence_length): # Iterate through time steps (0 to seq_len-1)
            time_suffix = f"_t-{self.sequence_length - 1 - i}" # Suffix like _t-9, _t-8 ... _t-0
            for feat_name in self.feature_names:
                self.flat_feature_names.append(feat_name + time_suffix)

        if len(self.flat_feature_names) != self._num_flat_features:
            raise RuntimeError(f"Internal Error: Generated {len(self.flat_feature_names)} flat feature names, but expected {self._num_flat_features}.")

        # Create list of continuous features using the flattened names
        self.continuous_flat_feature_names = []
        for flat_name in self.flat_feature_names:
            # Extract base name robustly
            parts = flat_name.split('_t-')
            if len(parts) == 2:
                 base_name = parts[0]
                 if base_name in self.continuous_feature_names_base:
                     self.continuous_flat_feature_names.append(flat_name)
            else:
                 if flat_name in self.continuous_feature_names_base:
                     self.continuous_flat_feature_names.append(flat_name)
                     warnings.warn(f"Flattened feature name '{flat_name}' matched a base continuous name directly. Ensure naming convention is consistent.", UserWarning)

        # --- Prepare Data for DiCE (needs 2D DataFrame) ---
        # print("Preparing data for DiCE...")
        n_bg_samples = background_data.shape[0]
        background_data_flat = background_data.reshape(n_bg_samples, self._num_flat_features)
        background_df_dict = {name: background_data_flat[:, i] for i, name in enumerate(self.flat_feature_names)}
        background_df_dict[self.outcome_name] = self.background_outcomes
        background_df = pd.DataFrame(background_df_dict)

        if self.mode == 'classification':
             try: background_df[self.outcome_name] = background_df[self.outcome_name].astype(int)
             except ValueError as e: warnings.warn(f"Could not convert outcome column '{self.outcome_name}' to int for DiCE classification. Check data. Error: {e}", RuntimeWarning)
        else: # Regression
             try: background_df[self.outcome_name] = background_df[self.outcome_name].astype(float)
             except ValueError as e: warnings.warn(f"Could not convert outcome column '{self.outcome_name}' to float for DiCE regression. Check data. Error: {e}", RuntimeWarning)

        # print(f"Background DataFrame created. Shape: {background_df.shape}")
        # print(f"Outcome column: '{self.outcome_name}', Continuous flat features identified: {len(self.continuous_flat_feature_names)}")

        # --- Initialize dice_ml.Data ---
        try:
            valid_continuous_features = [f for f in self.continuous_flat_feature_names if f in background_df.columns]
            if len(valid_continuous_features) != len(self.continuous_flat_feature_names):
                missing = set(self.continuous_flat_feature_names) - set(valid_continuous_features)
                warnings.warn(f"Some derived continuous features are not in the background DataFrame columns: {missing}. Using only valid ones.", RuntimeWarning)
                self.continuous_flat_feature_names = valid_continuous_features # Update self attribute

            # Store the initialized Data object
            self._dice_data_interface = dice_ml.Data(
                dataframe=background_df,
                continuous_features=self.continuous_flat_feature_names, # Use potentially updated list
                outcome_name=self.outcome_name
            )
            # print("dice_ml.Data initialized.")
        except Exception as e:
            # print(f"Error initializing dice_ml.Data: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to initialize DiCE Data object.") from e

        # --- Initialize dice_ml.Model ---
        if self.mode == 'classification': dice_model_type = 'classifier'
        elif self.mode == 'regression': dice_model_type = 'regressor'
        else: raise ValueError(f"Invalid mode '{self.mode}' encountered during DiCE model initialization.")

        # Define the prediction function wrapper FOR DICE
        def _predict_fn_dice(input_df: pd.DataFrame, **params) -> np.ndarray:
             num_samples = len(input_df)
             missing_cols = set(self.flat_feature_names) - set(input_df.columns)
             if missing_cols: raise ValueError(f"DiCE Prediction Error: Input DataFrame is missing: {missing_cols}")

             features_only_df = input_df[self.flat_feature_names]
             input_np_flat = features_only_df.to_numpy()
             try:
                input_np_3d = input_np_flat.reshape((num_samples,) + self._original_sequence_shape)
             except ValueError as e:
                raise ValueError(f"DiCE Reshape Error: Cannot reshape ({input_np_flat.shape}) to ({(num_samples,) + self._original_sequence_shape}). Error: {e}") from e

             if self.mode == 'classifier':
                 if not hasattr(self.model_wrapper, 'predict_proba'): raise AttributeError("DiCE Classification Error: Model wrapper must have 'predict_proba'.")
                 predictions = self.model_wrapper.predict_proba(input_np_3d)
                 if predictions.ndim != 2 or predictions.shape[0] != num_samples: raise ValueError(f"Wrapper predict_proba returned shape {predictions.shape}. Expected ({num_samples}, n_classes).")
                 return predictions
             else: # Regression
                 if not hasattr(self.model_wrapper, 'predict'): raise AttributeError("DiCE Regression Error: Model wrapper must have 'predict'.")
                 predictions = self.model_wrapper.predict(input_np_3d)
                 if predictions.ndim == 2 and predictions.shape[1] == 1: predictions = predictions.flatten()
                 if predictions.ndim != 1 or predictions.shape[0] != num_samples: raise ValueError(f"Wrapper predict returned shape {predictions.shape}. Expected ({num_samples},) or ({num_samples}, 1).")
                 return predictions

        try:
            # Store the initialized Model object
            self._dice_model_interface = dice_ml.Model(
                model=self.model_wrapper, 
                backend=self.backend, 
                model_type=dice_model_type
            )
            # print(f"dice_ml.Model initialized directly with model wrapper and backend '{self.backend}'. Model type: '{dice_model_type}'")
        except Exception as e:
            # print(f"Warning: Initializing dice_ml.Model directly failed ({e}). Retrying with explicit function wrapper...")
            try:
                self._dice_model_interface = dice_ml.Model(
                    func=_predict_fn_dice, backend=self.backend, model_type=dice_model_type
                )
                # print(f"dice_ml.Model initialized with explicit function wrapper and backend '{self.backend}'. Model type: '{dice_model_type}'")
            except Exception as e2:
                # print(f"Error initializing dice_ml.Model with function wrapper: {e2}")
                traceback.print_exc()
                raise RuntimeError("Failed to initialize DiCE Model object.") from e2

        # --- Initialize dice_ml.Dice explainer ---
        try:
            self._explainer = dice_ml.Dice(
                data_interface=self._dice_data_interface,
                model_interface=self._dice_model_interface,
                method=self.dice_method
            )
            # print(f"dice_ml.Dice explainer initialized with method '{self.dice_method}'.")
            # print("DiceExplainer initialization complete.")
        except Exception as e:
            # print(f"Error initializing dice_ml.Dice: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to initialize DiCE explainer.") from e

    def explain(self,
                instances_to_explain: np.ndarray,
                **kwargs: Any) -> dice_ml.counterfactual_explanations.CounterfactualExplanations:
        """
        Generates counterfactual explanations for the given instances using DiCE.

        Args:
            instances_to_explain (np.ndarray): A NumPy array containing one or more
                instance sequences to explain, with shape (n_instances, sequence_length, n_features).
            **kwargs (Any): Keyword arguments passed directly to the
                            `dice_explainer.generate_counterfactuals` method. Common arguments:
                            - total_CFs (int): Number of counterfactuals to generate per instance. REQUIRED.
                            - desired_class (int or str, optional): For classification, the target class
                              for counterfactuals ('opposite' or specific class index). Defaults to 'opposite'.
                            - desired_range (list, optional): For regression, the target range [min, max].
                            - features_to_vary (list, optional): List of BASE feature names allowed to change.
                              If provided, these will be converted to their flattened equivalents.
                              Defaults to all continuous features identified during init (already flattened).
                            - permitted_range (dict, optional): Dictionary specifying value ranges for features
                              (should use FLATTENED feature names).
                            # Other DiCE generate_counterfactuals args...

        Returns:
            dice_ml.counterfactual_explanations.CounterfactualExplanations: An object containing
                the generated counterfactuals and related information.

        Raises:
            ValueError: If input shape is incorrect, required `total_CFs` kwarg is missing,
                        or other DiCE requirements are not met.
            TypeError: If input is not a NumPy array.
            Any exceptions from the underlying DiCE `generate_counterfactuals` call.
        """
        # print(f"DiceExplainer: Received {instances_to_explain.shape[0]} instances to explain.")

        # --- Input Validation ---
        if not isinstance(instances_to_explain, np.ndarray):
             raise TypeError(f"{type(self).__name__} expects instances_to_explain as NumPy ndarray.")
        if instances_to_explain.ndim != 3:
            raise ValueError(f"DiCE explain expects input with 3 dimensions (n_instances, sequence_length, features), got {instances_to_explain.ndim}D.")
        if instances_to_explain.shape[1:] != self._original_sequence_shape:
            raise ValueError(f"Instance sequence shape {instances_to_explain.shape[1:]} does not match background data sequence shape {self._original_sequence_shape}.")

        # --- Prepare Instances for DiCE (needs 2D DataFrame) ---
        n_instances = instances_to_explain.shape[0]
        instances_flat_np = instances_to_explain.reshape(n_instances, self._num_flat_features)
        # Ensure columns are in the same order as self.flat_feature_names used in init
        query_df = pd.DataFrame(instances_flat_np, columns=self.flat_feature_names)
        # print(f"Query DataFrame created for DiCE. Shape: {query_df.shape}")

        # --- Extract DiCE runtime arguments ---
        if 'total_CFs' not in kwargs:
            raise ValueError("DiCE 'explain' requires 'total_CFs' (number of counterfactuals) in kwargs.")

        total_CFs = kwargs.get('total_CFs')

        features_to_vary_input = kwargs.get('features_to_vary', None) # Get input list (might be None or list of base names)
        final_features_to_vary = []

        if features_to_vary_input is None or len(features_to_vary_input) == 0:
            # If no specific features provided, default to all continuous flattened features
            final_features_to_vary = self.continuous_flat_feature_names
            # print(f"No 'features_to_vary' provided, defaulting to {len(final_features_to_vary)} continuous flattened features.")
        else:
            # Assume features_to_vary_input contains BASE feature names and convert them
            # print(f"Received 'features_to_vary' input (base names): {features_to_vary_input}")
            base_names_to_vary = set(features_to_vary_input)
            converted_count = 0
            for flat_name in self.flat_feature_names: # Iterate through all possible flat names
                parts = flat_name.split('_t-')
                base_name = parts[0] if len(parts) == 2 else flat_name # Fallback if no suffix

                if base_name in base_names_to_vary:
                    # Check if this specific flattened name is actually a column in the query_df
                    if flat_name in query_df.columns:
                         final_features_to_vary.append(flat_name)
                         converted_count += 1
                    else:
                         warnings.warn(f"Base feature '{base_name}' requested in features_to_vary, but corresponding flattened feature '{flat_name}' not found in query data columns. Skipping.", RuntimeWarning)

            # print(f"Converted base features to {len(final_features_to_vary)} valid flattened features to vary.")
            # Validate if any features remain after conversion
            if not final_features_to_vary:
                 warnings.warn("After converting base feature names in 'features_to_vary', the list is empty. Defaulting to all continuous flattened features.", RuntimeWarning)
                 final_features_to_vary = self.continuous_flat_feature_names # Fallback to default

        # Build the final arguments for generate_counterfactuals
        dice_runtime_args = {
             'total_CFs': total_CFs,
             'features_to_vary': final_features_to_vary, # Use the processed list
             'permitted_range': kwargs.get('permitted_range', None) # Pass if provided (should use flattened names)
        }



        if self.mode == 'classifier':
            dice_runtime_args['desired_class'] = kwargs.get('desired_class', 'opposite')
        else: # regression
            desired_range = kwargs.get('desired_range', None)
            if desired_range is None: warnings.warn("DiceExplainer Regression: 'desired_range' not provided. DiCE might error.", UserWarning)
            else: dice_runtime_args['desired_range'] = desired_range

        # --- Call DiCE generate_counterfactuals ---
        # print(f"Calling DiCE generate_counterfactuals with args: {dice_runtime_args}")
        try:
            if not hasattr(self, '_explainer') or self._explainer is None:
                 raise RuntimeError("DiCE explainer object was not initialized correctly.")

            # Check again if features_to_vary is empty before calling DiCE
            if not dice_runtime_args['features_to_vary']:
                 raise ValueError("Cannot generate counterfactuals because the final 'features_to_vary' list is empty. Check input and continuous feature definitions.")

            dice_exp = self._explainer.generate_counterfactuals(
                query_instances=query_df,
                **dice_runtime_args
            )
            # print("DiCE explanation finished.")
            return dice_exp
        except Exception as e:
            # print(f"Error during DiCE generate_counterfactuals calculation: {e}")
            traceback.print_exc()
            raise RuntimeError("DiCE explanation failed.") from e
