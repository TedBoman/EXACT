import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV # k-fold cross-validation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, get_scorer # Added for evaluation
from ML_models import model_interface 
from typing import Dict, List, Optional, Tuple, Union
import warnings
import traceback # Added for better error printing

class DecisionTreeModel(model_interface.ModelInterface): 
    """
    Supervised Decision Tree classification model for anomaly detection using labeled data,
    with k-fold cross-validation for performance estimation.

    Handles input as:
    1.  Pandas DataFrame: Converts features directly to a 2D NumPy array. NO LAGGING.
    2.  3D NumPy array (X) and 1D NumPy array (y): Assumes X has shape
        (samples, sequence_length, features). Flattens the last two dimensions.

    Includes internal scaling (MinMaxScaler) and imputation (SimpleImputer).
    Trains a DecisionTreeClassifier. Performance is estimated using k-fold cross-validation.
    A final model is then trained on the entire dataset.
    Handles class imbalance using the 'class_weight' parameter.
    """
    
    def __init__(self, **kwargs):
        """Initializes the Decision Tree classifier model.

        Args:
            criterion (str): Function to measure the quality of a split (default: 'gini').
            max_depth (int, optional): Maximum depth of the tree (default: None).
            min_samples_split (int): Minimum samples to split node (default: 2).
            min_samples_leaf (int): Minimum samples at leaf node (default: 1).
            random_state (int): Controls randomness (default: 42).
            class_weight (dict, 'balanced', optional): Class weights (default: 'balanced').
            imputer_strategy (str): Strategy for SimpleImputer ('mean', 'median', etc., default: 'mean').
            n_splits (int): Number of folds for StratifiedKFold cross-validation (default: 5).
            shuffle_kfold (bool): Whether to shuffle data before k-fold splitting (default: True).
            validation_metrics (list): Metrics to compute during cross-validation
                                        (e.g., ['accuracy', 'f1', 'roc_auc'], default: ['accuracy', 'f1', 'roc_auc']).
            auto_tune (bool): Whether to perform hyperparameter tuning (default: False).
            search_n_iter (int): Number of parameter settings sampled for RandomizedSearchCV (default: 10).
            search_scoring (str): Scoring metric for RandomizedSearchCV (default: 'f1').
            param_dist (dict, optional): Parameter distribution for RandomizedSearchCV. 
                                         If None, a default is used for DecisionTreeClassifier.
            **kwargs: Additional parameters passed to DecisionTreeClassifier.
        """
        self.model: Optional[DecisionTreeClassifier] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.input_type: Optional[str] = None
        self.processed_feature_names: Optional[List[str]] = None
        self.original_feature_names_: Optional[List[str]] = None
        self.sequence_length: Optional[int] = None
        self.n_original_features: Optional[int] = None
        self.label_col: Optional[str] = None

        self.validation_scores_: Dict[str, float] = {} 
        self.n_splits = kwargs.pop('n_splits', 5) # Use pop to remove from kwargs
        self.shuffle_kfold = kwargs.pop('shuffle_kfold', True)
        if self.n_splits <= 1:
            raise ValueError("n_splits for k-fold cross-validation must be greater than 1.")
        self.validation_metrics = kwargs.pop('validation_metrics', ['accuracy', 'f1', 'roc_auc'])

        random_state = kwargs.get('random_state', 42)

        # Auto-tuning parameters
        self.auto_tune = kwargs.pop('auto_tune', False)
        self.search_n_iter = kwargs.pop('search_n_iter', 100)
        self.search_scoring = kwargs.pop('search_scoring', 'f1') # Primary metric for tuning
        self.tuned_best_params_: Optional[Dict] = None
        
        # Base model parameters (will be updated if auto_tune is True)
        self.model_params = {
            'criterion': kwargs.get('criterion', 'gini'),
            'max_depth': kwargs.get('max_depth', None),
            'min_samples_split': kwargs.get('min_samples_split', 2),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
            'class_weight': kwargs.get('class_weight', 'balanced'),
            'random_state': random_state,
            'max_features': kwargs.get('max_features', None)
        }
        if self.model_params.get('max_features') == 'None': # Handle string 'None' from some configs
            self.model_params['max_features'] = None

        self._imputer_strategy = kwargs.pop('imputer_strategy', 'mean')

        if self.auto_tune:
            default_param_dist_dt = {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 5, 10, 15, 20, 30, 50],
                'min_weight_fraction_leaf': [0.0001, 0.001, 0.01, 0.1, 1],
                'min_samples_split': [2, 5, 10, 20, 30],
                'min_samples_leaf': [1, 2, 5, 10, 15],
                'max_features': [None, 'sqrt', 'log2'],
                'ccp_alpha': [0.0, 0.001, 0.01, 0.1]
            }
            self.param_dist = kwargs.pop('param_dist', default_param_dist_dt)
            # # print(f"DecisionTreeModel: Auto-tuning ENABLED. Search iterations: {self.search_n_iter}, Main scoring: '{self.search_scoring}'")
            # # print(f"Parameter distribution for tuning: {self.param_dist}")
            if self.search_scoring not in self.validation_metrics:
                 original_search_scoring = self.search_scoring
                 # Attempt to map to a primary validation metric if possible (e.g., if user passes 'f1_macro' but 'f1' is in validation_metrics)
                 if self.search_scoring.startswith('f1') and 'f1' in self.validation_metrics:
                     self.search_scoring = 'f1'
                 elif self.search_scoring.startswith('roc_auc') and 'roc_auc' in self.validation_metrics:
                     self.search_scoring = 'roc_auc'
                 elif self.search_scoring.startswith('accuracy') and 'accuracy' in self.validation_metrics:
                     self.search_scoring = 'accuracy'

                 if self.search_scoring != original_search_scoring:
                     warnings.warn(f"search_scoring '{original_search_scoring}' mapped to '{self.search_scoring}' to align with validation_metrics for refit.", UserWarning)
                 
                 if self.search_scoring not in self.validation_metrics:
                     warnings.warn(f"Primary search_scoring='{original_search_scoring}' (or mapped '{self.search_scoring}') is not in validation_metrics={self.validation_metrics}. "
                                   f"This might cause issues if '{self.search_scoring}' is not a recognized scikit-learn scorer for refit. "
                                   f"Consider adding it to validation_metrics or using a standard scorer name.", UserWarning)

        # Collect any remaining kwargs for DecisionTreeClassifier
        allowed_dt_params = set(DecisionTreeClassifier().get_params().keys())
        # Params already handled or specific to this class wrapper
        # 'n_splits', 'shuffle_kfold', 'validation_metrics', 'imputer_strategy', 'auto_tune', 'search_n_iter', 'search_scoring', 'param_dist'
        # The keys in self.model_params are also "handled"
        
        extra_dt_params = {
            k: v for k, v in kwargs.items()
            if k in allowed_dt_params and k not in self.model_params
        }
        self.model_params.update(extra_dt_params)

        if self.model_params.get('class_weight') is None:
            warnings.warn("class_weight was None, setting to 'balanced' by default.", UserWarning)
            self.model_params['class_weight'] = 'balanced'

        # print(f"DecisionTreeModel Initialized with base params: {self.model_params}")
        # print(f"Imputer Strategy: {self._imputer_strategy}")
        # print(f"K-fold CV / Search CV: n_splits={self.n_splits}, shuffle={self.shuffle_kfold}")
        # print(f"Validation metrics to report: {self.validation_metrics}")

    def _prepare_data_for_model(
        self, X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        label_col: Optional[str] = None, 
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Internal helper to preprocess data: reshape (if needed), scale, and impute NaNs.
        Sets internal attributes like input_type, sequence_length, n_original_features,
        processed_feature_names during training.
        Returns the final processed 2D NumPy array for the model, aligned labels (if training),
        and the list of processed feature names.
        """
        
        X_features_np = None # Holds 2D features before scaling/imputation
        y_aligned = None
        current_feature_names = None # Will hold the names corresponding to X_features_np columns
        
        # --- Stage 1: Determine input type, reshape (if needed), get initial features/labels ---
        if isinstance(X, pd.DataFrame):
            if is_training:
                if self.input_type is None: self.input_type = 'dataframe'
                elif self.input_type != 'dataframe': raise RuntimeError("Model already trained with different input type.")
                self.sequence_length = 1 

                if label_col is None or label_col not in X.columns:
                    raise ValueError(f"Label column '{label_col}' not found for DataFrame training.")
                self.label_col = label_col
                # Store original feature names
                original_names = X.columns.drop(label_col).tolist()
                if not original_names: raise ValueError("No feature columns found in DataFrame.")
                self.original_feature_names_ = original_names
                
                if self.n_original_features is None: self.n_original_features = len(original_names)
                elif self.n_original_features != len(original_names): raise ValueError("Feature count mismatch during training.")
                
                # For DataFrame input, processed names are the same as original names
                if self.processed_feature_names is None: self.processed_feature_names = original_names
                elif self.processed_feature_names != original_names: raise ValueError("Feature names mismatch during training.")

                current_feature_names = self.processed_feature_names
                X_features_np = X[current_feature_names].to_numpy()
                y_aligned = X[self.label_col].to_numpy()

                if X_features_np.shape[0] == 0: raise ValueError("No data rows found in DataFrame features.")

            else: # Detection/Scoring for DataFrame
                if self.input_type != 'dataframe' or self.scaler is None or self.imputer is None or self.original_feature_names_ is None or self.n_original_features is None:
                    raise RuntimeError("Model was not trained on DataFrame or required components (scaler/imputer/names) are missing.")
                
                missing_cols = set(self.original_feature_names_) - set(X.columns)
                if missing_cols: raise ValueError(f"Detection DataFrame missing required columns: {missing_cols}")

                current_feature_names = self.original_feature_names_ # Use original names for DF detection
                X_features_np = X[current_feature_names].to_numpy()

                if X_features_np.shape[0] == 0:
                    warnings.warn("No data rows provided for DataFrame detection/scoring.", RuntimeWarning)
                    # Return empty array matching expected 2D feature count
                    return np.empty((0, self.n_original_features)), None, current_feature_names

        elif isinstance(X, np.ndarray):
            if X.ndim != 3: raise ValueError(f"NumPy array input must be 3D (samples, seq_len, features), got {X.ndim}D.")
            n_samples, seq_len, n_feat = X.shape

            if is_training:
                if self.input_type is None: self.input_type = 'numpy'
                elif self.input_type != 'numpy': raise RuntimeError("Model already trained with different input type.")
                
                if self.sequence_length is None: self.sequence_length = seq_len
                elif self.sequence_length != seq_len: raise ValueError("Sequence length mismatch during training.")
                
                if self.n_original_features is None: self.n_original_features = n_feat
                elif self.n_original_features != n_feat: raise ValueError("Feature count mismatch during training.")

                # Assume original feature names are generic if not provided via run()
                if self.original_feature_names_ is None:
                    self.original_feature_names_ = [f"orig_feat_{i}" for i in range(self.n_original_features)]
                    warnings.warn("Original feature names not provided for NumPy input, generating generic names.", UserWarning)
                elif len(self.original_feature_names_) != self.n_original_features:
                     raise ValueError(f"Provided original feature names count ({len(self.original_feature_names_)}) != input features ({self.n_original_features}).")


                if y is None: raise ValueError("Labels 'y' are required for NumPy array training.")
                if not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != n_samples:
                    raise ValueError("Invalid 'y' array provided for NumPy training.")
                y_aligned = y

                n_flattened_features = seq_len * n_feat
                X_features_np = X.reshape(n_samples, n_flattened_features) # Flatten to 2D
                
                # Generate flattened feature names
                if self.processed_feature_names is None:
                    self.processed_feature_names = [f"{orig_name}_step_{j}" 
                                                    for j in range(seq_len) 
                                                    for orig_name in self.original_feature_names_]
                    if len(self.processed_feature_names) != n_flattened_features: # Fallback naming
                        self.processed_feature_names = [f"flat_feature_{k}" for k in range(n_flattened_features)]
                elif len(self.processed_feature_names) != n_flattened_features:
                    raise ValueError("Flattened feature name count mismatch during training.")
                current_feature_names = self.processed_feature_names

            else: # Detection/Scoring for NumPy
                if self.input_type != 'numpy' or self.scaler is None or self.imputer is None or self.processed_feature_names is None or self.original_feature_names_ is None or self.n_original_features is None or self.sequence_length is None:
                    raise RuntimeError("Model was not trained on NumPy or required components (scaler/imputer/names/dims) are missing.")

                if seq_len != self.sequence_length: raise ValueError(f"Input sequence length {seq_len} != train sequence length {self.sequence_length}.")
                if n_feat != self.n_original_features: raise ValueError(f"Input feature count {n_feat} != train feature count {self.n_original_features}.")

                current_feature_names = self.processed_feature_names # Use flattened names
                n_flattened_features = len(current_feature_names)
                if n_samples == 0:
                    warnings.warn("No samples provided for NumPy detection/scoring.", RuntimeWarning)
                    return np.empty((0, n_flattened_features)), None, current_feature_names

                X_features_np = X.reshape(n_samples, n_flattened_features) # Flatten to 2D
        else:
            raise TypeError("Input 'X' must be a pandas DataFrame or a 3D NumPy array.")

        # --- Stage 2: Scaling ---
        if is_training:
            self.scaler = MinMaxScaler()
            X_processed_scaled = self.scaler.fit_transform(X_features_np)
        else:
            if self.scaler is None or not hasattr(self.scaler, 'scale_'): 
                 raise RuntimeError("Scaler not fitted. Call run() first.")
            X_processed_scaled = self.scaler.transform(X_features_np)
        
        # --- Stage 3: Imputation ---
        if is_training:
            self.imputer = SimpleImputer(strategy=self._imputer_strategy)
            X_processed_imputed = self.imputer.fit_transform(X_processed_scaled)
        else:
            if self.imputer is None or not hasattr(self.imputer, 'statistics_'): 
                 raise RuntimeError("Imputer not fitted. Call run() first.")
            X_processed_imputed = self.imputer.transform(X_processed_scaled)
            
        # Check for remaining NaNs after imputation (should ideally be zero)
        if np.isnan(X_processed_imputed).any():
            warnings.warn("NaN values detected *after* imputation. Check input data or imputer strategy. Applying nan_to_num as fallback.", RuntimeWarning)
            # Apply nan_to_num as a final fallback
            X_processed_imputed = np.nan_to_num(X_processed_imputed, nan=0.0) 

        # Ensure current_feature_names is set
        if current_feature_names is None:
             raise RuntimeError("Internal Error: current_feature_names not set.")
             
        return X_processed_imputed, y_aligned, current_feature_names


    def run(self, X: Union[pd.DataFrame, np.ndarray],
            y: Optional[np.ndarray] = None,
            label_col: str = 'label',
            original_feature_names: Optional[List[str]] = None):
        """
        Prepares data. If auto_tune is True, performs RandomizedSearchCV.
        Otherwise, performs k-fold cross-validation for performance estimation.
        Then, trains a final Decision Tree classifier on the entire dataset using
        the best (if tuned) or initial parameters.
        """
        # print(f"Running training for DecisionTreeModel (Input type: {'DataFrame' if isinstance(X, pd.DataFrame) else 'NumPy'})...")
        # if self.auto_tune:
        #     # print(f"Hyperparameter auto-tuning is ENABLED.")
        # else:
        #     # print(f"Hyperparameter auto-tuning is DISABLED. Using fixed parameters: {self.model_params}")

        if isinstance(X, np.ndarray):
            if original_feature_names is None:
                raise ValueError("`original_feature_names` must be provided when training with NumPy input `X`.")
            if X.shape[2] != len(original_feature_names):
                 raise ValueError(f"NumPy input feature dimension ({X.shape[2]}) doesn't match length of original_feature_names ({len(original_feature_names)}).")
            self.original_feature_names_ = original_feature_names
        elif original_feature_names is not None:
             warnings.warn("`original_feature_names` provided but input `X` is DataFrame. Names will be inferred from DataFrame columns.", UserWarning)

        # --- Step 1: Prepare FULL data (fits scaler/imputer globally) ---
        X_processed_full, y_aligned_full, _ = self._prepare_data_for_model(
            X, y=y, label_col=label_col, is_training=True
        )

        if y_aligned_full is None:
            raise RuntimeError("No labels available for training after preprocessing.")
        if X_processed_full.shape[0] != len(y_aligned_full):
            raise RuntimeError(f"Mismatch between processed data samples ({X_processed_full.shape[0]}) and labels ({len(y_aligned_full)}).")
        if X_processed_full.shape[0] < self.n_splits:
             warnings.warn(f"Number of samples ({X_processed_full.shape[0]}) is less than n_splits ({self.n_splits}). K-fold CV/Search might behave unexpectedly or fail. Consider reducing n_splits or increasing data.", RuntimeWarning)

        current_model_params = self.model_params.copy() # Start with base params

        if self.auto_tune:
            # --- Step 2a: Hyperparameter Tuning with RandomizedSearchCV ---
            # print(f"\nStarting RandomizedSearchCV for hyperparameter tuning...")
            # print(f"  n_iter={self.search_n_iter}, cv_folds={self.n_splits}, refit_scoring='{self.search_scoring}'")

            # Base estimator for tuning - use only non-tunable params like random_state, class_weight
            # Tunable params will come from param_dist.
            # Ensure random_state and class_weight are part of the initial setup if not tuned.
            base_estimator_params = {
                'random_state': current_model_params.get('random_state'),
                'class_weight': current_model_params.get('class_weight')
            }
             # Filter out params that are in param_dist from base_estimator_params, as they'll be set by search
            for key_to_tune in self.param_dist.keys():
                if key_to_tune in base_estimator_params:
                    del base_estimator_params[key_to_tune]
            
            base_estimator_params = {k: v for k, v in base_estimator_params.items() if v is not None}
            temp_model_for_search = DecisionTreeClassifier(**base_estimator_params)

            # Prepare scoring for RandomizedSearchCV
            # Scikit-learn's standard scoring names:
            sklearn_scoring_map = {
                'accuracy': 'accuracy',
                'f1': 'f1_macro',  # Using f1_macro for robustness, can be configured
                'roc_auc': 'roc_auc'
                # Add other mappings if your validation_metrics use custom names for standard scorers
            }
            
            # Create the scoring dictionary for RandomizedSearchCV
            # This allows calculating all desired validation_metrics during the search's CV
            scoring_dict_for_search = {}
            for metric_name in self.validation_metrics:
                if metric_name in sklearn_scoring_map:
                    scoring_dict_for_search[metric_name] = sklearn_scoring_map[metric_name]
                elif metric_name in ['f1_weighted', 'f1_micro', 'roc_auc_ovr', 'roc_auc_ovo']: # Directly usable sklearn scorers
                    scoring_dict_for_search[metric_name] = metric_name
                else:
                    warnings.warn(f"Metric '{metric_name}' from validation_metrics is not directly mapped to a standard scikit-learn scorer or a known alias. "
                                  f"It will be SKIPPED for RandomizedSearchCV's multi-metric evaluation. "
                                  f"Ensure it's a valid scorer string if you intend to use it.", UserWarning)
            
            if not scoring_dict_for_search:
                raise ValueError(f"No scikit-learn compatible metrics could be derived from validation_metrics ({self.validation_metrics}) for RandomizedSearchCV. "
                                 f"Must include at least one of 'accuracy', 'f1', 'roc_auc' or other valid sklearn scorer strings.")

            # Determine the 'refit_metric_key' for RandomizedSearchCV.
            # This key must be present in `scoring_dict_for_search`.
            refit_metric_key = None

            # Priority 1: If self.search_scoring (e.g., 'f1') is a direct key in scoring_dict_for_search.
            # This is the most common and intended case, especially after __init__'s mapping.
            if self.search_scoring in scoring_dict_for_search:
                refit_metric_key = self.search_scoring
            else:
                # Priority 2: If self.search_scoring is a scikit-learn scorer name (e.g., 'f1_macro')
                # that is a VALUE in scoring_dict_for_search, find the corresponding KEY.
                found_key_for_value = False
                for key_in_dict, value_in_dict in scoring_dict_for_search.items():
                    if value_in_dict == self.search_scoring:
                        refit_metric_key = key_in_dict
                        warnings.warn(
                            f"User's 'search_scoring' ('{self.search_scoring}') is a scikit-learn scorer name. "
                            f"Mapped to key '{refit_metric_key}' for refit, based on your validation_metrics.", UserWarning
                        )
                        found_key_for_value = True
                        break
                
                # Priority 3: Fallback if no match found yet (e.g., self.search_scoring was something entirely different)
                if not found_key_for_value: # and refit_metric_key is still None
                    if scoring_dict_for_search: # Ensure the dictionary is not empty
                        first_key = list(scoring_dict_for_search.keys())[0]
                        warnings.warn(
                            f"Specified 'search_scoring' ('{self.search_scoring}') could not be directly used or mapped as a refit metric key "
                            f"from available scoring metrics ({list(scoring_dict_for_search.keys())}). "
                            f"Defaulting refit to the first available metric key: '{first_key}'.", UserWarning
                        )
                        refit_metric_key = first_key
                    else:
                        # This should have been caught by the earlier check that scoring_dict_for_search is not empty.
                        raise ValueError("Cannot determine refit metric: scoring_dict_for_search is empty and no valid refit key could be derived.")

            # Final check: refit_metric_key must be a valid key from the scoring dictionary
            if refit_metric_key not in scoring_dict_for_search:
                 # This would indicate a logic error or an unhandled edge case if scoring_dict_for_search was valid.
                 raise ValueError(
                    f"Internal error or configuration issue: Determined refit_metric_key '{refit_metric_key}' is not a valid key in "
                    f"scoring_dict_for_search (keys: {list(scoring_dict_for_search.keys())}). "
                    f"Original user 'search_scoring': '{self.search_scoring}'."
                 )

            # print(f"  RandomizedSearchCV will use metrics for scoring: {scoring_dict_for_search}")
            # print(f"  It will refit the best model based on the metric KEY: '{refit_metric_key}' "
                #   f"(which corresponds to scikit-learn scorer: '{scoring_dict_for_search[refit_metric_key]}')")

            search_cv_strategy = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle_kfold,
                                        random_state=current_model_params.get('random_state'))

            random_search = RandomizedSearchCV(
                estimator=temp_model_for_search,
                param_distributions=self.param_dist,
                n_iter=self.search_n_iter,
                scoring=scoring_dict_for_search,
                refit=refit_metric_key,
                cv=search_cv_strategy,
                random_state=current_model_params.get('random_state'),
                n_jobs=-1, 
                error_score=np.nan 
            )

            try:
                random_search.fit(X_processed_full, y_aligned_full)
                self.tuned_best_params_ = random_search.best_params_
                # print(f"  Best parameters found by RandomizedSearchCV: {self.tuned_best_params_}")
                
                # Update current_model_params with the best ones found for the final model
                current_model_params.update(self.tuned_best_params_)
                # print(f"  Model parameters for final training updated to: {current_model_params}")

                # Populate validation_scores_ from cv_results_ of the best estimator
                self.validation_scores_ = {}
                # print("\nCross-validation summary from RandomizedSearchCV (for best parameter set):")
                results_df = pd.DataFrame(random_search.cv_results_)
                best_idx = random_search.best_index_

                for metric_key_user in scoring_dict_for_search.keys(): # User's metric name (e.g., 'f1')
                    # The actual column name in cv_results_ will be 'mean_test_{scorer_name}'
                    scorer_name_sklearn = scoring_dict_for_search[metric_key_user]
                    mean_score_col = f"mean_test_{scorer_name_sklearn}" 
                    std_score_col = f"std_test_{scorer_name_sklearn}"
                    
                    if mean_score_col in results_df.columns:
                        avg_score = results_df.loc[best_idx, mean_score_col]
                        std_score = results_df.loc[best_idx, std_score_col] if std_score_col in results_df.columns else np.nan
                        self.validation_scores_[metric_key_user] = avg_score # Store with user's original metric name
                        # print(f"  Average {metric_key_user} (as {scorer_name_sklearn}): {avg_score:.4f} (Std: {std_score:.4f})")
                    else:
                        self.validation_scores_[metric_key_user] = np.nan
                        # This case should be rare if scoring_dict_for_search was constructed correctly
                        # print(f"  Average {metric_key_user}: Score not found in cv_results_ (expected col like {mean_score_col}).")
                
                if np.isnan(random_search.best_score_):
                    warnings.warn(f"RandomizedSearchCV best_score_ is NaN for refit metric '{refit_metric_key}'. "
                                  "This might indicate all parameter combinations failed or resulted in NaN scores for this metric during CV.", RuntimeWarning)

            except Exception as e:
                warnings.warn(f"RandomizedSearchCV failed: {e}. Proceeding with base model_params: {self.model_params}. Validation scores will be from fallback if any.", RuntimeWarning)
                traceback.print_exc()
                self.validation_scores_ = {metric: np.nan for metric in self.validation_metrics}
                # current_model_params remain the initial self.model_params if search fails
                current_model_params = self.model_params.copy() 
                self.tuned_best_params_ = None
        else:
            # --- Step 2b: K-Fold Cross-Validation for Performance Estimation (if not auto-tuning) ---
            # This part is largely the same as your original k-fold loop.
            # It uses the initial self.model_params.
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle_kfold,
                                  random_state=current_model_params.get('random_state')) # Use current_model_params for consistency
            
            fold_scores: Dict[str, List[float]] = {metric: [] for metric in self.validation_metrics}
            fold_count = 0

            # print(f"\nStarting {self.n_splits}-fold cross-validation with fixed parameters: {current_model_params}")
            for fold_idx, (train_index, val_index) in enumerate(skf.split(X_processed_full, y_aligned_full)):
                fold_count += 1
                # print(f"  Fold {fold_idx + 1}/{self.n_splits}...")
                X_train_fold, X_val_fold = X_processed_full[train_index], X_processed_full[val_index]
                y_train_fold, y_val_fold = y_aligned_full[train_index], y_aligned_full[val_index]

                if X_train_fold.shape[0] == 0 or X_val_fold.shape[0] == 0:
                    warnings.warn(f"    Fold {fold_idx+1} has an empty train or validation set. Skipping this fold.", RuntimeWarning)
                    for metric_name in self.validation_metrics: fold_scores[metric_name].append(np.nan) # Append NaN
                    continue
                
                n_neg_train_fold = np.sum(y_train_fold == 0); n_pos_train_fold = np.sum(y_train_fold == 1)
                # # print(f"    Training fold {fold_idx+1} label composition: {n_neg_train_fold} negative, {n_pos_train_fold} positive.")
                if n_pos_train_fold == 0: warnings.warn(f"    No positive samples in training part of fold {fold_idx+1}.", RuntimeWarning)
                if n_neg_train_fold == 0: warnings.warn(f"    No negative samples in training part of fold {fold_idx+1}.", RuntimeWarning)

                temp_model = DecisionTreeClassifier(**current_model_params) # Use current_model_params
                try:
                    temp_model.fit(X_train_fold, y_train_fold)
                except Exception as e:
                    warnings.warn(f"    Decision Tree fitting failed for fold {fold_idx + 1}: {e}. Skipping fold.", RuntimeWarning)
                    for metric_name in self.validation_metrics: fold_scores[metric_name].append(np.nan)
                    continue

                try:
                    y_pred_val_fold = temp_model.predict(X_val_fold)
                    y_proba_val_fold = temp_model.predict_proba(X_val_fold)

                    for metric_name in self.validation_metrics:
                        score = np.nan
                        try:
                            if metric_name == 'accuracy':
                                score = accuracy_score(y_val_fold, y_pred_val_fold)
                            elif metric_name == 'f1': # This is the user's key, map to a specific f1 if needed
                                score = f1_score(y_val_fold, y_pred_val_fold, average='macro', zero_division=0) # Or 'binary' if appropriate
                            elif metric_name == 'roc_auc':
                                if len(np.unique(y_val_fold)) > 1 and hasattr(temp_model, 'classes_'):
                                    pos_class_idx = np.where(temp_model.classes_ == 1)[0]
                                    if len(pos_class_idx) > 0 and y_proba_val_fold.shape[1] > pos_class_idx[0]:
                                        score = roc_auc_score(y_val_fold, y_proba_val_fold[:, pos_class_idx[0]])
                                    # else: warnings.warn(f"    Positive class (1) not found or proba shape insufficient for ROC AUC in fold {fold_idx + 1}.", RuntimeWarning)
                                # else: warnings.warn(f"    ROC AUC undefined for fold {fold_idx + 1} (single class in y_val_fold or classes_ missing).", RuntimeWarning)
                            else: # Handle other sklearn compatible string metrics
                                try:
                                    scorer_fn = get_scorer(metric_name)
                                    score = scorer_fn(temp_model, X_val_fold, y_val_fold)
                                except Exception:
                                    warnings.warn(f"    Unsupported/failed validation metric '{metric_name}' in fold {fold_idx + 1}. Skipping.", UserWarning)
                            
                            fold_scores[metric_name].append(score)
                            # if not np.isnan(score): print(f"    Fold {fold_idx + 1} {metric_name}: {score:.4f}")
                            # else: print(f"    Fold {fold_idx + 1} {metric_name}: NaN")

                        except Exception as metric_e:
                            # print(f"    Failed to calculate metric '{metric_name}' for fold {fold_idx + 1}: {metric_e}")
                            fold_scores[metric_name].append(np.nan)
                except Exception as eval_e:
                    # print(f"    Failed during validation evaluation for fold {fold_idx + 1}: {eval_e}")
                    for metric_name in self.validation_metrics: fold_scores[metric_name].append(np.nan)


            if fold_count == 0 and self.n_splits > 0:
                warnings.warn("K-fold cross-validation completed but no folds were successfully processed. Validation scores will be empty or NaN.", RuntimeWarning)
                self.validation_scores_ = {metric: np.nan for metric in self.validation_metrics}
            else:
                # print("\nCross-validation summary (fixed parameters):")
                for metric_name in self.validation_metrics:
                    valid_fold_metric_scores = [s for s in fold_scores[metric_name] if not np.isnan(s)]
                    if valid_fold_metric_scores:
                        avg_score = np.mean(valid_fold_metric_scores)
                        std_score = np.std(valid_fold_metric_scores)
                        self.validation_scores_[metric_name] = avg_score
                        # print(f"  Average {metric_name}: {avg_score:.4f} (Std: {std_score:.4f})")
                    else:
                        self.validation_scores_[metric_name] = np.nan
                        # print(f"  Average {metric_name}: NaN (no valid scores from folds)")
        # print("-" * 30)

        # --- Step 3: Training the FINAL Classifier on the ENTIRE Prepared Dataset ---
        # This step uses `current_model_params`, which are either initial or updated by RandomizedSearchCV.
        # print(f"Training final DecisionTreeClassifier on {X_processed_full.shape[0]} samples, {X_processed_full.shape[1]} features...")
        # print(f"Using final parameters: {current_model_params}")
        
        n_neg_full = np.sum(y_aligned_full == 0); n_pos_full = np.sum(y_aligned_full == 1)
        # print(f"Full dataset label composition for final training: {n_neg_full} negative, {n_pos_full} positive.")
        if n_pos_full == 0: warnings.warn(f"No positive samples ({self.label_col or 'label'}=1) found in the ENTIRE dataset for final model training.", RuntimeWarning)
        if n_neg_full == 0: warnings.warn(f"No negative samples ({self.label_col or 'label'}=0) found in the ENTIRE dataset for final model training.", RuntimeWarning)

        self.model = DecisionTreeClassifier(**current_model_params)
        try:
            self.model.fit(X_processed_full, y_aligned_full)
        except Exception as e:
            self.model = None 
            raise RuntimeError(f"Final Decision Tree fitting failed on the full dataset: {e}") from e

        # print("Final model training complete.")
        # if hasattr(self.model, 'classes_'): print(f"Final model trained with classes: {self.model.classes_}")
        
        # Store the effective parameters used for the final model
        self.model_params = current_model_params.copy()

    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # This method uses the final self.model trained on the full dataset.
        if self.model is None or self.scaler is None or self.imputer is None or self.input_type is None:
            raise RuntimeError("Model is not trained or key components are missing. Cannot score.")

        n_input_samples = len(detection_data) if isinstance(detection_data, pd.DataFrame) else detection_data.shape[0]
        final_scores = np.full(n_input_samples, np.nan, dtype=float)

        if n_input_samples == 0:
            return final_scores

        try:
            X_processed_imputed, _, _ = self._prepare_data_for_model(
                detection_data, is_training=False, label_col=self.label_col
            )

            if X_processed_imputed.shape[0] == 0:
                warnings.warn("No processable data found for scoring after preparation.", RuntimeWarning)
                return final_scores

            probabilities = self.model.predict_proba(X_processed_imputed)

            if hasattr(self.model, 'classes_'):
                positive_class_index = np.where(self.model.classes_ == 1)[0]
                if len(positive_class_index) > 0:
                    class_index_to_use = positive_class_index[0]
                    if probabilities.shape[1] > class_index_to_use:
                         anomaly_scores = probabilities[:, class_index_to_use]
                    else:
                         warnings.warn(f"Positive class index {class_index_to_use} is out of bounds. Returning NaN.", RuntimeWarning)
                         anomaly_scores = np.full(X_processed_imputed.shape[0], np.nan)
                else:
                    warnings.warn(f"Positive class (1) not found in model classes. Returning prob of first class.", RuntimeWarning)
                    class_index_to_use = 0
                    anomaly_scores = probabilities[:, class_index_to_use] if probabilities.shape[1] > 0 else np.zeros(X_processed_imputed.shape[0])
            else:
                warnings.warn("Model classes_ attribute not found. Assuming class 1 is second column.", RuntimeWarning)
                if probabilities.shape[1] < 2:
                    raise RuntimeError("Cannot determine anomaly score: predict_proba returned fewer than 2 columns.")
                anomaly_scores = probabilities[:, 1]

            len_to_copy = min(len(anomaly_scores), n_input_samples)
            if len(anomaly_scores) != n_input_samples:
                 warnings.warn(f"Score length ({len(anomaly_scores)}) != input samples ({n_input_samples}). Aligning output.", RuntimeWarning)
            final_scores[:len_to_copy] = anomaly_scores[:len_to_copy]

        except Exception as e:
            warnings.warn(f"Anomaly score calculation failed: {e}. Returning NaNs.", RuntimeWarning)
            traceback.print_exc()

        return final_scores

    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Detects anomalies (predicts class 1) in new data.
        """
        if self.model is None or self.scaler is None or self.imputer is None or self.input_type is None:
            raise RuntimeError("Model is not trained or key components are missing. Cannot detect.")

        n_input_samples = len(detection_data) if isinstance(detection_data, pd.DataFrame) else detection_data.shape[0]
        final_anomalies = np.zeros(n_input_samples, dtype=bool) # Default to False

        if n_input_samples == 0: 
            return final_anomalies 

        try:
            # Prepare data (handles input type, reshape, scaling, imputation)
            X_processed_imputed, _, processed_names = self._prepare_data_for_model(
                detection_data, is_training=False, label_col=self.label_col
            )

            if X_processed_imputed.shape[0] == 0:
                warnings.warn("No processable data found for detection after preparation.", RuntimeWarning)
                return final_anomalies 

            # Prediction using the fitted base model
            predictions = self.model.predict(X_processed_imputed)
            anomalies = (predictions == 1)

            # Assign results - length should match processed data length
            # Note: _prepare_data_for_model handles alignment for DataFrames
            # For NumPy, input/output length should match unless input was empty
            len_to_copy = min(len(anomalies), n_input_samples)
            if len(anomalies) != n_input_samples:
                 warnings.warn(f"Detection length ({len(anomalies)}) != input samples ({n_input_samples}). This might happen with sequence models or data issues. Aligning output.", RuntimeWarning)
            final_anomalies[:len_to_copy] = anomalies[:len_to_copy]

        except Exception as e:
            warnings.warn(f"Anomaly detection failed: {e}. Returning False for affected samples.", RuntimeWarning)
            traceback.print_exc() # Print details for debugging
            # final_anomalies is already initialized with False

        return final_anomalies


    # --- METHOD FOR XAI (SHAP/LIME) ---
    def predict_proba(self, X_xai: np.ndarray) -> np.ndarray:
        """
        Prediction function for XAI methods (SHAP/LIME).
        Handles potential 3D input (if seq_len=1 for DF model) by reshaping.
        Ensures data is scaled and imputed before prediction.
        Returns probabilities for all classes.
        """
        if self.model is None or self.scaler is None or self.imputer is None or self.input_type is None \
            or self.n_original_features is None:
            raise RuntimeError("Model is not trained or key components (scaler/imputer/input_type/dims/names) are missing for XAI prediction.")
        
        if not isinstance(X_xai, np.ndarray):
            raise TypeError("Input X_xai for XAI must be a NumPy array.")

        n_instances = X_xai.shape[0]
        if n_instances == 0:
            n_classes = len(self.model.classes_) if hasattr(self.model, 'classes_') and self.model.classes_ is not None else 2
            return np.empty((0, n_classes))

        X_to_process_2d = None # Variable to hold the correctly shaped 2D data before scaling

        try:
            # --- Step 1: Validate input shape and reshape if necessary ---
            if self.input_type == 'dataframe':
                # DataFrame models expect 2D (n_instances, n_features) input internally,
                # but XAI wrappers might send 3D (n_instances, 1, n_features). Handle this.

                if X_xai.ndim == 3:
                    n_inst, seq_len, n_feat = X_xai.shape
                    # ALLOW 3D input IF seq_len is 1 for DataFrame models
                    if seq_len == 1:
                        if n_feat != self.n_original_features:
                             raise ValueError(f"XAI 3D input feature mismatch: got {n_feat} features, expected {self.n_original_features}")
                        # Reshape (n_instances, 1, n_features) to (n_instances, n_features)
                        # print(f"DEBUG (predict_proba DF): Reshaping XAI input {X_xai.shape} to 2D.") 
                        X_to_process_2d = X_xai.reshape(n_inst, n_feat)
                    else:
                        # If seq_len is not 1, then it's an invalid 3D shape for DF model
                        raise ValueError(f"XAI 3D input for DataFrame model must have seq_len=1, got seq_len={seq_len}")

                elif X_xai.ndim == 2:
                    if X_xai.shape[1] != self.n_original_features:
                        raise ValueError(f"XAI 2D input has {X_xai.shape[1]} features, expected {self.n_original_features}")
                    X_to_process_2d = X_xai # Use 2D input directly
                else:
                    # Reject dimensions other than 2 or 3 (with seq_len=1)
                    raise ValueError(f"XAI input must be 2D or 3D (with seq_len=1) for DataFrame-trained model, got {X_xai.ndim}D.")

            elif self.input_type == 'numpy':
                # NumPy models expect 3D (n_instances, seq_len, features) input from XAI and flatten it.
                if self.sequence_length is None: raise RuntimeError("Sequence length not set for NumPy-trained model.")
                if X_xai.ndim != 3:
                    raise ValueError(f"XAI input must be 3D (n_instances, seq_len, features) for NumPy-trained model, got {X_xai.ndim}D.")

                n_inst_np, seq_len_np, n_feat_np = X_xai.shape # Use different var names
                if seq_len_np != self.sequence_length: raise ValueError(f"XAI input seq len ({seq_len_np}) != train seq len ({self.sequence_length}).")
                if n_feat_np != self.n_original_features: raise ValueError(f"XAI input features ({n_feat_np}) != train features ({self.n_original_features}).")

                # Flatten 3D input to 2D for internal model
                n_flattened = seq_len_np * n_feat_np
                # # print(f"DEBUG (predict_proba NP): Reshaping XAI input {X_xai.shape} to 2D ({n_inst_np}, {n_flattened}).") # Optional debug
                X_to_process_2d = X_xai.reshape(n_inst_np, n_flattened)

            else:
                raise RuntimeError(f"Unsupported training input_type '{self.input_type}' for XAI.")

            # --- Step 2: Scaling ---
            if X_to_process_2d is None: # Should not happen if logic above is correct
                raise RuntimeError("Internal error: X_to_process_2d is None before scaling.")
            if self.scaler is None or not hasattr(self.scaler, 'scale_'):
                raise RuntimeError("Scaler has not been fitted.")
            X_scaled = self.scaler.transform(X_to_process_2d)

            # --- Step 3: Imputation ---
            if self.imputer is None or not hasattr(self.imputer, 'statistics_'):
                raise RuntimeError("Imputer has not been fitted.")
            X_imputed = self.imputer.transform(X_scaled)

            # Final check for NaNs before prediction in XAI context
            if np.isnan(X_imputed).any():
                warnings.warn("NaNs detected in XAI input *after* imputation. Applying nan_to_num.", RuntimeWarning)
                X_imputed = np.nan_to_num(X_imputed, nan=0.0) # Fallback for XAI

            # --- Step 4: Predict probabilities using the internal model ---
            if X_imputed is None: # Should not happen
                 raise RuntimeError("Internal error: Imputed data for XAI prediction is None.")

            probabilities = self.model.predict_proba(X_imputed)

        except Exception as e:
            # Catch errors during preprocessing or prediction within XAI
            # Log detailed error before raising generic one
            # print(f"ERROR during predict_proba execution: {type(e).__name__} - {e}")
            traceback.print_exc() # Print full traceback for debugging
            raise RuntimeError(f"XAI prediction failed during preprocessing or model prediction. Error: {e}") from e

        # --- Step 5: Validate output shape ---
        # Check if model is fitted and has classes_ attribute
        if not hasattr(self.model, 'classes_') or self.model.classes_ is None:
             # This can happen if fit failed or model doesn't expose classes_
             warnings.warn("Model classes_ attribute not available. Cannot validate output probability shape accurately.", RuntimeWarning)
             expected_cols = None 
        else:
             expected_cols = len(self.model.classes_)
        
        # Perform shape validation if possible
        if expected_cols is not None:
            if probabilities.ndim != 2 or probabilities.shape[0] != n_instances or probabilities.shape[1] != expected_cols:
                warnings.warn(f"predict_proba output shape {probabilities.shape} unexpected. Expected ({n_instances}, {expected_cols}). Check model.", RuntimeWarning)
                # Decide how to handle this? Return as is? Raise error? Pad?
                # For now, returning the potentially incorrect shape with a warning.

        return probabilities


    # --- Optional: Method to get validation scores ---
    def get_validation_scores(self) -> Dict[str, float]:
        """Returns the computed validation scores."""
        if not hasattr(self, 'validation_scores_') or not self.validation_scores_:
             # print("Validation scores not available (model not run, validation failed, or validation_set_size=0).")
             return {}
        return self.validation_scores_
        
    # --- Optional: Method to get feature importances ---
    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Returns feature importances from the trained decision tree, 
           using the processed (potentially flattened) feature names."""
        if self.model is None:
            # print("Model not trained yet.")
            return None
            
        if not hasattr(self.model, 'feature_importances_'):
            # print("Model does not have feature_importances_ attribute.")
            return None

        # Use the processed_feature_names which correspond to the model's input
        if self.processed_feature_names is None:
            # print("Processed feature names not available.")
            # Return importances without names if names aren't stored
            return {f"feature_{i}": imp for i, imp in enumerate(self.model.feature_importances_)}

        importances = self.model.feature_importances_
        if len(importances) != len(self.processed_feature_names):
            warnings.warn(f"Mismatch between number of importances ({len(importances)}) and processed feature names ({len(self.processed_feature_names)}). Returning potentially misaligned results.")
            # Try to return anyway, might be misaligned
            max_len = min(len(importances), len(self.processed_feature_names))
            return {self.processed_feature_names[i]: importances[i] for i in range(max_len)}
            
        return dict(zip(self.processed_feature_names, importances))
    
    def get_validation_scores(self) -> Dict[str, float]:
        """Returns the average validation scores from k-fold cross-validation."""
        if not self.validation_scores_: # Checks if dict is empty
             # print("K-fold cross-validation scores not available (model not run or CV failed).")
             return {}
        # Check if scores are NaN, which might happen if all folds failed for a metric
        # if all(np.isnan(score) for score in self.validation_scores_.values()):
            # print("K-fold cross-validation scores are all NaN (likely due to issues in all folds).")
        return self.validation_scores_

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        # This method uses the final self.model trained on the full dataset.
        # Its internal logic remains the same.
        if self.model is None:
            # print("Model not trained yet (final model).")
            return None
        if not hasattr(self.model, 'feature_importances_'):
            # print("Final model does not have feature_importances_ attribute.")
            return None
        if self.processed_feature_names is None:
            # print("Processed feature names not available for final model.")
            return {f"feature_{i}": imp for i, imp in enumerate(self.model.feature_importances_)}

        importances = self.model.feature_importances_
        if len(importances) != len(self.processed_feature_names):
            warnings.warn(f"Mismatch in importances and processed feature names for final model.")
            max_len = min(len(importances), len(self.processed_feature_names))
            return {self.processed_feature_names[i]: importances[i] for i in range(max_len)}
        return dict(zip(self.processed_feature_names, importances))