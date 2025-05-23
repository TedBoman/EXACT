import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV # k-fold cross-validation
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score # For direct evaluation
from ML_models import model_interface 
from typing import List, Dict, Optional, Tuple, Union
import warnings

class XGBoostModel(model_interface.ModelInterface):
    """
    Supervised XGBoost classification model for anomaly detection using labeled data,
    with k-fold cross-validation for performance estimation.

    Handles input as:
    1.  Pandas DataFrame: Converts features directly to a 2D NumPy array.
    2.  3D NumPy array (X) and 1D NumPy array (y): Flattens the last two dimensions.

    Performance is estimated using k-fold cross-validation.
    A final XGBClassifier model is then trained on the entire dataset.
    Handles class imbalance using 'scale_pos_weight'.
    """

    def __init__(self, **kwargs):
        """
        Initializes the XGBoost classifier model.

        Args:
            n_estimators (int): Number of boosting rounds (default: 100).
            learning_rate (float): Step size shrinkage (default: 0.1).
            max_depth (int): Maximum depth of a tree (default: 6).
            objective (str): Learning task (default: 'binary:logistic').
            eval_metric (str or list): Metric for XGBoost's internal eval during CV folds (default: 'logloss').
                                    Note: For early stopping, if a list is provided, the last metric is used.
            random_state (int): Random seed (default: 42).
            n_jobs (int): Number of parallel threads (default: -1).
            n_splits (int): Number of folds for StratifiedKFold cross-validation (default: 5).
            shuffle_kfold (bool): Whether to shuffle data before k-fold splitting (default: True).
            early_stopping_rounds (int, optional): Activates early stopping. XGBoost stops if eval metric
                                                   doesn't improve in this many rounds for a fold (default: None).
            validation_metrics (list): Metrics to compute and average across folds for final reporting
                                       (e.g., ['roc_auc', 'f1', 'accuracy'], default: ['roc_auc', 'f1']).
                                       These are calculated manually after each fold's training.
            verbose_eval (bool, int): Verbosity for XGBoost training within folds (default: False).
            ... other XGBClassifier parameters ...
        """
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.input_type: Optional[str] = None
        self.processed_feature_names: Optional[List[str]] = None
        self.sequence_length: Optional[int] = None
        self.n_original_features: Optional[int] = None
        self.label_col: Optional[str] = None

        self.validation_scores_: Dict[str, float] = {} # Stores average validation metrics from k-fold
        self.avg_best_iteration_cv_: Optional[float] = None # Average best iteration from CV if early stopping used
        self.avg_best_score_cv_: Optional[float] = None # Average best score (from eval_metric) from CV

        self.random_state = kwargs.get('random_state', 42)
        self.n_splits = kwargs.get('n_splits', 5)
        self.shuffle_kfold = kwargs.get('shuffle_kfold', True)
        if self.n_splits <= 1 and self.n_splits != 0 : # Allow 0 if user wants to skip CV
             warnings.warn(f"n_splits is {self.n_splits}. If >0, it must be >1. Setting to 2. If you want to skip CV, set n_splits=0.")
             if self.n_splits > 0 : self.n_splits = 2 # Correct to a valid value if > 0 but <=1
        # if self.n_splits == 0:
             # print("n_splits is 0. K-Fold Cross-Validation will be skipped.")


        # Metrics for manual calculation and averaging after each fold
        self.user_validation_metrics = kwargs.get('validation_metrics', ['roc_auc', 'f1', 'accuracy'])

        # Hyperparameter Tuning specific parameters
        self.auto_tune = kwargs.get('auto_tune', False)
        self.hyperparam_grid = kwargs.get('hyperparam_grid', None)
        self.n_iter_search = kwargs.get('n_iter_search', 10)
        self.search_scoring_metric = kwargs.get('search_scoring_metric', 'roc_auc')
        self.search_cv_splits = kwargs.get('search_cv_splits', 3)
        self.search_verbose = kwargs.get('search_verbose', 1)

        if self.auto_tune:
            if self.hyperparam_grid is None:
                self.hyperparam_grid = {
                    'n_estimators': [50, 100, 150, 200, 250, 500, 1000],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
                    'max_depth': [3, 4, 5, 6, 7, 8, 10, 12],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2, 0.3],
                    'min_child_weight': [1, 3, 5, 7],
                    'reg_alpha':[0.001, 0.01, 0.1, 1],
                }
                # print("XGBoostModel auto_tune=True: Using default hyperparameter grid.")
            # else:
                # print("XGBoostModel auto_tune=True: Using provided hyperparameter grid.")
            if not isinstance(self.hyperparam_grid, dict) or not self.hyperparam_grid:
                 raise ValueError("Hyperparameter grid must be a non-empty dictionary when auto_tune is True.")
            # print(f"  RandomizedSearchCV: n_iter={self.n_iter_search}, scoring='{self.search_scoring_metric}', "
                #   f"inner_cv_splits={self.search_cv_splits}, verbose={self.search_verbose}")
            if self.search_cv_splits <= 1:
                raise ValueError("search_cv_splits for RandomizedSearchCV must be greater than 1.")


        self.model_params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'max_depth': kwargs.get('max_depth', 6),
            'objective': kwargs.get('objective', 'binary:logistic'),
            'eval_metric': kwargs.get('eval_metric', 'logloss'),
            'random_state': self.random_state,
            'n_jobs': kwargs.get('n_jobs', -1) # n_jobs for XGBoost itself
        }
        
        allowed_xgb_params = set(xgb.XGBClassifier().get_params().keys())
        # Parameters managed by this class constructor or tuning setup
        managed_params = {
            'scale_pos_weight', 'early_stopping_rounds', 'n_splits', 'shuffle_kfold',
            'validation_metrics', 'verbose_eval',
            'auto_tune', 'hyperparam_grid', 'n_iter_search', 'search_scoring_metric',
            'search_cv_splits', 'search_verbose'
        }
        # Add any other kwargs that are valid for XGBClassifier and not already handled
        extra_xgb_params = {
            k: v for k, v in kwargs.items()
            if k in allowed_xgb_params
            and k not in self.model_params # Not already set by defaults above
            and k not in managed_params   # Not one of the class's own managed params
        }
        self.model_params.update(extra_xgb_params)

        self.early_stopping_rounds = kwargs.get('early_stopping_rounds', None)
        self.verbose_eval = kwargs.get('verbose_eval', False)

        # print(f"XGBoostModel Initialized with base params: {self.model_params}")
        # if self.n_splits > 0:
            # print(f"K-fold CV: n_splits={self.n_splits}, shuffle={self.shuffle_kfold}")
        # print(f"Early Stopping Rounds (per fold/final): {self.early_stopping_rounds}, XGBoost Verbose Eval: {self.verbose_eval}")
        # print(f"User-defined CV metrics for averaging: {self.user_validation_metrics}")

    def _prepare_data_for_model(
        self, X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        # time_steps is now ignored for DataFrame input
        time_steps: Optional[int] = None,
        label_col: Optional[str] = None,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Internal helper to preprocess data.
        For DataFrames: Converts features to 2D NumPy, scales.
        For NumPy: Reshapes 3D to 2D, scales.

        Returns:
        - X_processed_scaled: Processed and scaled features (2D NumPy array).
        - y_aligned: Aligned labels (1D NumPy array, only if y is provided).
        - feature_names: List of feature names for the columns in X_processed_scaled.
        """
        if isinstance(X, pd.DataFrame):
            # print("Processing DataFrame input as 2D NumPy (no lagging)...")
            self.input_type = 'dataframe'
            self.sequence_length = 1 # No sequence dimension

            if is_training:
                if label_col is None or label_col not in X.columns:
                    raise ValueError(f"Label column '{label_col}' not found.")
                self.label_col = label_col
                original_feature_names = X.columns.drop(label_col).tolist()
                if not original_feature_names: raise ValueError("No feature columns found.")
                self.n_original_features = len(original_feature_names)
                self.processed_feature_names = original_feature_names

                X_features_df = X[original_feature_names]
                y_series = X[self.label_col]

                # Convert features directly to 2D NumPy array
                X_features_np = X_features_df.to_numpy()
                y_aligned = y_series.to_numpy()

                if X_features_np.shape[0] == 0: raise ValueError("No data after feature extraction.")

                # Scaling
                self.scaler = MinMaxScaler()
                X_processed_scaled = self.scaler.fit_transform(X_features_np)

                return X_processed_scaled, y_aligned, self.processed_feature_names

            else: # Detection for DataFrame (convert to 2D NumPy)
                if self.scaler is None or self.processed_feature_names is None or self.n_original_features is None:
                    raise RuntimeError("Model (trained on DataFrame) not ready.")
                missing_cols = set(self.processed_feature_names) - set(X.columns)
                if missing_cols: raise ValueError(f"Missing required columns: {missing_cols}")

                X_features_df = X[self.processed_feature_names]
                X_features_np = X_features_df.to_numpy()

                if X_features_np.shape[0] == 0:
                    warnings.warn("No data provided for detection.", RuntimeWarning)
                    return np.empty((0, self.n_original_features)), None, self.processed_feature_names

                # Scaling
                X_processed_scaled = self.scaler.transform(X_features_np)
                return X_processed_scaled, None, self.processed_feature_names # No labels

        elif isinstance(X, np.ndarray):
            # Handle both 3D and 2D NumPy input, especially for inference
            # print(f"DEBUG _prepare_data: Processing NumPy input shape {X.shape}")

            original_ndim = X.ndim
            temp_X = X # Work with a temporary variable

            # If input is 2D during inference, reshape to 3D assuming seq_len=1
            if not is_training and original_ndim == 2:
                n_samples_2d, n_feat_2d = temp_X.shape
                # Check if feature count matches expected original features
                if self.n_original_features is not None and n_feat_2d != self.n_original_features:
                    raise ValueError(f"Inference 2D NumPy input has {n_feat_2d} features, expected {self.n_original_features}.")
                temp_X = temp_X[:, np.newaxis, :] # Reshape to (samples, 1, features)
                # print(f"DEBUG _prepare_data: Reshaped 2D NumPy input to 3D {temp_X.shape} for inference.")

            # Now proceed assuming temp_X is 3D (either originally or after reshape)
            if temp_X.ndim != 3:
                # This check should now primarily catch invalid initial shapes other than 2D during inference
                raise ValueError(f"NumPy X must be 3D (or 2D during inference), got {original_ndim}D initially.")

            n_samples, seq_len, n_feat = temp_X.shape

            if is_training:
                # Training path expects 3D input initially
                if original_ndim != 3: # Ensure original input was 3D for training
                    raise ValueError(f"NumPy training input must be 3D, got {original_ndim}D.")

                # print(f"DEBUG _prepare_data (Train): NumPy input shape {temp_X.shape}") # Add shape print
                self.input_type = 'numpy'
                self.sequence_length = seq_len
                self.n_original_features = n_feat

                if y is None: raise ValueError("Labels 'y' required for NumPy training.")
                if not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != n_samples:
                    raise ValueError("Invalid 'y' for NumPy training.")

                # Flatten 3D -> 2D for XGBoost training
                X_flattened = temp_X.reshape(n_samples, seq_len * n_feat)
                # print(f"DEBUG _prepare_data (Train): Flattened 3D to 2D for scaling/training: {X_flattened.shape}")

                # Scaling (Fit and Transform flattened data)
                self.scaler = MinMaxScaler()
                X_processed_scaled = self.scaler.fit_transform(X_flattened)
                # print(f"DEBUG _prepare_data (Train): Scaled 2D data shape: {X_processed_scaled.shape}")

                # Generate flattened feature names
                n_flattened_features = seq_len * n_feat
                self.processed_feature_names = [f"feature_{i}_step_{j}" for j in range(seq_len) for i in range(n_feat)]
                if len(self.processed_feature_names) != n_flattened_features:
                    warnings.warn("Feature name generation mismatch.")
                    self.processed_feature_names = [f"flat_feature_{k}" for k in range(n_flattened_features)]

                y_aligned = y
                # Return 2D scaled data for XGBoost training
                return X_processed_scaled, y_aligned, self.processed_feature_names

            else: # Detection/Inference for NumPy input
                # print(f"DEBUG _prepare_data (Inference): NumPy input shape {temp_X.shape}")
                if self.scaler is None or self.sequence_length is None or self.n_original_features is None or self.processed_feature_names is None:
                    raise RuntimeError("Model (trained on NumPy or receiving NumPy for inference) not ready.")

                # Validate sequence length and features against trained values
                # Note: If input was 2D and reshaped, seq_len will be 1 here.
                # This check might need adjustment if models trained on numpy should *only* accept their original seq_len during inference.
                # For now, we allow seq_len=1 if input was 2D.
                if self.input_type == 'numpy' and seq_len != self.sequence_length:
                    # Only strictly enforce seq_len if model was trained on numpy
                    raise ValueError(f"Input seq len {seq_len} != train seq len {self.sequence_length} for NumPy-trained model.")
                if n_feat != self.n_original_features:
                    raise ValueError(f"Input features {n_feat} != train features {self.n_original_features}.")

                # Flatten 3D -> 2D for XGBoost prediction
                X_flattened = temp_X.reshape(n_samples, seq_len * n_feat)
                # print(f"DEBUG _prepare_data (Inference): Flattened 3D to 2D for scaling/prediction: {X_flattened.shape}")

                # Scaling (Transform only)
                X_processed_scaled = self.scaler.transform(X_flattened)
                # print(f"DEBUG _prepare_data (Inference): Scaled 2D data shape: {X_processed_scaled.shape}")

                # Return 2D scaled data for XGBoost prediction
                return X_processed_scaled, None, self.processed_feature_names # No labels

        else:
            raise TypeError("Input 'X' must be pandas DataFrame or NumPy array.")


    def run(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None, label_col: str = 'label'):
        """
        Prepares data, optionally performs hyperparameter tuning with RandomizedSearchCV,
        then (if n_splits > 0) performs k-fold cross-validation for performance estimation,
        and finally trains a final XGBoost model on the entire dataset.
        """
        # print(f"Running training for XGBoostModel (Input type: {'DataFrame' if isinstance(X, pd.DataFrame) else 'NumPy'})...")

        # --- Step 1: Prepare FULL data (fits scaler globally) ---
        if isinstance(X, pd.DataFrame):
            if y is not None: warnings.warn("Arg 'y' ignored for DataFrame input.", UserWarning)
            X_processed_full, y_aligned_full, _ = self._prepare_data_for_model(
                X, y=None, label_col=label_col, is_training=True
            )
        elif isinstance(X, np.ndarray):
            X_processed_full, y_aligned_full, _ = self._prepare_data_for_model(
                X, y=y, label_col=None, is_training=True
            )
        else:
            raise TypeError("Input 'X' must be pandas DataFrame or NumPy array.")

        if X_processed_full.shape[0] == 0 or y_aligned_full is None:
            warnings.warn("No data or labels for training after preprocessing. Model not trained.", RuntimeWarning)
            self.model = None
            return
        
        if self.n_splits > 0 and X_processed_full.shape[0] < self.n_splits:
            warnings.warn(f"Number of samples ({X_processed_full.shape[0]}) is less than n_splits for CV ({self.n_splits}). K-fold CV might fail or results may be unreliable.", RuntimeWarning)
        if self.auto_tune and X_processed_full.shape[0] < self.search_cv_splits:
            warnings.warn(f"Number of samples ({X_processed_full.shape[0]}) is less than search_cv_splits ({self.search_cv_splits}). Hyperparameter search might fail or be unreliable.", RuntimeWarning)


        # --- Step 1.5: Hyperparameter Tuning (Optional) ---
        if self.auto_tune:
            # print("\n" + "-" * 30)
            # print("Starting Hyperparameter Tuning with RandomizedSearchCV...")
            
            n_neg_search = np.sum(y_aligned_full == 0)
            n_pos_search = np.sum(y_aligned_full == 1)
            scale_pos_weight_search = 1.0
            if n_pos_search > 0 and n_neg_search > 0:
                scale_pos_weight_search = float(n_neg_search) / float(n_pos_search)
            elif n_pos_search == 0:
                warnings.warn("No positive samples in data for hyperparameter search scale_pos_weight calculation. Using 1.0.", RuntimeWarning)
            elif n_neg_search == 0:
                 warnings.warn("No negative samples in data for hyperparameter search scale_pos_weight calculation. Using 1.0.", RuntimeWarning)
            # print(f"  Using scale_pos_weight for search estimator: {scale_pos_weight_search:.4f} (calculated on full preprocessed data)")

            search_estimator_base_params = self.model_params.copy()
            search_estimator_base_params['scale_pos_weight'] = scale_pos_weight_search
            search_estimator_base_params.setdefault('max_delta_step', 1) 

            for tuned_param_key in self.hyperparam_grid.keys():
                if tuned_param_key in search_estimator_base_params:
                    del search_estimator_base_params[tuned_param_key]
            
            search_estimator_base_params.setdefault('objective', 'binary:logistic') # Ensure core params
            search_estimator_base_params.setdefault('random_state', self.random_state)
            # n_jobs for estimator in search is inherited from self.model_params['n_jobs']
            # eval_metric for estimator in search is inherited from self.model_params['eval_metric']

            estimator_for_search = xgb.XGBClassifier(**search_estimator_base_params)

            search_cv_strategy = StratifiedKFold(
                n_splits=self.search_cv_splits, 
                shuffle=True, 
                random_state=self.random_state
            )
            
            # Make sure n_iter is not more than total combinations if grid is small
            num_param_combinations = 1
            if isinstance(self.hyperparam_grid, dict):
                for k in self.hyperparam_grid:
                    num_param_combinations *= len(self.hyperparam_grid[k])
            
            actual_n_iter = min(self.n_iter_search, num_param_combinations)
            # if actual_n_iter < self.n_iter_search:
                # print(f"  Note: n_iter_search ({self.n_iter_search}) is > total param combinations ({num_param_combinations}). Using n_iter={actual_n_iter}.")


            random_search = RandomizedSearchCV(
                estimator=estimator_for_search,
                param_distributions=self.hyperparam_grid,
                n_iter=actual_n_iter,
                scoring=self.search_scoring_metric,
                n_jobs=-1,  # Parallelize search trials
                cv=search_cv_strategy,
                random_state=self.random_state,
                verbose=self.search_verbose
            )

            # print(f"  Fitting RandomizedSearchCV on {X_processed_full.shape[0]} samples...")
            search_fit_params = {} # No early stopping during RandomizedSearchCV's internal folds for simplicity.
                                   # n_estimators from grid will be used as-is.
            # if self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
                 # print("  Note: Configured early_stopping_rounds is NOT applied *during* RandomizedSearchCV's internal folds.")
                 # print("        It will be applied during the subsequent main K-Fold CV and final model fit, using the tuned n_estimators as a max.")

            try:
                if X_processed_full.shape[0] < self.search_cv_splits:
                     warnings.warn(f"  Cannot run RandomizedSearchCV: number of samples ({X_processed_full.shape[0]}) is less than search_cv_splits ({self.search_cv_splits}). "
                                   "Skipping hyperparameter tuning.", RuntimeWarning)
                else:
                    random_search.fit(X_processed_full, y_aligned_full, **search_fit_params)
                    # print(f"  Best parameters found by RandomizedSearchCV: {random_search.best_params_}")
                    # print(f"  Best score ('{self.search_scoring_metric}') from RandomizedSearchCV: {random_search.best_score_:.4f}")
                    self.model_params.update(random_search.best_params_)
                    # print(f"  self.model_params updated to: {self.model_params}")

            except Exception as e:
                warnings.warn(f"Hyperparameter tuning with RandomizedSearchCV failed: {e}. "
                              "Proceeding with original/default model parameters.", RuntimeWarning)
            # print("-" * 30 + "\n")
        # --- End of Hyperparameter Tuning ---


        # --- Step 2: K-Fold Cross-Validation for Performance Estimation (if n_splits > 0) ---
        if self.n_splits > 0:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle_kfold, random_state=self.random_state)
            
            fold_scores: Dict[str, List[float]] = {metric: [] for metric in self.user_validation_metrics}
            fold_best_iterations = []
            fold_best_eval_scores = []
            fold_count = 0

            # print(f"Starting {self.n_splits}-fold cross-validation for XGBoost with effective params: {self.model_params}...")
            for fold_idx, (train_index, val_index) in enumerate(skf.split(X_processed_full, y_aligned_full)):
                fold_count += 1
                # print(f"  Fold {fold_idx + 1}/{self.n_splits}...")
                X_train_fold, X_val_fold = X_processed_full[train_index], X_processed_full[val_index]
                y_train_fold, y_val_fold = y_aligned_full[train_index], y_aligned_full[val_index]

                if X_train_fold.shape[0] == 0 or X_val_fold.shape[0] == 0:
                    warnings.warn(f"    Fold {fold_idx+1} has empty train/validation set. Skipping.", RuntimeWarning)
                    continue
                if len(np.unique(y_train_fold)) < 2 and self.n_splits > 1 : # XGBoost might error if only one class in train
                    warnings.warn(f"    Fold {fold_idx+1} train set has only one class. Skipping fold as XGBoost may error.", RuntimeWarning)
                    continue


                n_neg_train_fold = np.sum(y_train_fold == 0); n_pos_train_fold = np.sum(y_train_fold == 1)
                scale_pos_weight_fold = 1.0
                if n_pos_train_fold == 0: warnings.warn(f"    No positive samples in train part of fold {fold_idx+1}. scale_pos_weight=1.0.", RuntimeWarning)
                elif n_neg_train_fold == 0: warnings.warn(f"    No negative samples in train part of fold {fold_idx+1}. scale_pos_weight=1.0.", RuntimeWarning)
                else: scale_pos_weight_fold = float(n_neg_train_fold) / float(n_pos_train_fold)
                # print(f"    Fold {fold_idx+1} scale_pos_weight: {scale_pos_weight_fold:.4f}")

                current_fold_model_params = self.model_params.copy()
                current_fold_model_params['scale_pos_weight'] = scale_pos_weight_fold
                current_fold_model_params.setdefault('max_delta_step', 1)

                temp_model_fold = xgb.XGBClassifier(**current_fold_model_params)
                
                fit_params_fold = {}
                if self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
                    if X_val_fold.shape[0] > 0 and len(np.unique(y_val_fold)) > 1: # Ensure val set is usable
                        fit_params_fold['early_stopping_rounds'] = self.early_stopping_rounds
                        fit_params_fold['eval_set'] = [(X_val_fold, y_val_fold)]
                        fit_params_fold['verbose'] = self.verbose_eval
                    else:
                        warnings.warn(f"    Fold {fold_idx+1}: Cannot use early stopping due to empty or single-class validation set. Fitting without it.", RuntimeWarning)
                        fit_params_fold['verbose'] = self.verbose_eval # Still respect verbosity
                elif self.verbose_eval:
                    if X_val_fold.shape[0] > 0:
                         fit_params_fold['eval_set'] = [(X_val_fold, y_val_fold)] # For verbose output
                    fit_params_fold['verbose'] = self.verbose_eval
                
                try:
                    temp_model_fold.fit(X_train_fold, y_train_fold, **fit_params_fold)
                    if self.early_stopping_rounds and 'early_stopping_rounds' in fit_params_fold and hasattr(temp_model_fold, 'best_iteration') and temp_model_fold.best_iteration is not None:
                        fold_best_iterations.append(temp_model_fold.best_iteration)
                        if hasattr(temp_model_fold, 'best_score') and temp_model_fold.best_score is not None:
                            fold_best_eval_scores.append(temp_model_fold.best_score)
                        # print(f"    Fold {fold_idx+1} - Best Iteration (early stopping): {temp_model_fold.best_iteration if temp_model_fold.best_iteration is not None else 'N/A'}")

                except Exception as e:
                    warnings.warn(f"    XGBoost fitting failed for fold {fold_idx + 1}: {e}. Skipping fold evaluation.", RuntimeWarning)
                    continue # Skip to next fold

                # Manual evaluation using user-defined metrics
                if X_val_fold.shape[0] > 0:
                    try:
                        y_pred_val_fold = temp_model_fold.predict(X_val_fold)
                        y_proba_val_fold = temp_model_fold.predict_proba(X_val_fold)[:, 1]

                        for metric_name in self.user_validation_metrics:
                            score = np.nan
                            try:
                                if metric_name == 'accuracy':
                                    score = accuracy_score(y_val_fold, y_pred_val_fold)
                                elif metric_name == 'f1':
                                    score = f1_score(y_val_fold, y_pred_val_fold, zero_division=0)
                                elif metric_name == 'roc_auc':
                                    if len(np.unique(y_val_fold)) > 1:
                                        score = roc_auc_score(y_val_fold, y_proba_val_fold)
                                    else:
                                        warnings.warn(f"    ROC AUC undefined for fold {fold_idx + 1} (single class in y_val_fold). Score is NaN.", RuntimeWarning)
                                else:
                                    warnings.warn(f"    Unsupported user metric '{metric_name}' for fold {fold_idx + 1}.", UserWarning)
                                    continue
                                fold_scores[metric_name].append(score)
                                # print(f"    Fold {fold_idx + 1} User Metric '{metric_name}': {score:.4f}")
                            except Exception as metric_e:
                                # print(f"    Failed to calculate user metric '{metric_name}' for fold {fold_idx + 1}: {metric_e}")
                                fold_scores[metric_name].append(np.nan)
                    except Exception as eval_e:
                        print(f"    Failed during manual validation evaluation for fold {fold_idx + 1}: {eval_e}")
                else:
                    # print(f"    Fold {fold_idx+1}: Skipping evaluation metrics as validation set is empty.")
                    for metric_name in self.user_validation_metrics:
                        fold_scores[metric_name].append(np.nan)


            if fold_count == 0 and self.n_splits > 0:
                warnings.warn("XGBoost K-fold CV: No folds successfully processed. Validation scores will be empty/NaN.", RuntimeWarning)
                self.validation_scores_ = {metric: np.nan for metric in self.user_validation_metrics}
            elif self.n_splits > 0:
                # print("\nCross-validation summary (XGBoost):")
                for metric_name in self.user_validation_metrics:
                    valid_metric_scores = [s for s in fold_scores[metric_name] if not np.isnan(s)]
                    if valid_metric_scores:
                        avg_score = np.mean(valid_metric_scores)
                        std_score = np.std(valid_metric_scores)
                        self.validation_scores_[metric_name] = avg_score
                        # print(f"  Average User Metric '{metric_name}': {avg_score:.4f} (Std: {std_score:.4f}) from {len(valid_metric_scores)} folds")
                    else:
                        self.validation_scores_[metric_name] = np.nan
                        # print(f"  Average User Metric '{metric_name}': NaN (no valid scores from folds)")
                
                if fold_best_iterations:
                    self.avg_best_iteration_cv_ = np.mean(fold_best_iterations)
                    # print(f"  Average Best Iteration (from CV with early stopping): {self.avg_best_iteration_cv_:.2f}")
                if fold_best_eval_scores:
                    self.avg_best_score_cv_ = np.mean(fold_best_eval_scores)
                    # print(f"  Average Best XGBoost Eval Metric Score (from CV): {self.avg_best_score_cv_:.4f}")
            # print("-" * 30)
        else: # n_splits == 0
            # print("K-Fold Cross-Validation skipped as n_splits=0.")
            self.validation_scores_ = {metric: np.nan for metric in self.user_validation_metrics}
            self.avg_best_iteration_cv_ = None
            self.avg_best_score_cv_ = None


        # --- Step 3: Training the FINAL XGBoost Model on the ENTIRE Prepared Dataset ---
        # print(f"Training final XGBoost model on {X_processed_full.shape[0]} samples, {X_processed_full.shape[1]} features...")
        # print(f"  Using final model_params: {self.model_params}")

        final_model_params = self.model_params.copy()
        n_neg_full = np.sum(y_aligned_full == 0); n_pos_full = np.sum(y_aligned_full == 1)
        scale_pos_weight_full = 1.0
        if n_pos_full == 0: warnings.warn("No positive samples in ENTIRE dataset for final model. scale_pos_weight=1.0.", RuntimeWarning)
        elif n_neg_full == 0: warnings.warn("No negative samples in ENTIRE dataset for final model. scale_pos_weight=1.0.", RuntimeWarning)
        else: scale_pos_weight_full = float(n_neg_full) / float(n_pos_full)
        final_model_params['scale_pos_weight'] = scale_pos_weight_full
        final_model_params.setdefault('max_delta_step', 1)
        
        # print(f"Final model scale_pos_weight: {scale_pos_weight_full:.4f}")
        
        # Adjust n_estimators for final model based on CV's avg_best_iteration_cv_ IF early stopping was effective in CV
        # AND n_estimators was NOT part of hyperparam_grid (or if tuning wasn't run).
        # If n_estimators WAS tuned, we should generally respect the tuned value.
        # However, if early stopping was also used in CV with that tuned n_estimators, avg_best_iteration_cv_ might suggest
        # an even lower optimal number of trees for the full dataset.
        
        original_n_estimators = self.model_params.get('n_estimators') # This could be initial or tuned
        
        if self.early_stopping_rounds and self.avg_best_iteration_cv_ is not None and self.avg_best_iteration_cv_ > 0:
            candidate_n_estimators = max(1, int(np.ceil(self.avg_best_iteration_cv_)))
            
            if self.auto_tune and 'n_estimators' in self.hyperparam_grid:
                 # If n_estimators was tuned, use the tuned value as the upper bound
                 # and avg_best_iteration_cv_ as a potentially lower value found by early stopping.
                 final_n_estimators = min(original_n_estimators, candidate_n_estimators)
                #  if final_n_estimators < candidate_n_estimators:
                     # print(f"  Note: Tuned n_estimators ({original_n_estimators}) is lower than avg_best_iteration_cv_ from CV ({candidate_n_estimators}). "
                        #    f"Using {final_n_estimators} for final model (min of tuned and CV-derived).")
                #  else:
                     # print(f"  Adjusting final model n_estimators from tuned {original_n_estimators} to {final_n_estimators} based on average best_iteration from CV.")

            else: # n_estimators was not tuned, so avg_best_iteration_cv_ directly informs it.
                final_n_estimators = candidate_n_estimators
                # print(f"  Adjusting final model n_estimators from initial {original_n_estimators} to {final_n_estimators} based on average best_iteration from CV.")

            final_model_params['n_estimators'] = final_n_estimators
        else:
            # Use n_estimators from self.model_params (either initial, or tuned by RandomizedSearchCV if early stopping in CV was not used/effective)
            # print(f"  Using n_estimators ({original_n_estimators}) for final model (from initial or tuned params, no CV early stopping adjustment).")
            final_model_params['n_estimators'] = original_n_estimators # Ensure it's explicitly set


        self.model = xgb.XGBClassifier(**final_model_params)
        try:
            # For the final model, early stopping can be used if you have a mental "validation set" idea,
            # but typically it's trained on all data. If early_stopping_rounds is set, XGBoost itself won't use it
            # in fit() unless an eval_set is provided. Here, we train on all X_processed_full, y_aligned_full.
            # The n_estimators is already potentially adjusted.
            self.model.fit(X_processed_full, y_aligned_full, verbose=self.verbose_eval)
        except Exception as e:
            self.model = None
            # It's better to raise the error rather than just warn, as the model isn't usable.
            raise RuntimeError(f"Final XGBoost fitting failed on the full dataset: {e}") from e

        # print("Final XGBoost model training complete.")


    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ Detects anomalies (predicts class 1). """
        
        if self.model is None or self.scaler is None or self.input_type is None or self.processed_feature_names is None:
            raise RuntimeError("Model is not trained or ready.")

        n_input_samples = len(detection_data) if isinstance(detection_data, pd.DataFrame) else detection_data.shape[0]
        if n_input_samples == 0: return np.array([], dtype=bool)

        # Prepare data (returns 2D NumPy array)
        X_processed_scaled, _, _ = self._prepare_data_for_model(
            detection_data, is_training=False, label_col=self.label_col
        )

        if X_processed_scaled.shape[0] == 0:
            return np.zeros(n_input_samples, dtype=bool) # Return all False

        # Prediction using the base XGBoost model's predict method
        try:
            predictions = self.model.predict(X_processed_scaled)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e
        anomalies = (predictions == 1)

        # Result directly corresponds to input order
        if len(anomalies) != n_input_samples:
            warnings.warn(f"Output detection length ({len(anomalies)}) mismatch vs input ({n_input_samples}).", RuntimeWarning)
            final_anomalies = np.zeros(n_input_samples, dtype=bool)
            len_to_copy = min(len(anomalies), n_input_samples)
            final_anomalies[:len_to_copy] = anomalies[:len_to_copy]
            return final_anomalies

        return anomalies

    def predict_proba(self, X_input: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts class probabilities for the input data using the BASE XGBoost model.
        NOTE: These probabilities are NOT calibrated.

        Handles DataFrame or 3D NumPy input consistent with training.

        Args:
            X_input (Union[pd.DataFrame, np.ndarray]): Input data for probability prediction.

        Returns:
            np.ndarray: Array of shape (n_samples, 2) with probabilities for class 0 and class 1.
                      Returns empty array shape (0, 2) if no data after processing.
        """
        if self.model is None or self.scaler is None or self.input_type is None or self.processed_feature_names is None or self.n_original_features is None:
            raise RuntimeError("Model is not trained or ready for predict_proba.")

        # print(f"Predicting UNCALIBRATED probabilities for input type: {type(X_input)}")
        # Prepare data (returns 2D NumPy array suitable for the model)
        X_processed_scaled, _, _ = self._prepare_data_for_model(
            X_input, is_training=False, label_col=self.label_col
        )

        # Handle case where preprocessing results in no data
        if X_processed_scaled.shape[0] == 0:
            warnings.warn("No data to predict probabilities after preprocessing.", RuntimeWarning)
            return np.empty((0, 2)) # Return shape (0, 2)

        # print(f"Input shape to base model's predict_proba: {X_processed_scaled.shape}")
        try:
            # Use the predict_proba method of the base XGBoost model
            probabilities = self.model.predict_proba(X_processed_scaled)
        except Exception as e:
            raise RuntimeError(f"Probability prediction failed: {e}") from e

        if probabilities.shape[1] != 2:
            warnings.warn(f"Expected 2 columns in probability output, but got {probabilities.shape[1]}. Returning as is.", RuntimeWarning)

        # print(f"Predicted probabilities shape: {probabilities.shape}")
        return probabilities
    
    def get_validation_scores(self) -> Dict[str, float]:
        """Returns the average validation scores from k-fold cross-validation."""
        if not self.validation_scores_:
             # print("K-fold CV scores (user metrics) not available.")
             return {}
        # if all(np.isnan(score) for score in self.validation_scores_.values()):
            # print("K-fold CV scores (user metrics) are all NaN.")
        return self.validation_scores_

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Returns feature importances from the final trained XGBoost model."""
        if self.model is None:
            # print("Final model not trained yet.")
            return None
        if not hasattr(self.model, 'feature_importances_'):
            # print("Final model does not have feature_importances_ attribute.")
            return None
        if self.processed_feature_names is None:
            # print("Processed feature names not available for final model.")
            # Fallback if names are missing for some reason
            return {f"feature_{i}": imp for i, imp in enumerate(self.model.feature_importances_)}

        importances = self.model.feature_importances_
        if len(importances) != len(self.processed_feature_names):
            warnings.warn(f"Mismatch in importances ({len(importances)}) and processed feature names ({len(self.processed_feature_names)}) for final model.")
            max_len = min(len(importances), len(self.processed_feature_names))
            return {self.processed_feature_names[i]: importances[i] for i in range(max_len)}
        return dict(zip(self.processed_feature_names, importances))
