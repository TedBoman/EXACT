import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler # StandardScaler is often preferred for SGD
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, get_scorer
from sklearn.calibration import CalibratedClassifierCV # For probability estimates with hinge loss
from ML_models import model_interface
from typing import Dict, List, Optional, Tuple, Union
import warnings
import traceback

class SGDLinearModel(model_interface.ModelInterface):
    """
    Supervised Linear Model (SVM or Logistic Regression) using Stochastic Gradient Descent (SGD)
    for anomaly detection with labeled data, including k-fold cross-validation.

    Handles input as:
    1.  Pandas DataFrame: Converts features directly to a 2D NumPy array.
    2.  3D NumPy array (X) and 1D NumPy array (y): Flattens the last two dimensions.

    Includes internal scaling (StandardScaler by default) and imputation (SimpleImputer).
    Trains an SGDClassifier. Performance is estimated using k-fold cross-validation.
    A final model is then trained on the entire dataset.
    Handles class imbalance using the 'class_weight' parameter.
    """

    def __init__(self, **kwargs):
        """Initializes the SGD Linear classifier model.

        Args:
            loss (str): The loss function to be used. Defaults to 'hinge' (linear SVM).
                        Other options: 'log_loss' (logistic regression), 'modified_huber'.
            penalty (str): The penalty (aka regularization term) to be used. Defaults to 'l2'.
                           Options: 'l1', 'l2', 'elasticnet'.
            alpha (float): Constant that multiplies the regularization term. Defaults to 0.0001.
            max_iter (int): The maximum number of passes over the training data (epochs). Defaults to 1000.
            tol (float, optional): The stopping criterion. Defaults to 1e-3.
            class_weight (dict, 'balanced', optional): Class weights. Defaults to 'balanced'.
            random_state (int): Controls randomness. Defaults to 42.
            learning_rate (str): Learning rate schedule. Defaults to 'optimal'.
            eta0 (float): Initial learning rate for 'constant', 'invscaling', 'adaptive'. Defaults to 0.0.
            n_jobs (int, optional): Number of CPUs to use for OvA. -1 means all. Defaults to -1.
            imputer_strategy (str): Strategy for SimpleImputer. Defaults to 'mean'.
            scaler_type (str): Type of scaler ('standard' or 'minmax'). Defaults to 'standard'.
            n_splits (int): Folds for StratifiedKFold. Defaults to 5.
            shuffle_kfold (bool): Shuffle for k-fold. Defaults to True.
            validation_metrics (list): Metrics for CV. Defaults to ['accuracy', 'f1', 'roc_auc'].
            auto_tune (bool): Perform hyperparameter tuning. Defaults to False.
            search_n_iter (int): Iterations for RandomizedSearchCV. Defaults to 10.
            search_scoring (str): Scoring for RandomizedSearchCV. Defaults to 'f1'.
            param_dist (dict, optional): Parameter distribution for RandomizedSearchCV.
            calibrate_probabilities (bool): If True and loss is 'hinge' or 'squared_hinge',
                                            the final model will be wrapped in CalibratedClassifierCV
                                            to provide better probability estimates. Defaults to True.
            **kwargs: Additional parameters passed to SGDClassifier.
        """
        self.model: Optional[Union[SGDClassifier, CalibratedClassifierCV]] = None
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.imputer: Optional[SimpleImputer] = None
        self.input_type: Optional[str] = None
        self.processed_feature_names_: Optional[List[str]] = None
        self.original_feature_names_: Optional[List[str]] = None
        self.sequence_length: Optional[int] = None
        self.n_original_features: Optional[int] = None
        self.label_col: Optional[str] = None
        self._is_calibrated = False

        self.validation_scores_: Dict[str, float] = {}
        self.n_splits = kwargs.pop('n_splits', 5)
        self.shuffle_kfold = kwargs.pop('shuffle_kfold', True)
        if self.n_splits <= 1:
            raise ValueError("n_splits for k-fold CV must be > 1.")
        self.validation_metrics = kwargs.pop('validation_metrics', ['accuracy', 'f1', 'roc_auc'])

        random_state = kwargs.get('random_state', 42)

        self.auto_tune = kwargs.pop('auto_tune', False)
        self.search_n_iter = kwargs.pop('search_n_iter', 10)
        self.search_scoring = kwargs.pop('search_scoring', 'f1')
        self.tuned_best_params_: Optional[Dict] = None
        self.calibrate_probabilities = kwargs.pop('calibrate_probabilities', True)


        self.model_params = {
            'loss': kwargs.get('loss', 'hinge'),
            'penalty': kwargs.get('penalty', 'l2'),
            'alpha': kwargs.get('alpha', 0.0001),
            'max_iter': kwargs.get('max_iter', 1000),
            'tol': kwargs.get('tol', 1e-3),
            'class_weight': kwargs.get('class_weight', 'balanced'),
            'random_state': random_state,
            'learning_rate': kwargs.get('learning_rate', 'optimal'),
            'eta0': kwargs.get('eta0', 0.0),
            'n_jobs': kwargs.get('n_jobs', -1),
            'early_stopping': kwargs.get('early_stopping', False),
            'validation_fraction': kwargs.get('validation_fraction', 0.1),
            'n_iter_no_change': kwargs.get('n_iter_no_change', 5),
        }
        if self.model_params.get('class_weight') is None:
            self.model_params['class_weight'] = 'balanced'

        self._imputer_strategy = kwargs.pop('imputer_strategy', 'mean')
        self._scaler_type = kwargs.pop('scaler_type', 'standard').lower()
        if self._scaler_type not in ['standard', 'minmax']:
            warnings.warn(f"Invalid scaler_type '{self._scaler_type}'. Defaulting to 'standard'.")
            self._scaler_type = 'standard'

        if self.auto_tune:
            # Loss functions that naturally produce probabilities or can be easily calibrated
            prob_losses = ['log_loss', 'modified_huber']
            # Losses that often benefit from CalibratedClassifierCV for probabilities
            non_prob_losses_for_tuning = ['hinge', 'squared_hinge', 'perceptron']

            default_param_dist_sgd = {
                'loss': prob_losses + non_prob_losses_for_tuning,
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'max_iter': [500, 1000, 1500, 2000],
                'tol': [1e-4, 1e-3],
                'learning_rate': ['optimal', 'adaptive', 'constant', 'invscaling'],
                'eta0': [0.0001, 0.001, 0.01, 0.1],
                'class_weight': ['balanced', None],
            }
            self.param_dist = kwargs.pop('param_dist', default_param_dist_sgd)
            # print(f"SGDLinearModel: Auto-tuning ENABLED. Search iterations: {self.search_n_iter}, Main scoring: '{self.search_scoring}'")
            # print(f"Parameter distribution for tuning: {self.param_dist}")
            # (Validation metric mapping logic for search_scoring, similar to other models)
            if self.search_scoring not in self.validation_metrics:
                 original_search_scoring = self.search_scoring
                 if self.search_scoring.startswith('f1') and 'f1' in self.validation_metrics: self.search_scoring = 'f1'
                 elif self.search_scoring.startswith('roc_auc') and 'roc_auc' in self.validation_metrics: self.search_scoring = 'roc_auc'
                 elif self.search_scoring.startswith('accuracy') and 'accuracy' in self.validation_metrics: self.search_scoring = 'accuracy'
                 if self.search_scoring != original_search_scoring:
                     warnings.warn(f"search_scoring '{original_search_scoring}' mapped to '{self.search_scoring}'.", UserWarning)
                 if self.search_scoring not in self.validation_metrics:
                     warnings.warn(f"Primary search_scoring='{self.search_scoring}' not in validation_metrics.", UserWarning)


        allowed_sgd_params = set(SGDClassifier().get_params().keys())
        extra_sgd_params = {
            k: v for k, v in kwargs.items()
            if k in allowed_sgd_params and k not in self.model_params
        }
        self.model_params.update(extra_sgd_params)

        # print(f"SGDLinearModel Initialized with base params: {self.model_params}")
        # print(f"Imputer Strategy: {self._imputer_strategy}, Scaler Type: {self._scaler_type}")
        # print(f"K-fold CV / Search CV: n_splits={self.n_splits}, shuffle={self.shuffle_kfold}")
        # print(f"Validation metrics to report: {self.validation_metrics}")
        # print(f"Calibrate probabilities for hinge/squared_hinge loss: {self.calibrate_probabilities}")


    def _prepare_data_for_model(
        self, X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        label_col: Optional[str] = None,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Internal helper to preprocess data. (Largely identical to DecisionTreeModel._prepare_data_for_model)
        Uses self._scaler_type to choose between StandardScaler and MinMaxScaler.
        """
        X_features_np = None
        y_aligned = None
        current_feature_names = None

        # --- Stage 1: Determine input type, reshape, get initial features/labels ---
        if isinstance(X, pd.DataFrame):
            if is_training:
                if self.input_type is None: self.input_type = 'dataframe'
                elif self.input_type != 'dataframe': raise RuntimeError("Model trained on different input type.")
                self.sequence_length = 1
                if label_col is None or label_col not in X.columns:
                    raise ValueError(f"Label col '{label_col}' not found.")
                self.label_col = label_col
                original_names = X.columns.drop(label_col).tolist()
                if not original_names: raise ValueError("No feature columns found.")
                self.original_feature_names_ = original_names
                if self.n_original_features is None: self.n_original_features = len(original_names)
                elif self.n_original_features != len(original_names): raise ValueError("Feature count mismatch.")
                if self.processed_feature_names_ is None: self.processed_feature_names_ = original_names
                elif self.processed_feature_names_ != original_names: raise ValueError("Feature names mismatch.")
                current_feature_names = self.processed_feature_names_
                X_features_np = X[current_feature_names].to_numpy()
                y_aligned = X[self.label_col].to_numpy()
                if X_features_np.shape[0] == 0: raise ValueError("No data rows in DataFrame.")
            else: # Detection/Scoring for DataFrame
                if self.input_type != 'dataframe' or self.scaler is None or self.imputer is None or \
                   self.original_feature_names_ is None or self.n_original_features is None:
                    raise RuntimeError("Model not trained on DataFrame or components missing.")
                missing_cols = set(self.original_feature_names_) - set(X.columns)
                if missing_cols: raise ValueError(f"Detection DataFrame missing columns: {missing_cols}")
                current_feature_names = self.original_feature_names_
                X_features_np = X[current_feature_names].to_numpy()
                if X_features_np.shape[0] == 0:
                    warnings.warn("No data rows for DataFrame detection/scoring.", RuntimeWarning)
                    return np.empty((0, self.n_original_features)), None, current_feature_names

        elif isinstance(X, np.ndarray):
            if X.ndim != 3: raise ValueError(f"NumPy X must be 3D, got {X.ndim}D.")
            n_samples, seq_len, n_feat = X.shape
            if is_training:
                if self.input_type is None: self.input_type = 'numpy'
                elif self.input_type != 'numpy': raise RuntimeError("Model trained on different input type.")
                if self.sequence_length is None: self.sequence_length = seq_len
                elif self.sequence_length != seq_len: raise ValueError("Sequence length mismatch.")
                if self.n_original_features is None: self.n_original_features = n_feat
                elif self.n_original_features != n_feat: raise ValueError("Feature count mismatch.")
                if self.original_feature_names_ is None:
                    self.original_feature_names_ = [f"orig_feat_{i}" for i in range(self.n_original_features)]
                elif len(self.original_feature_names_) != self.n_original_features:
                     raise ValueError("Provided original_feature_names count mismatch.")
                if y is None: raise ValueError("'y' required for NumPy training.")
                if not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != n_samples:
                    raise ValueError("Invalid 'y' for NumPy training.")
                y_aligned = y
                n_flattened_features = seq_len * n_feat
                X_features_np = X.reshape(n_samples, n_flattened_features)
                if self.processed_feature_names_ is None:
                    self.processed_feature_names_ = [f"{orig_name}_step_{j}" for j in range(seq_len) for orig_name in self.original_feature_names_]
                    if len(self.processed_feature_names_) != n_flattened_features:
                        self.processed_feature_names_ = [f"flat_feature_{k}" for k in range(n_flattened_features)]
                elif len(self.processed_feature_names_) != n_flattened_features:
                    raise ValueError("Flattened feature name count mismatch.")
                current_feature_names = self.processed_feature_names_
            else: # Detection/Scoring for NumPy
                if self.input_type != 'numpy' or self.scaler is None or self.imputer is None or \
                   self.processed_feature_names_ is None or self.original_feature_names_ is None or \
                   self.n_original_features is None or self.sequence_length is None:
                    raise RuntimeError("Model not trained on NumPy or components missing.")
                if seq_len != self.sequence_length: raise ValueError(f"Input seq len mismatch.")
                if n_feat != self.n_original_features: raise ValueError(f"Input feature count mismatch.")
                current_feature_names = self.processed_feature_names_
                n_flattened_features = len(current_feature_names)
                if n_samples == 0:
                    warnings.warn("No samples for NumPy detection/scoring.", RuntimeWarning)
                    return np.empty((0, n_flattened_features)), None, current_feature_names
                X_features_np = X.reshape(n_samples, n_flattened_features)
        else:
            raise TypeError("Input 'X' must be pandas DataFrame or 3D NumPy array.")

        # --- Stage 2: Scaling ---
        if is_training:
            if self._scaler_type == 'standard':
                self.scaler = StandardScaler()
            else: # minmax
                self.scaler = MinMaxScaler()
            X_processed_scaled = self.scaler.fit_transform(X_features_np)
        else:
            if self.scaler is None or not hasattr(self.scaler, 'scale_') and not hasattr(self.scaler, 'mean_'):
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
            
        if np.isnan(X_processed_imputed).any():
            warnings.warn("NaN values detected *after* imputation. Applying nan_to_num.", RuntimeWarning)
            X_processed_imputed = np.nan_to_num(X_processed_imputed, nan=0.0) 

        if current_feature_names is None: raise RuntimeError("Internal: current_feature_names not set.")
        return X_processed_imputed, y_aligned, current_feature_names

    def run(self, X: Union[pd.DataFrame, np.ndarray],
            y: Optional[np.ndarray] = None,
            label_col: str = 'label',
            original_feature_names: Optional[List[str]] = None):
        """
        Prepares data. If auto_tune, performs RandomizedSearchCV.
        Otherwise, performs k-fold CV for performance estimation.
        Then, trains a final SGDClassifier (possibly calibrated) on the entire dataset.
        """
        # print(f"Running training for SGDLinearModel (Input: {'DataFrame' if isinstance(X, pd.DataFrame) else 'NumPy'})...")
        # if self.auto_tune: print(f"Hyperparameter auto-tuning ENABLED.")
        # else: print(f"Hyperparameter auto-tuning DISABLED. Using fixed parameters: {self.model_params}")

        if isinstance(X, np.ndarray):
            if original_feature_names is None: raise ValueError("`original_feature_names` required for NumPy `X`.")
            if X.shape[2] != len(original_feature_names): raise ValueError(f"NumPy feature dim mismatch.")
            self.original_feature_names_ = original_feature_names
        elif original_feature_names is not None:
            warnings.warn("`original_feature_names` ignored for DataFrame input.", UserWarning)

        X_processed_full, y_aligned_full, _ = self._prepare_data_for_model(
            X, y=y, label_col=label_col, is_training=True
        )

        if y_aligned_full is None: raise RuntimeError("No labels for training.")
        if X_processed_full.shape[0] != len(y_aligned_full): raise RuntimeError("Data samples and labels mismatch.")
        if X_processed_full.shape[0] < self.n_splits:
            warnings.warn(f"Samples < n_splits. CV/Search might fail.", RuntimeWarning)

        current_model_params = self.model_params.copy()
        self._is_calibrated = False # Reset calibration flag

        if self.auto_tune:
            # print(f"\nStarting RandomizedSearchCV for SGD hyperparameter tuning...")
            # (RandomizedSearchCV setup similar to other models, using SGDClassifier)
            base_estimator_params = {
                'random_state': current_model_params.get('random_state'),
                'class_weight': current_model_params.get('class_weight'),
                'max_iter': current_model_params.get('max_iter', 1000),
                'tol': current_model_params.get('tol', 1e-3)
            }
            for key_to_tune in self.param_dist.keys():
                if key_to_tune in base_estimator_params: del base_estimator_params[key_to_tune]
            
            base_estimator_params = {k: v for k, v in base_estimator_params.items() if v is not None}
            temp_model_for_search = SGDClassifier(**base_estimator_params)

            sklearn_scoring_map = {'accuracy': 'accuracy', 'f1': 'f1_macro', 'roc_auc': 'roc_auc'}
            scoring_dict_for_search = {
                m: sklearn_scoring_map[m] for m in self.validation_metrics if m in sklearn_scoring_map
            }
            # Add other direct sklearn scorers if present in self.validation_metrics
            for m in self.validation_metrics:
                if m not in scoring_dict_for_search and m in ['f1_weighted', 'f1_micro', 'roc_auc_ovr', 'roc_auc_ovo']:
                    scoring_dict_for_search[m] = m
                elif m not in scoring_dict_for_search:
                     warnings.warn(f"Metric '{m}' SKIPPED for RandomizedSearchCV.", UserWarning)

            if not scoring_dict_for_search: raise ValueError("No scikit-learn compatible metrics for RandomizedSearchCV.")
            
            refit_metric_key = self.search_scoring
            if refit_metric_key not in scoring_dict_for_search: # Fallback if mapping failed or was complex
                if scoring_dict_for_search : refit_metric_key = list(scoring_dict_for_search.keys())[0]
                else: raise ValueError("Cannot determine refit metric for RandomizedSearchCV.")
                warnings.warn(f"Using '{refit_metric_key}' as refit metric for RandomizedSearchCV.", UserWarning)

            search_cv_strategy = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle_kfold, random_state=current_model_params.get('random_state'))
            random_search = RandomizedSearchCV(
                estimator=temp_model_for_search, param_distributions=self.param_dist,
                n_iter=self.search_n_iter, scoring=scoring_dict_for_search, refit=refit_metric_key,
                cv=search_cv_strategy, random_state=current_model_params.get('random_state'),
                n_jobs=-1, error_score=np.nan # n_jobs for RandomizedSearchCV itself
            )
            try:
                random_search.fit(X_processed_full, y_aligned_full)
                self.tuned_best_params_ = random_search.best_params_
                # print(f"  Best parameters from RandomizedSearchCV: {self.tuned_best_params_}")
                current_model_params.update(self.tuned_best_params_)
                # (Populate validation_scores_ from random_search.cv_results_ - similar to other models)
                self.validation_scores_ = {}
                results_df = pd.DataFrame(random_search.cv_results_)
                best_idx = random_search.best_index_
                for metric_key_user in scoring_dict_for_search.keys():
                    scorer_name_sklearn = scoring_dict_for_search[metric_key_user]
                    mean_score_col = f"mean_test_{scorer_name_sklearn}"
                    std_score_col = f"std_test_{scorer_name_sklearn}"
                    if mean_score_col in results_df.columns:
                        avg_score = results_df.loc[best_idx, mean_score_col]
                        std_score = results_df.loc[best_idx, std_score_col] if std_score_col in results_df.columns else np.nan
                        self.validation_scores_[metric_key_user] = avg_score
                        # print(f"  Avg {metric_key_user} (as {scorer_name_sklearn}): {avg_score:.4f} (Std: {std_score:.4f})")

            except Exception as e:
                warnings.warn(f"RandomizedSearchCV failed: {e}. Using base params.", RuntimeWarning)
                traceback.print_exc()
                self.validation_scores_ = {metric: np.nan for metric in self.validation_metrics}
                current_model_params = self.model_params.copy()
                self.tuned_best_params_ = None
        else: # K-Fold CV without auto-tuning
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle_kfold, random_state=current_model_params.get('random_state'))
            fold_scores: Dict[str, List[float]] = {metric: [] for metric in self.validation_metrics}
            # print(f"\nStarting {self.n_splits}-fold CV with fixed SGD parameters: {current_model_params}")
            for fold_idx, (train_index, val_index) in enumerate(skf.split(X_processed_full, y_aligned_full)):
                X_train_fold, X_val_fold = X_processed_full[train_index], X_processed_full[val_index]
                y_train_fold, y_val_fold = y_aligned_full[train_index], y_aligned_full[val_index]

                temp_model_fold = SGDClassifier(**current_model_params)
                try:
                    temp_model_fold.fit(X_train_fold, y_train_fold)
                except Exception as e:
                    warnings.warn(f"SGD fitting failed for fold {fold_idx + 1}: {e}. Skipping.", RuntimeWarning)
                    for metric_name in self.validation_metrics: fold_scores[metric_name].append(np.nan)
                    continue

                y_pred_val_fold = temp_model_fold.predict(X_val_fold)
                for metric_name in self.validation_metrics:
                    score = np.nan
                    try:
                        if metric_name == 'accuracy': score = accuracy_score(y_val_fold, y_pred_val_fold)
                        elif metric_name == 'f1': score = f1_score(y_val_fold, y_pred_val_fold, average='macro', zero_division=0)
                        elif metric_name == 'roc_auc':
                            if hasattr(temp_model_fold, "predict_proba"):
                                y_proba_val_fold = temp_model_fold.predict_proba(X_val_fold)
                                if len(np.unique(y_val_fold)) > 1:
                                    pos_class_idx = np.where(temp_model_fold.classes_ == 1)[0]
                                    if len(pos_class_idx) > 0 and y_proba_val_fold.shape[1] > pos_class_idx[0]:
                                        score = roc_auc_score(y_val_fold, y_proba_val_fold[:, pos_class_idx[0]])
                            elif hasattr(temp_model_fold, "decision_function"): # For hinge loss etc.
                                y_decision_val_fold = temp_model_fold.decision_function(X_val_fold)
                                if len(np.unique(y_val_fold)) > 1:
                                     score = roc_auc_score(y_val_fold, y_decision_val_fold)
                        else: # Other sklearn scorers
                            try: scorer_fn = get_scorer(metric_name); score = scorer_fn(temp_model_fold, X_val_fold, y_val_fold)
                            except: pass
                        fold_scores[metric_name].append(score)
                    except Exception: fold_scores[metric_name].append(np.nan)
            # (Average fold scores - similar to other models)
            for metric_name in self.validation_metrics:
                valid_scores = [s for s in fold_scores[metric_name] if not np.isnan(s)]
                self.validation_scores_[metric_name] = np.mean(valid_scores) if valid_scores else np.nan
                # print(f"  Avg {metric_name}: {self.validation_scores_[metric_name]:.4f}")

        # --- Training the FINAL Model ---
        # print(f"Training final SGDLinearModel on {X_processed_full.shape[0]} samples...")
        # print(f"Using final parameters: {current_model_params}")
        
        final_sgd_model = SGDClassifier(**current_model_params)
        try:
            final_sgd_model.fit(X_processed_full, y_aligned_full)
        except Exception as e:
            self.model = None
            raise RuntimeError(f"Final SGDClassifier fitting failed: {e}") from e

        # Conditionally wrap with CalibratedClassifierCV for better probabilities if loss is hinge/squared_hinge
        loss_function = current_model_params.get('loss', 'hinge')
        if self.calibrate_probabilities and loss_function in ['hinge', 'squared_hinge']:
            # print(f"Calibrating probabilities for final model (loss: {loss_function})...")
            # Use a new StratifiedKFold for calibration's internal CV
            # Ensure n_splits for calibration is reasonable, e.g., min(5, self.n_splits, num_samples_per_class)
            # For simplicity, using fixed 3 or 5, or self.n_splits if large enough.
            calib_cv_splits = min(self.n_splits if self.n_splits >=2 else 3, 5) 
            # Check if we have enough samples for these splits, especially in minority class
            unique_classes, counts = np.unique(y_aligned_full, return_counts=True)
            if len(counts) > 1 and np.min(counts) < calib_cv_splits:
                 calib_cv_splits = max(2, int(np.min(counts))) # Adjust to available samples
                 warnings.warn(f"Reduced calibration CV splits to {calib_cv_splits} due to small class size.", UserWarning)

            if calib_cv_splits >= 2:
                try:
                    self.model = CalibratedClassifierCV(final_sgd_model, method='isotonic', cv=calib_cv_splits, n_jobs=current_model_params.get('n_jobs'))
                    self.model.fit(X_processed_full, y_aligned_full)
                    self._is_calibrated = True
                    # print("Probability calibration complete.")
                except Exception as cal_e:
                    warnings.warn(f"Calibration failed: {cal_e}. Using uncalibrated SGD model.", RuntimeWarning)
                    self.model = final_sgd_model # Fallback to uncalibrated
                    self._is_calibrated = False
            else:
                warnings.warn(f"Not enough samples/splits ({calib_cv_splits}) for calibration. Using uncalibrated SGD model.", UserWarning)
                self.model = final_sgd_model
                self._is_calibrated = False

        else:
            self.model = final_sgd_model # Use the SGDClassifier directly
            self._is_calibrated = False
            if loss_function not in ['log_loss', 'modified_huber']:
                warnings.warn(f"Loss function is '{loss_function}'. Probabilities from predict_proba might not be well-calibrated unless 'calibrate_probabilities' was True (for hinge) or loss is log_loss/modified_huber.", UserWarning)


        # print("Final model training complete.")
        # if hasattr(self.model, 'classes_'): print(f"Final model classes: {self.model.classes_}")
        self.model_params = current_model_params.copy() # Store effective params

    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns the probability of the positive class (anomaly)."""
        if self.model is None: raise RuntimeError("Model not trained.")
        
        n_input_samples = len(detection_data) if isinstance(detection_data, pd.DataFrame) else detection_data.shape[0]
        final_scores = np.full(n_input_samples, np.nan, dtype=float)
        if n_input_samples == 0: return final_scores

        try:
            X_processed_imputed, _, _ = self._prepare_data_for_model(detection_data, is_training=False, label_col=self.label_col)
            if X_processed_imputed.shape[0] == 0: return final_scores

            if not hasattr(self.model, "predict_proba"):
                warnings.warn("Model does not have predict_proba (e.g. SGD with hinge loss and no calibration). Returning decision_function or NaNs.", RuntimeWarning)
                if hasattr(self.model, "decision_function"):
                    dec_func = self.model.decision_function(X_processed_imputed)
                    anomaly_scores = 1.0 / (1.0 + np.exp(-dec_func)) # Basic sigmoid
                else:
                    anomaly_scores = np.full(X_processed_imputed.shape[0], np.nan)
            else:
                probabilities = self.model.predict_proba(X_processed_imputed)
                positive_class_index = np.where(self.model.classes_ == 1)[0]
                if len(positive_class_index) > 0:
                    anomaly_scores = probabilities[:, positive_class_index[0]]
                else: # Should not happen if classes are [0, 1]
                    warnings.warn("Positive class (1) not found. Using second column or NaNs.", RuntimeWarning)
                    anomaly_scores = probabilities[:, 1] if probabilities.shape[1] > 1 else np.full(probabilities.shape[0], np.nan)
            
            len_to_copy = min(len(anomaly_scores), n_input_samples)
            final_scores[:len_to_copy] = anomaly_scores[:len_to_copy]

        except Exception as e:
            warnings.warn(f"Anomaly score calculation failed: {e}. Returning NaNs.", RuntimeWarning)
        return final_scores


    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Detects anomalies (predicts class 1)."""
        if self.model is None: raise RuntimeError("Model not trained.")
        
        n_input_samples = len(detection_data) if isinstance(detection_data, pd.DataFrame) else detection_data.shape[0]
        final_anomalies = np.zeros(n_input_samples, dtype=bool)
        if n_input_samples == 0: return final_anomalies
        
        try:
            X_processed_imputed, _, _ = self._prepare_data_for_model(detection_data, is_training=False, label_col=self.label_col)
            if X_processed_imputed.shape[0] == 0: return final_anomalies

            predictions = self.model.predict(X_processed_imputed)
            anomalies = (predictions == 1)
            len_to_copy = min(len(anomalies), n_input_samples)
            final_anomalies[:len_to_copy] = anomalies[:len_to_copy]
        except Exception as e:
            warnings.warn(f"Anomaly detection failed: {e}. Returning False.", RuntimeWarning)
        return final_anomalies

    def predict_proba(self, X_xai: np.ndarray) -> np.ndarray:
        """Prediction function for XAI methods. Returns probabilities for all classes."""
        if self.model is None: raise RuntimeError("Model not trained for XAI.")
        if not isinstance(X_xai, np.ndarray): raise TypeError("X_xai must be NumPy array.")
        n_instances = X_xai.shape[0]
        if n_instances == 0:
            n_classes = len(self.model.classes_) if hasattr(self.model, 'classes_') and self.model.classes_ is not None else 2
            return np.empty((0, n_classes))
        X_to_process_2d = None
        try:
            # (Input type handling and reshaping for XAI as in previous models)
            if self.input_type == 'dataframe':
                if X_xai.ndim == 3:
                    if X_xai.shape[1] == 1 and X_xai.shape[2] == self.n_original_features:
                        X_to_process_2d = X_xai.reshape(n_instances, self.n_original_features)
                    else: raise ValueError("XAI 3D input shape mismatch for DF model.")
                elif X_xai.ndim == 2 and X_xai.shape[1] == self.n_original_features:
                    X_to_process_2d = X_xai
                else: raise ValueError("XAI input dim/shape mismatch for DF model.")
            elif self.input_type == 'numpy':
                if X_xai.ndim != 3 or X_xai.shape[1] != self.sequence_length or X_xai.shape[2] != self.n_original_features:
                    raise ValueError("XAI input dim/shape mismatch for NumPy model.")
                X_to_process_2d = X_xai.reshape(n_instances, self.sequence_length * self.n_original_features)
            else: raise RuntimeError(f"Unsupported input_type '{self.input_type}' for XAI.")

            X_scaled = self.scaler.transform(X_to_process_2d)
            X_imputed = self.imputer.transform(X_scaled)
            if np.isnan(X_imputed).any(): X_imputed = np.nan_to_num(X_imputed, nan=0.0)
            
            if not hasattr(self.model, "predict_proba"):
                 # This case should be rare if calibration is handled correctly or loss is prob-native
                warnings.warn("predict_proba called for XAI on SGD model without direct probability method. "
                              "Using decision_function with sigmoid or returning 0.5.", RuntimeWarning)
                if hasattr(self.model, "decision_function"):
                    dec_func = self.model.decision_function(X_imputed)
                    prob_pos = 1.0 / (1.0 + np.exp(-dec_func)) # Sigmoid
                    probabilities = np.vstack([1 - prob_pos, prob_pos]).T
                else: # Fallback if neither exists
                    n_classes = len(self.model.classes_) if hasattr(self.model, 'classes_') else 2
                    probabilities = np.full((X_imputed.shape[0], n_classes), 1.0 / n_classes)
            else:
                probabilities = self.model.predict_proba(X_imputed)

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"XAI prediction failed. Error: {e}") from e
        # (Shape validation for probabilities output - similar to other models)
        return probabilities

    def get_validation_scores(self) -> Dict[str, float]:
        """Returns the computed validation scores."""
        return self.validation_scores_

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Returns feature importances (coefficients) from the trained linear model.
        If CalibratedClassifierCV is used, importances are from its base estimator.
        """
        if self.model is None: return None
        
        actual_model_for_coef = self.model
        if self._is_calibrated and hasattr(self.model, 'calibrated_classifiers_'):
            # For CalibratedClassifierCV, coef_ are on the base estimators
            # We'll take the first one as representative (they should be similar if data is i.i.d.)
            if self.model.calibrated_classifiers_: # List of CalibratedClassifier
                # Each element has a base_estimator which is the SGDClassifier
                base_estimator = self.model.calibrated_classifiers_[0].base_estimator
                if hasattr(base_estimator, 'coef_'):
                    actual_model_for_coef = base_estimator
                else:
                    warnings.warn("Base estimator in CalibratedClassifierCV does not have 'coef_'.", UserWarning)
                    return None
            else: # Should not happen if fitted
                return None

        if hasattr(actual_model_for_coef, 'coef_'):
            # For binary classification, coef_ is usually shape (1, n_features)
            # For OvA multiclass, it's (n_classes, n_features)
            if actual_model_for_coef.coef_.ndim == 2 and actual_model_for_coef.coef_.shape[0] == 1:
                importances_raw = actual_model_for_coef.coef_[0]
            elif actual_model_for_coef.coef_.ndim == 1: # Can happen for binary case with some setups
                importances_raw = actual_model_for_coef.coef_
            else: # Multiclass or unexpected shape
                warnings.warn(f"Feature importances (coefficients) have shape {actual_model_for_coef.coef_.shape}. "
                              "Interpretation might be complex for multiclass. Returning average absolute coefficient per feature.")
                importances_raw = np.mean(np.abs(actual_model_for_coef.coef_), axis=0)

            importances = np.abs(importances_raw) # Magnitude

            if self.processed_feature_names_ is None:
                return {f"feature_{i}": imp for i, imp in enumerate(importances)}
            if len(importances) != len(self.processed_feature_names_):
                warnings.warn(f"Mismatch in importances ({len(importances)}) and feature names ({len(self.processed_feature_names_)}).")
                max_len = min(len(importances), len(self.processed_feature_names_))
                return {self.processed_feature_names_[i]: importances[i] for i in range(max_len)}
            return dict(zip(self.processed_feature_names_, importances))
        else:
            # print("Model does not have 'coef_' attribute for feature importances.")
            return None