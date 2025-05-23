import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler 
from ML_models import model_interface 
from typing import Union, List, Optional

class IsolationForestModel(model_interface.ModelInterface):
    """ Isolation Forest for anomaly detection on 2D data. """
    sequence_length = 1 # Indicate non-sequential nature

    def __init__(self, **kwargs):
        """
        Initializes Isolation Forest using parameters from kwargs.

        Expected kwargs (examples):
            n_estimators (int): Number of base estimators (trees) in the ensemble (default: 100).
            contamination (float or 'auto'): Expected proportion of outliers (default: 'auto').
            max_samples (int or float): Number/fraction of samples to draw for each tree (default: 'auto').
            max_features (int or float): Number/fraction of features to draw for each tree (default: 1.0).
            bootstrap (bool): Whether samples are drawn with replacement (default: False).
            random_state (int): Controls the randomness of the estimator (default: None).
            n_jobs (int): Number of jobs to run in parallel (default: -1).
            ... other IsolationForest parameters ...
        """
        self.scaler: Optional[StandardScaler] = StandardScaler() # Always use scaler
        self.n_features: Optional[int] = None
        self.feature_names: Optional[List[str]] = None

        # --- Extract parameters from kwargs with defaults ---
        n_estimators = kwargs.get('n_estimators', 100)
        contamination = kwargs.get('contamination', 'auto')
        max_samples = kwargs.get('max_samples', 'auto')
        max_features = kwargs.get('max_features', 1.0)
        bootstrap = kwargs.get('bootstrap', False) # Note: sklearn default might change
        random_state = kwargs.get('random_state', None)
        n_jobs = kwargs.get('n_jobs', -1)

        # Store all parameters passed to the underlying model
        self.model_params = {
            'n_estimators': n_estimators,
            'contamination': contamination,
            'max_samples': max_samples,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        # Add any other kwargs intended for IsolationForest
        allowed_if_params = set(IsolationForest().get_params().keys())
        extra_if_params = {k: v for k, v in kwargs.items() if k in allowed_if_params and k not in self.model_params}
        self.model_params.update(extra_if_params)

        # Instantiate the model using the collected parameters
        self.model: Optional[IsolationForest] = IsolationForest(**self.model_params)

        # print(f"IsolationForestModel Initialized with effective params: {self.model_params}")
     
    def run(self, df: pd.DataFrame):
        """ Trains the Isolation Forest model. """
        if not isinstance(df, pd.DataFrame): raise TypeError("Input 'df' must be a pandas DataFrame.")
        if df.empty: raise ValueError("Input DataFrame 'df' is empty.")

        # print(f"Running IsolationForest training on data shape: {df.shape}")
        self.n_features = df.shape[1]
        self.feature_names = df.columns.tolist() # Store feature names
        if self.n_features == 0: raise ValueError("Input DataFrame has no feature columns.")

        X_train = df.values # Use NumPy array for fitting

        # --- Scaling ---
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        # print("Scaler fitted.")
        # --- End Scaling ---

        # print(f"Fitting IsolationForest model...")
        self.model.fit(X_train)
        # print("Model fitting complete.")

    def _prepare_data_for_predict(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Handles 2D/3D input, selects last step if 3D, applies scaling (if used),
        and returns 2D NumPy array ready for prediction.
        """
        if self.n_features is None: raise RuntimeError("Model not trained (n_features not set).")
        if self.model is None: raise RuntimeError("Model not trained.")

        data_2d = None
        input_is_3d = False

        if isinstance(detection_data, pd.DataFrame):
            if detection_data.shape[1] != self.n_features: raise ValueError(f"DataFrame input features {detection_data.shape[1]} != expected {self.n_features}.")
            # Ensure column order matches training if scaler was used and fitted on DF
            if self.feature_names and list(detection_data.columns) != self.feature_names:
                warnings.warn("Input DataFrame columns may differ from training order. Reordering.", UserWarning)
                detection_data = detection_data[self.feature_names]
            data_2d = detection_data.values
        elif isinstance(detection_data, np.ndarray):
            if detection_data.ndim == 3:
                input_is_3d = True
                n_samples, seq_len, n_feat = detection_data.shape
                if n_feat != self.n_features: raise ValueError(f"3D NumPy input features {n_feat} != expected {self.n_features}.")
                # print("IsolationForestModel: Received 3D NumPy, selecting last time step.")
                data_2d = detection_data[:, -1, :] # Extract last step -> (samples, features)
            elif detection_data.ndim == 2:
                if detection_data.shape[1] != self.n_features: raise ValueError(f"2D NumPy input features {detection_data.shape[1]} != expected {self.n_features}.")
                data_2d = detection_data
            elif detection_data.ndim == 1: # Single sample
                if len(detection_data) != self.n_features: raise ValueError(f"1D NumPy input length {len(detection_data)} != expected {self.n_features}.")
                data_2d = detection_data.reshape(1, -1)
            else: raise ValueError(f"Unsupported NumPy input dimension: {detection_data.ndim}")
        else: raise TypeError("Input must be DataFrame or NumPy array.")

        if data_2d is None or data_2d.size == 0:
            warnings.warn("No processable data found after handling input.", RuntimeWarning)
            return np.empty((0, self.n_features)) # Return empty 2D

        # Apply scaling if scaler was fitted
        if self.scaler:
            data_2d = self.scaler.transform(data_2d)

        return data_2d


    def predict_proba(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculates anomaly scores using the Isolation Forest decision_function.
        Lower scores are more anomalous. Handles 2D/3D input.

        Returns:
            np.ndarray: 1D array of anomaly scores, length matches input samples.
        """
        # print("Calculating anomaly scores (Isolation Forest decision_function)...")
        processed_data_2d = self._prepare_data_for_predict(detection_data)

        if processed_data_2d.size == 0: return np.array([])

        scores = self.model.decision_function(processed_data_2d) # Lower = more anomalous
        # The score is calculated for each input sample (row in processed_data_2d)
        # print(f"Calculated {len(scores)} scores.")
        return scores # Returns 1D array


    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Detects anomalies using Isolation Forest predict method.
        Handles 2D/3D input.

        Returns:
            np.ndarray: 1D boolean array (True=Anomaly), length matches input samples.
        """
        # print("Detecting anomalies (Isolation Forest predict)...")
        processed_data_2d = self._prepare_data_for_predict(detection_data)

        if processed_data_2d.size == 0: return np.array([], dtype=bool)

        # Predict returns +1 for inliers, -1 for outliers
        predictions = self.model.predict(processed_data_2d)
        anomalies = (predictions == -1) # Convert -1 (outlier) to True
        # print(f"Detected {np.sum(anomalies)} anomalies out of {len(anomalies)} samples.")
        return anomalies # Returns 1D boolean array

    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        scores = self.predict_proba(detection_data)
        # Convert scores (lower=anomaly) to probabilities P(anomaly) ~ [0,1]
        # Needs careful scaling based on score distribution / offsets
        offset = getattr(self.model, 'offset_', 0) # Internal threshold related to contamination
        # Simple sigmoid approach (needs tuning!)
        prob_anomaly = 1 / (1 + np.exp(-(offset - scores) / np.std(scores))) # Example scaling
        prob_normal = 1.0 - prob_anomaly
        return np.vstack([prob_normal, prob_anomaly]).T