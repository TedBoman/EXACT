import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ML_models import model_interface
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model # Use tensorflow.keras
from tensorflow.keras.layers import Input, Dense # Use tensorflow.keras
from tensorflow.keras.optimizers import Adam # Import Adam optimizer
from typing import Optional, Union, List, Any
import warnings

class SVMModel(model_interface.ModelInterface):
    """
    Anomaly detection using an Autoencoder + OneClassSVM.
    Operates on 2D data (samples, features). Refactored for correctness & XAI.
    """
    sequence_length = 1 # Indicates it processes samples individually

    def __init__(self, **kwargs):
        """
        Initializes the Autoencoder + OneClassSVM model state and stores configuration.

        Expected kwargs (examples):
            encoding_dim (int): Dimensionality of the autoencoder's latent space (default: 10).
            ae_activation (str): Activation function for AE hidden layer (default: 'relu').
            ae_output_activation (str): Activation for AE output layer (default: 'linear').
            optimizer (str): Optimizer name for AE training (default: 'adam').
            learning_rate (float): Learning rate for AE optimizer (default: 0.001).
            loss (str): Loss function for AE training (default: 'mse').
            svm_nu (float): An upper bound on the fraction of training errors and a lower bound
                            of the fraction of support vectors (SVM parameter, default: 0.1).
            svm_kernel (str): Kernel type for OneClassSVM (default: 'rbf').
            svm_gamma (str or float): Kernel coefficient for 'rbf', 'poly', 'sigmoid' (default: 'scale').
            # Training specific params also stored here
            epochs (int): Default number of AE training epochs (default: 10).
            batch_size (int): Default AE training batch size (default: 32).
        """
        self.scaler = StandardScaler() # Using StandardScaler
        self.encoder: Optional[Model] = None
        self.autoencoder: Optional[Model] = None
        self.svm_model: Optional[OneClassSVM] = None # Initialize later
        self.threshold: Optional[float] = None
        self.n_features: Optional[int] = None

        # --- Store configuration from kwargs ---
        self.config = {
            'encoding_dim': kwargs.get('encoding_dim', 10),
            'ae_activation': kwargs.get('ae_activation', 'relu'),
            'ae_output_activation': kwargs.get('ae_output_activation', 'linear'), # Linear for StandardScaler
            'optimizer_name': kwargs.get('optimizer', 'adam'),
            'learning_rate': kwargs.get('learning_rate', 0.001),
            'loss': kwargs.get('loss', 'mse'),
            'svm_nu': kwargs.get('svm_nu', 0.1),
            'svm_kernel': kwargs.get('svm_kernel', 'rbf'),
            'svm_gamma': kwargs.get('svm_gamma', 'scale'),
            'epochs': kwargs.get('epochs', 10),
            'batch_size': kwargs.get('batch_size', 32)
        }
        # ---

        # --- Initialize SVM with parameters from config ---
        self.svm_model = OneClassSVM(
            kernel=self.config['svm_kernel'],
            gamma=self.config['svm_gamma'],
            nu=self.config['svm_nu']
        )
        # ---

        # print(f"SVMModel Initialized with config: {self.config}")

    def __build_autoencoder(self, input_dim):
        """ Builds AE using parameters stored in self.config """
        encoding_dim = self.config['encoding_dim']
        activation = self.config['ae_activation']
        output_activation = self.config['ae_output_activation']
        optimizer_name = self.config['optimizer_name']
        learning_rate = self.config['learning_rate']
        loss_function = self.config['loss']

        # print(f"Building Autoencoder: input_dim={input_dim}, encoding_dim={encoding_dim}, activation={activation}, output_activation={output_activation}")

        input_layer = Input(shape=(input_dim,), name='input_layer')
        encoded = Dense(encoding_dim, activation=activation, name='encoder_output')(input_layer)
        decoded = Dense(input_dim, activation=output_activation, name='decoder_output')(encoded) # Use config

        autoencoder = Model(inputs=input_layer, outputs=decoded, name='autoencoder')
        encoder = Model(inputs=input_layer, outputs=encoded, name='encoder')

        # Configure optimizer
        if isinstance(optimizer_name, str):
            if optimizer_name.lower() == 'adam':
                optimizer = Adam(learning_rate=learning_rate)
            else:
                warnings.warn(f"Optimizer '{optimizer_name}' not fully configured, defaulting to Adam with LR={learning_rate}.", UserWarning)
                optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = optimizer_name

        autoencoder.compile(optimizer=optimizer, loss=loss_function) # Use config
        # print("Autoencoder Architecture:")
        autoencoder.summary(print_fn=print)
        return autoencoder, encoder

    def run(self, df: pd.DataFrame): # Remove epochs, batch_size from signature
        """ Trains the Autoencoder and then the OneClassSVM using parameters from __init__. """
        if not isinstance(df, pd.DataFrame): raise TypeError("Input 'df' must be a pandas DataFrame.")
        if df.empty: raise ValueError("Input DataFrame for training is empty.")

        # --- Use parameters from self.config ---
        epochs = self.config['epochs']
        batch_size = self.config['batch_size']
        # ---

        # print(f"Running SVMModel training on data shape: {df.shape}, AE epochs={epochs}, batch_size={batch_size}")
        self.n_features = df.shape[1]
        if self.n_features == 0: raise ValueError("Input DataFrame has no feature columns.")

        # --- Scaling ---
        # print("Fitting and transforming scaler...")
        X_train_scaled = self.scaler.fit_transform(df)
        X_train_scaled = X_train_scaled.astype(np.float32)

        # --- Train Autoencoder (builds model internally using config) ---
        self.autoencoder, self.encoder = self.__build_autoencoder(self.n_features)
        # print(f"Fitting Autoencoder for {epochs} epochs...")
        self.autoencoder.fit(X_train_scaled, X_train_scaled,
                             epochs=epochs, batch_size=batch_size, # Use config
                             validation_split=0.1, shuffle=True, verbose=1)
        # print("Autoencoder fitting complete.")

        # --- Get Encoded Representation & Train SVM ---
        # print("Encoding training data...")
        train_encoded_data = self.encoder.predict(X_train_scaled)
        # print(f"Encoded training data shape: {train_encoded_data.shape}")

        # SVM Model was already initialized with params in __init__
        if self.svm_model is None:
            raise RuntimeError("SVM model was not initialized correctly.")

        # print(f"Fitting OneClassSVM with parameters: {self.svm_model.get_params()}")
        self.svm_model.fit(train_encoded_data)
        # print("OneClassSVM fitting complete.")

        # --- Set Threshold ---
        # print("Calculating anomaly threshold...")
        decision_values_train = self.svm_model.decision_function(train_encoded_data)
        # Use svm_nu directly from config as it's the expected error rate
        self.threshold = np.percentile(decision_values_train, 100 * self.config['svm_nu'])
        # print(f"Anomaly threshold set to: {self.threshold:.6f} (based on nu={self.config['svm_nu']})")
        # print("--- SVMModel Training Finished ---")

    def _preprocess_and_encode(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ Internal helper: Scales (using fitted scaler) and encodes input data. """
        if self.scaler is None or self.encoder is None:
            raise RuntimeError("Model is not trained (scaler/encoder missing). Call run() first.")

        if isinstance(data, pd.DataFrame):
            if data.shape[1] != self.n_features: raise ValueError(f"Input data has {data.shape[1]} feats, expected {self.n_features}.")
            input_np = data.values
        elif isinstance(data, np.ndarray):
            if data.ndim == 1: 
                data = data.reshape(1, -1)
            elif data.ndim == 3 and data.shape[1] == 1: # Add this condition
                # Reshape from (samples, 1, features) to (samples, features)
                data = data.reshape(data.shape[0], data.shape[2]) 
            
            if data.ndim != 2: # This check will now pass for the (5000, 1, 29) case after reshaping
                raise ValueError(f"NumPy input must be 2D (samples, features), got {data.ndim}D. Original shape might have been 3D and not squeezable.")
            if data.shape[1] != self.n_features: 
                raise ValueError(f"NumPy input has {data.shape[1]} feats, expected {self.n_features}.")
            input_np = data
        else: raise TypeError("Input must be a pandas DataFrame or a 2D NumPy array.")

        ## print(f"Preprocessing {input_np.shape[0]} samples...")
        data_scaled = self.scaler.transform(input_np) # Use TRANSFORM only
        data_scaled = data_scaled.astype(np.float32)

        ## print(f"Encoding {data_scaled.shape[0]} scaled samples...")
        encoded_data = self.encoder.predict(data_scaled)
        ## print(f"Encoded data shape: {encoded_data.shape}")
        return encoded_data

    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ Detects anomalies based on threshold. Expects 2D input. """
        if self.threshold is None: raise RuntimeError("Threshold not set. Call run() first.")
        scores = self.get_anomaly_score(detection_data)
        anomalies = scores < self.threshold
        # print(f"Detected {np.sum(anomalies)} anomalies using threshold {self.threshold:.6f}.")
        return anomalies # Returns 1D boolean array
    
    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculates the anomaly score (SVM decision function) for input data.
        Operates on the encoded representation. Lower scores indicate higher anomaly likelihood.
        Expects 2D input (samples, features).

        Args:
            detection_data (Union[pd.DataFrame, np.ndarray]): Input data (samples, features).

        Returns:
            np.ndarray: The decision function scores (1D array, shape: (n_samples,)).
        """
        # Reuse the internal preprocessing and encoding helper
        # This ensures scaling and encoding are done consistently
        # print("Calculating anomaly scores via get_anomaly_score...")
        # Ensure model is trained before calling helper
        if self.scaler is None or self.encoder is None or self.svm_model is None:
             raise RuntimeError("Model components (scaler/encoder/svm) not available. Call run() first.")

        encoded_data = self._preprocess_and_encode(detection_data) # Gets 2D encoded data

        # Get SVM score on encoded data - lower means more anomalous
        # Ensure svm_model is fitted (checked implicitly by _preprocess_and_encode checking encoder)
        if not hasattr(self.svm_model, "decision_function"):
             raise RuntimeError("Internal SVM model is not fitted or invalid.")

        scores = self.svm_model.decision_function(encoded_data)
        return scores # Return the raw scores (1D array)
    
    def predict_proba(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts the pseudo-probability of each sample being an anomaly based on the
        OneClassSVM decision function score. Lower scores result in higher anomaly probability.
        This probability is derived by applying a sigmoid function to the scaled
        difference between the threshold and the score.

        Accepts 2D DataFrame/NumPy (samples, features) or 1D NumPy (single sample).

        Returns:
            np.ndarray: Array of shape (n_samples, 2) where:
                        Column 0: Probability of being Normal (1 - P(Anomaly))
                        Column 1: Probability of being Anomaly (P(Anomaly))
                        Returns empty array shape (0, 2) if no samples processed.
        """
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Call run() first to train and set threshold.")
        if not np.isfinite(self.threshold):
             warnings.warn(f"Threshold is not finite ({self.threshold}). Probability calculation might be unreliable.", RuntimeWarning)
             # Handle non-finite threshold for scaling gracefully
             # Get scores first to determine output shape
             scores = self.get_anomaly_score(detection_data)
             n_scores = len(scores)
             if n_scores == 0: return np.empty((0, 2))

             # If threshold is -inf, all scores are > threshold => P(Anomaly) = 0
             if self.threshold == -np.inf:
                 return np.hstack([np.ones((n_scores, 1)), np.zeros((n_scores, 1))])
             # If threshold is +inf, all scores are < threshold => P(Anomaly) = 1
             if self.threshold == np.inf:
                  return np.hstack([np.zeros((n_scores, 1)), np.ones((n_scores, 1))])
             # Handle NaN case - return uncertain probabilities? Or 0.5/0.5? Let's use 0.5/0.5
             return np.full((n_scores, 2), 0.5)


        # Get the SVM decision function scores (lower = more anomalous)
        scores = self.get_anomaly_score(detection_data)

        if scores.size == 0:
            return np.empty((0, 2)) # Return empty array with correct number of columns

        # Define the sigmoid function
        def sigmoid(x):
            # Clip x to avoid overflow/underflow in exp
            x_clipped = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x_clipped))

        # --- Calculate scale for sigmoid ---
        # Use the configured scale factor relative to the threshold's magnitude.
        proba_svm_scale_factor = self.config.get('proba_svm_scale_factor', 4.0) # Default to 4.0
        # Use max with epsilon to prevent division by zero and handle threshold=0
        scale = max(abs(self.threshold) / proba_svm_scale_factor, 1e-9)

        # Calculate the scaled difference: -(score - threshold) / scale
        # This ensures scores below threshold give positive input to sigmoid
        scaled_neg_diff = -(scores - self.threshold) / scale

        # Apply sigmoid to get the probability of anomaly (Class 1)
        prob_anomaly = sigmoid(scaled_neg_diff)

        # Probability of normal (Class 0) is 1 - P(Anomaly)
        prob_normal = 1.0 - prob_anomaly

        # Stack probabilities into the desired (n_samples, 2) shape
        probabilities = np.vstack([prob_normal, prob_anomaly]).T

        ## print(f"Calculated anomaly probabilities for {probabilities.shape[0]} samples.") # Optional print
        return probabilities
