import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from ML_models import model_interface
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam 
from typing import Optional, Union, List 
import warnings 

class LSTMModel(model_interface.ModelInterface):

    def __init__(self, **kwargs):
        """
        Initializes the LSTM Autoencoder model state and stores configuration.

        Expected kwargs (examples):
            units (int): Number of units in LSTM layers (default: 64).
            activation (str): Activation function for LSTM layers (default: 'relu').
            optimizer (str or tf.keras.optimizers.Optimizer): Optimizer to use (default: 'adam').
            learning_rate (float): Learning rate for the optimizer (default: 0.001).
            loss (str or tf.keras.losses.Loss): Loss function (default: 'mse').
            # Training specific params also stored here
            epochs (int): Default number of training epochs (default: 10).
            batch_size (int): Default training batch size (default: 256).
            time_steps (int): Default sequence length (lookback window) (default: 10).
        """
        self.model: Optional[Model] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.threshold: Optional[float] = None
        self.sequence_length: Optional[int] = None # Will be set during run or from params

        # --- Store configuration from kwargs ---
        self.config = {
            'units': kwargs.get('units', 64),
            'activation': kwargs.get('activation', 'relu'),
            'optimizer_name': kwargs.get('optimizer', 'adam'), # Store name or object
            'learning_rate': kwargs.get('learning_rate', 0.001),
            'loss': kwargs.get('loss', 'mse'),
            'epochs': kwargs.get('epochs', 10), # Store training param
            'batch_size': kwargs.get('batch_size', 256), # Store training param
            'time_steps': kwargs.get('time_steps', 10) # Store sequence length param
        }
        # Set initial sequence length if provided
        self.sequence_length = self.config['time_steps']

        # print(f"LSTMModel Initialized with config: {self.config}")
        # --- End __init__ ---


    # --- run Method ---
    def run(self, df: pd.DataFrame):
        """
        Preprocesses data, builds, trains, and fits the LSTM autoencoder model
        using parameters stored during __init__.

        Args:
            df (pd.DataFrame): Input DataFrame containing features for training.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        # --- Use parameters from self.config ---
        time_steps = self.config['time_steps']
        epochs = self.config['epochs']
        batch_size = self.config['batch_size']
        units = self.config['units']
        activation = self.config['activation']
        optimizer_name = self.config['optimizer_name']
        learning_rate = self.config['learning_rate']
        loss_function = self.config['loss']
        dropout_rate = self.config.get('dropout', 0.0) # Get from stored config
        rec_dropout_rate = self.config.get('recurrent_dropout', 0.0) # Get from stored config
        # ---

        if time_steps <= 0:
            raise ValueError("Configured 'time_steps' must be positive.")
        self.sequence_length = time_steps # Ensure instance variable is set

        # print(f"Running LSTMModel training with time_steps={self.sequence_length}, epochs={epochs}, batch_size={batch_size}...")

        features = df.shape[1]
        if features == 0:
            raise ValueError("Input DataFrame has no columns (features).")

        # --- Define Keras Model using config ---
        inputs = Input(shape=(self.sequence_length, features))
        encoded = LSTM(units, activation=activation, return_sequences=False,
                    dropout=dropout_rate, recurrent_dropout=rec_dropout_rate)(inputs) # Add params
        decoded = RepeatVector(self.sequence_length)(encoded)
        decoded = LSTM(units, activation=activation, return_sequences=True,
                    dropout=dropout_rate, recurrent_dropout=rec_dropout_rate)(decoded) # Add params
        outputs = TimeDistributed(Dense(features))(decoded)

        autoencoder = Model(inputs, outputs)

        # Configure optimizer
        if isinstance(optimizer_name, str):
            if optimizer_name.lower() == 'adam':
                optimizer = Adam(learning_rate=learning_rate) # Use configured LR
            else:
                # Add other optimizers if needed, or default to Adam
                warnings.warn(f"Optimizer '{optimizer_name}' not fully configured, defaulting to Adam with LR={learning_rate}.", UserWarning)
                optimizer = Adam(learning_rate=learning_rate)
        else: # Assume it's an optimizer instance
            optimizer = optimizer_name

        autoencoder.compile(optimizer=optimizer, loss=loss_function) # Use config
        self.model = autoencoder
        # print("LSTM Autoencoder Model Compiled:")
        self.model.summary()
        # --- End Model Definition ---

        # --- Data Preprocessing ---
        self.scaler = MinMaxScaler()
        data_normalized = self.scaler.fit_transform(df)

        # print(f"Creating sequences with length {self.sequence_length}...")
        X = self.__create_sequences(data_normalized, self.sequence_length)
        if X.size == 0:
            raise ValueError(f"Data is too short ({len(df)} rows) to create sequences of length {self.sequence_length}.")
        # print(f"Created {X.shape[0]} sequences with shape {X.shape[1:]}")
        # --- End Data Preprocessing ---

        # --- Training ---
        train_size = int(len(X) * 0.8)
        if train_size == 0 and len(X) > 0: train_size = 1
        X_train = X[:train_size]
        X_test_threshold = X[train_size:]

        if X_train.size == 0:
            warnings.warn("Training split is empty, model cannot be trained.", RuntimeWarning)
            self.threshold = np.inf
            return

        # print(f"Fitting model on {X_train.shape[0]} training sequences...")
        self.model.fit(
            X_train, X_train,
            epochs=epochs,       # Use config
            batch_size=batch_size, # Use config
            validation_split=0.2,
            verbose=1,
            shuffle=True
        )
        # print("Model fitting complete.")
        # --- End Training ---

        # --- Threshold Calculation ---
        if X_test_threshold.size > 0:
            # print(f"Calculating threshold on {X_test_threshold.shape[0]} test sequences...")
            reconstructed = self.model.predict(X_test_threshold)
            reconstruction_error = np.mean(np.square(X_test_threshold - reconstructed), axis=(1, 2))
            self.threshold = np.percentile(reconstruction_error, 95)
            # print(f"Anomaly threshold set to: {self.threshold:.6f}")
        else:
            warnings.warn("Test split for threshold calculation is empty. Threshold may be unreliable.", RuntimeWarning)
            if X_train.size > 0:
                reconstructed_train = self.model.predict(X_train)
                reconstruction_error_train = np.mean(np.square(X_train - reconstructed_train), axis=(1, 2))
                self.threshold = np.percentile(reconstruction_error_train, 95)
                # print(f"Anomaly threshold set from training data: {self.threshold:.6f}")
            else:
                self.threshold = np.inf
                # print("Error: Cannot set threshold - no data available.")

    def __create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """ Helper method to create 3D windowed sequences from 2D data. """
        sequences = []
        n_samples_total = data.shape[0]
        if n_samples_total < sequence_length:
            # print(f"Warning in __create_sequences: Data length ({n_samples_total}) < sequence_length ({sequence_length}).")
            # Return empty array with correct feature dimension if possible
            n_features = data.shape[1] if data.ndim == 2 else 0
            return np.empty((0, sequence_length, n_features))
        for i in range(n_samples_total - sequence_length + 1):
            sequences.append(data[i:(i + sequence_length)])
        if not sequences: return np.empty((0, sequence_length, data.shape[1]))
        return np.array(sequences)
    
    def _preprocess_and_create_sequences(self, input_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Scales and windows input data, returning 3D NumPy sequences.
        Handles:
          - pandas DataFrame (2D)
          - NumPy array (2D): Scales and creates sequences.
          - NumPy array (3D): Assumes sequences are pre-made, scales features.
        """
        if self.scaler is None or self.sequence_length is None or self.model is None:
            raise RuntimeError("Model/scaler not ready or sequence length unknown. Call run() first.")

        n_features_expected = self.model.input_shape[-1]
        X: Optional[np.ndarray] = None # Initialize X

        if isinstance(input_data, pd.DataFrame):
            # print("Preprocessing DataFrame...") # Optional print
            if input_data.shape[1] != n_features_expected:
                raise ValueError(f"Input DataFrame has {input_data.shape[1]} features, model expects {n_features_expected}.")
            try:
                # Ensure data is numeric
                input_data_numeric = input_data.astype(np.number)
                data_normalized = self.scaler.transform(input_data_numeric)
                # Create 3D sequences from scaled 2D data
                X = self.__create_sequences(data_normalized, self.sequence_length)
            except Exception as e:
                raise RuntimeError(f"Failed to scale/sequence DataFrame: {e}") from e

        elif isinstance(input_data, np.ndarray):
            # print(f"Preprocessing NumPy array with {input_data.ndim} dimensions...") # Optional print

            # --- NEW: Handle 2D NumPy array ---
            if input_data.ndim == 2:
                if input_data.shape[1] != n_features_expected:
                     raise ValueError(f"Input 2D NumPy array has {input_data.shape[1]} features, model expects {n_features_expected}.")
                try:
                    # Scale the 2D data
                    data_normalized = self.scaler.transform(input_data)
                    # Create 3D sequences from scaled 2D data
                    X = self.__create_sequences(data_normalized, self.sequence_length)
                    # print(f"Created {X.shape[0]} sequences from 2D NumPy input.")
                except Exception as e:
                    raise RuntimeError(f"Failed to scale/sequence 2D NumPy array: {e}") from e
            # --- End New 2D Handling ---

            # --- Existing 3D NumPy array handling ---
            elif input_data.ndim == 3:
                if input_data.shape[1] != self.sequence_length:
                    raise ValueError(f"Input 3D NumPy sequence length {input_data.shape[1]} != expected {self.sequence_length}.")
                if input_data.shape[2] != n_features_expected:
                    raise ValueError(f"Input 3D NumPy features {input_data.shape[2]} != expected {n_features_expected}.")

                # Input is already 3D sequences, just need to scale the features
                X_input_3d = input_data
                n_samples, seq_len, n_feat = X_input_3d.shape
                try:
                    # Reshape to 2D for scaler [(n_samples * seq_len), n_feat]
                    X_reshaped_2d = X_input_3d.reshape(-1, n_feat)
                    # Scale the 2D data
                    X_scaled_2d = self.scaler.transform(X_reshaped_2d)
                    # Reshape back to 3D [n_samples, seq_len, n_feat]
                    X = X_scaled_2d.reshape(n_samples, seq_len, n_feat)
                    # print(f"Scaled features within {X.shape[0]} existing 3D sequences.")
                except Exception as e:
                    raise RuntimeError(f"Failed to scale features within 3D NumPy input: {e}") from e
            # --- End 3D Handling ---

            else:
                raise ValueError(f"NumPy input must be 2D or 3D, got {input_data.ndim}D.")
        else:
            raise TypeError("Input must be a pandas DataFrame or a 2D/3D NumPy array.")

        # Check if X was successfully created and is not empty
        if X is None or X.size == 0:
            # This can happen if input is too short for __create_sequences
            warnings.warn("No sequences generated from input data after preprocessing.", RuntimeWarning)
            # Return an empty array with the correct final dimensions
            return np.empty((0, self.sequence_length, n_features_expected))

        # Ensure float32 for model prediction compatibility
        if X.dtype != np.float32:
            try:
                X = X.astype(np.float32)
            except ValueError:
                warnings.warn("Could not cast sequences to float32.", RuntimeWarning)

        return X


    # --- Method to get anomaly score ---
    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculates reconstruction error for input data using the trained autoencoder.
        Higher error indicates a higher likelihood of anomaly.
        Accepts DataFrame (features) or 3D NumPy (pre-windowed sequences).

        Returns:
            np.ndarray: 1D array of reconstruction errors per sequence (shape: (n_sequences,)).
        """
        ## print("Calculating anomaly scores (reconstruction error)...") # Optional print
        # Preprocess input data into scaled 3D sequences
        X = self._preprocess_and_create_sequences(detection_data)

        if X.size == 0: return np.array([]) # No sequences to score

        # Get reconstruction from autoencoder
        ## print(f"Predicting reconstructions for {X.shape[0]} sequences...") # Optional print
        try:
            reconstructed = self.model.predict(X)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed during scoring. Input shape: {X.shape}. Error: {e}") from e

        # Handle shape mismatch (optional, but good practice)
        if X.shape != reconstructed.shape:
            warnings.warn(f"Shape mismatch input {X.shape} vs reconstruction {reconstructed.shape}.", RuntimeWarning)
            min_samples = min(X.shape[0], reconstructed.shape[0])
            if min_samples == 0: return np.array([])
            # Calculate error only on matching samples
            reconstruction_error = np.mean(np.square(X[:min_samples] - reconstructed[:min_samples]), axis=(1, 2))
        else:
            # Calculate reconstruction error (MSE per sequence)
            reconstruction_error = np.mean(np.square(X - reconstructed), axis=(1, 2))

        ## print(f"Calculated {len(reconstruction_error)} scores.") # Optional print
        return reconstruction_error # Return 1D scores

    # Detects anomalies and returns a list of boolean values
    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Detects anomalies by comparing reconstruction error scores to the threshold.
        Accepts DataFrame (features) or 3D NumPy (pre-windowed sequences).

        Returns:
            np.ndarray: A 1D boolean array (True=Anomaly), shape (n_sequences,).
        """
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Call run() first.")

        # Get the reconstruction error scores using the new method
        scores = self.get_anomaly_score(detection_data)

        if scores.size == 0: return np.array([], dtype=bool)

        # Compare scores to threshold (higher error = anomaly for reconstruction)
        anomalies = scores > self.threshold
        # print(f"Detected {np.sum(anomalies)} anomalies using threshold {self.threshold:.6f}.")
        return np.array(anomalies) # Return 1D boolean array
    
    
    def predict_proba(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts the probability of each sequence being an anomaly based on reconstruction error.
        This is a pseudo-probability derived by applying a sigmoid function to the scaled
        difference between the anomaly score and the threshold.

        Accepts DataFrame (features) or 3D NumPy (pre-windowed sequences).

        Returns:
            np.ndarray: Array of shape (n_sequences, 2) where:
                        Column 0: Probability of being Normal (1 - P(Anomaly))
                        Column 1: Probability of being Anomaly
                        Returns empty array shape (0, 2) if no sequences are generated.
        """
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Call run() first to train and set threshold.")
        if not np.isfinite(self.threshold) or self.threshold <= 0:
             warnings.warn(f"Threshold is not finite or non-positive ({self.threshold}). Probability calculation might be unreliable.", RuntimeWarning)
             # Handle non-positive threshold for scaling gracefully
             # If threshold is inf, all scores are below, prob_anomaly should be 0
             if self.threshold == np.inf:
                 n_scores = len(self.get_anomaly_score(detection_data)) # Need to know how many scores
                 return np.hstack([np.ones((n_scores, 1)), np.zeros((n_scores, 1))])


        # Get the reconstruction error scores
        scores = self.get_anomaly_score(detection_data)

        if scores.size == 0:
            return np.empty((0, 2)) # Return empty array with correct number of columns

        # Define the sigmoid function
        def sigmoid(x):
            # Clip x to avoid overflow in exp for very large negative values
            # and underflow for very large positive values.
            x_clipped = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x_clipped))

        # --- Calculate scale for sigmoid ---
        # Use the configured scale factor relative to the threshold.
        # Add a small epsilon to prevent division by zero if threshold is very close to zero.
        proba_scale_factor = self.config.get('proba_scale_factor', 4.0) # Default to 4.0 if not set
        scale = max(self.threshold / proba_scale_factor, 1e-9) # Ensure scale is positive

        # Calculate the scaled difference from the threshold
        scaled_diff = (scores - self.threshold) / scale

        # Apply sigmoid to get the probability of anomaly (Class 1)
        prob_anomaly = sigmoid(scaled_diff)

        # Probability of normal (Class 0) is 1 - P(Anomaly)
        prob_normal = 1.0 - prob_anomaly

        # Stack probabilities into the desired (n_sequences, 2) shape
        probabilities = np.vstack([prob_normal, prob_anomaly]).T

        ## print(f"Calculated anomaly probabilities for {probabilities.shape[0]} sequences.") # Optional print
        return probabilities