# EXACT - Explainable Anomaly Classification Tool

## 📑 Table of contents

- [About The Project](#-about-the-project)
- [How To Build](#-how-to-build)
- [Tools And Frameworks](#%EF%B8%8F-tools-and-frameworks)
- [Guide](#-guide)
  - [Frontend](#frontend)
  - [CLI-tool](#cli-tool)
- [For Developers](#-for-developers)
  - [Adding a new Machine Learning (ML) Model](#adding-a-new-machine-learning-ml-model)
  - [Adding a new Explainable AI (XAI) Method](#adding-a-new-explainable-ai-xai-method)
  - [Existing ML Models](#existing-ml-models)
  - [Existing XAI Methods](#existing-xai-methods)
  - [Adding an Anomaly Injection Method](#adding-an-anomaly-injection-method)
  - [Backend API](#backend-api)
  - [Database API](#database-api)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Authors](#-authors)

## 💻 About The Project

### Overview of EXACT
EXACT (Explainable Anomaly Classification Tool) is an evolution of the AnomDet tool, enhancing its capabilities with robust Explainable AI (XAI) features. While AnomDet focused on providing a framework for evaluating anomaly detection models against various anomaly injections, EXACT extends this by integrating XAI methods to offer insights into why an anomaly is detected. This allows users not only to identify anomalies but also to understand the contributing factors, making the system more transparent and actionable.

The system manages different anomaly detection algorithms and anomaly injection methods, simulating real-time data streams or processing data in batches. It provides a working framework for evaluating pre-defined anomaly detection models, understanding their responses to pre-defined anomaly injections, and explaining the reasoning behind detections. The architecture is designed to be modular, allowing users to easily define and integrate their own detection models, XAI techniques, and injection methods.

Interaction with the system is primarily through a web-based Frontend, with a CLI-tool also available (though its functionality with XAI features may be limited as it's a remnant from the original AnomDet).

### Features provided

EXACT allows for anomaly detection by importing a complete dataset in one batch. A machine learning model processes the data, labeling it as normal or anomalous. XAI methods are then applied to provide explanations for the detected anomalies. The results, including data visualizations, anomaly flags, and explanations, are presented in a user-friendly frontend.

The system includes a set of pre-defined anomaly detection algorithms, XAI methods, and anomaly injection techniques. The Frontend offer ways to interact with the system, check available models and methods, and manage jobs.

## 📝 How To Build

### Installation

1.  Install and run Docker Desktop or install Headless Docker on your system.
2.  Ensure Git is installed.
3.  Clone the repository:
    ```sh
    git clone [https://github.com/TedBoman/EXACT.git](https://github.com/TedBoman/EXACT.git)
    ```
4.  Navigate to the `Docker` directory:
    ```sh
    cd Docker
    ```
5.  Create a `.env` file in the `Docker` directory (you can copy `env.example` if provided, or create it manually).
6.  Set up the following environment variables in the `.env` file:
    ```env
    DATABASE_USER=your_db_user
    DATABASE_PASSWORD=your_db_password
    DATABASE_NAME=your_db_name
    DATABASE_HOST=host.docker.internal # Bridges the host localhost to the containers
    DATABASE_PORT=5432 # Default PostgreSQL port
    FRONTEND_PORT=8050 # Or your desired frontend port
    BACKEND_PORT=your_backend_port
    BACKEND_HOST=backend # Service name in docker-compose
    XAI_PLOT_OUTPUT_PATH=./xai_outputs # Path accessible by backend and frontend containers for XAI plots
    ```
    *Note: `XAI_PLOT_OUTPUT_PATH` should be a path that can be volume-mounted into both the backend and frontend containers. The example `./xai_outputs` implies it's relative to where `docker-compose` is run and will be created on the host.*
7.  Run the following command from the `Docker` directory to build and start the Docker containers:
    ```sh
    docker-compose up -d --build
    ```
8.  Your system should now be built and running. The frontend will typically be accessible at `http://localhost:YOUR_FRONTEND_PORT`.

### Additional Commands

* **Access the database (TimescaleDB/PostgreSQL) shell:**
    ```sh
    docker exec -it TSdatabase psql -U your_db_user -d your_db_name
    ```
    (Replace `TSdatabase`, `your_db_user`, `your_db_name` with actual values from your `docker-compose.yml` and `.env`)
    To exit `psql`, type `\q`.
* **Stop containers (preserving data):**
    ```sh
    docker-compose down
    ```
* **Stop containers and remove volumes (deletes data):**
    ```sh
    docker-compose down -v
    ```
* **Access a running container's shell (e.g., for the backend):**
    ```sh
    docker exec -it Backend bash
    ```
    (Replace `Backend` with the service name, e.g., `Frontend`, `TSdatabase`)

## 🛠️ Tools And Frameworks

* **Python:** The core language for the entire stack.
* **Docker:** For containerization of application components.
* **Dash by Plotly:** Python framework for building the interactive web application frontend.
* **TimescaleDB:** Open-source time-series SQL database, built on PostgreSQL, for storing telemetry and job data.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For machine learning utilities and some models.
* **XGBoost:** Gradient boosting library used for one of the ML models.
* **TensorFlow/Keras:** Deep learning library used for LSTM model.
* **SHAP (SHapley Additive exPlanations):** Library for calculating SHAP values for model explainability.
* **LIME (Local Interpretable Model-agnostic Explanations):** Library for generating local model explanations.
* **DiCE (Diverse Counterfactual Explanations):** Library for generating counterfactual explanations.
* **Psycopg2:** PostgreSQL adapter for Python to interact with TimescaleDB.

## 📚 Guide

Users primarily interact with EXACT through the web frontend. A CLI-tool exists but its compatibility with the latest XAI features is not guaranteed.

### Frontend

The Frontend is designed for ease of use in managing and interpreting anomaly detection jobs.

* **Main Page (`/`):**
    * **Job Creation:** Users can configure and start new anomaly detection jobs. This involves:
        * Uploading a dataset (CSV) or selecting an existing one.
        * Specifying the time column and, if the dataset is labeled, the label column.
        * Choosing an ML model for anomaly detection from a list of available models. Model-specific hyperparameters can be configured in a dynamic panel, with descriptions provided for each parameter.
        * Optionally enabling XAI, selecting one or more XAI methods, and configuring their specific parameters. This includes setting a sampling strategy for instances to explain and a seed for reproducibility.
        * Optionally configuring anomaly injection parameters to test model resilience.
        * Naming the job and starting it. The system only supports batch processing mode.
    * **Active Jobs List:** Displays currently running or completed jobs. Each job entry links to its dedicated results page and provides a button to stop/cancel the job.
    * **Explanation Boxes:** Contextual help is provided for ML model hyperparameters and XAI method descriptions as they are selected.

* **Job Results Page (`/job/<job_name>`):**
    * **Navigation:** A "Back to Home" button allows easy return to the main page.
    * **Job Metadata:** Displays a comprehensive summary of the job, including run timestamp, status, dataset used, model configuration, XAI settings, anomaly injection parameters (if any), data summary (total rows, features, anomalies), performance metrics (accuracy, precision, recall, F1-score, etc. for both testing data and all data), cross-validation metrics (if applicable to the model), and execution times for different stages of the job (simulation, training, detection, XAI). Detailed definitions of performance metrics are also available.
    * **Time Series Visualization:** Plots the time series data with detected anomalies highlighted. Users can select which features to display on the graph.
    * **XAI Results Display:** This section dynamically loads and displays the outputs from the XAI methods run for the job. This can include:
        * Feature importance plots (e.g., SHAP summary plots, LIME explanations as HTML).
        * Counterfactual explanations (e.g., DiCE results in tabular format highlighting changes).
        * An aggregated feature importance comparison plot if multiple XAI methods were used and produced comparable scores.
        * Other visualizations specific to the XAI method (e.g., SHAP waterfall, force plots, heatmaps saved as images or HTML).
        XAI outputs are served from a dedicated directory specified by `XAI_PLOT_OUTPUT_PATH`.

### CLI-tool

The CLI-tool (`EXACT/Backend/api/cli_tool.py`) allows for interaction with the backend via command-line arguments. Its features largely reflect the capabilities of the original AnomDet tool. While it can list models, datasets, and manage basic job operations, its support for initiating jobs with detailed XAI parameters may be limited. Refer to `python cli_tool.py help` (if the script is made executable or run directly) for command options. The old README provides more details on its original usage.

## ☕ For Developers

EXACT is designed to be extensible. Here's how you can add new components:

### Adding a new Machine Learning (ML) Model

1.  **Implement the Model Interface:**
    Create a new Python file in `EXACT/Backend/ML_models/`. Your new model class must inherit from `ModelInterface` (`EXACT/Backend/ML_models/model_interface.py`).
    ```python
    from ML_models import model_interface
    import pandas as pd
    import numpy as np
    from typing import Optional, Union, List, Dict, Any

    class MyNewModel(model_interface.ModelInterface):
        def __init__(self, **kwargs: Any):
            # Initialize your model, store parameters from kwargs
            # Example: self.my_param = kwargs.get('my_param', default_value)
            self.model = None # Your actual model instance
            self.scaler = None # e.g., MinMaxScaler()
            # ... other necessary attributes like sequence_length for XAI wrapper ...
            self.sequence_length = kwargs.get('sequence_length', 1) # Important for XAI
            self.original_feature_names_: Optional[List[str]] = None
            self.processed_feature_names_: Optional[List[str]] = None # If features are transformed
            self.input_type: Optional[str]] = None # 'dataframe' or 'numpy'
            self.n_original_features: Optional[int] = None
            self.label_col: Optional[str] = None

        def run(self, X: Union[pd.DataFrame, np.ndarray],
                y: Optional[np.ndarray] = None, # For NumPy array inputs
                label_col: str = 'label', # For DataFrame inputs
                original_feature_names: Optional[List[str]] = None # For NumPy array inputs
               ) -> None:
            # Preprocess data (e.g., scaling, reshaping if necessary)
            # self._prepare_data_for_model(...) can be a helper
            # Train your model (self.model.fit(...))
            # Set any thresholds or learned parameters
            pass

        def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            # Preprocess detection_data similar to run method (use fitted scaler, etc.)
            # predictions = self.model.predict(...)
            # Return a NumPy array of booleans (True for anomaly)
            pass

        # Optional: Implement if your model provides scores/probabilities
        def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            # Return a NumPy array of anomaly scores
            pass

        # REQUIRED if XAI methods need probability outputs (most do for classification)
        def predict_proba(self, X_xai: np.ndarray) -> np.ndarray:
            # Preprocess X_xai (often a 2D or 3D NumPy array from XAI tool)
            # Ensure data is scaled/transformed as in training
            # probabilities = self.model.predict_proba(...)
            # Return NumPy array of shape (n_samples, n_classes), e.g., (n_samples, 2) for binary
            pass

        # Optional: For models with cross-validation scores
        def get_validation_scores(self) -> Dict[str, float]:
            # return self.validation_scores_ # Or however you store them
            return {}

        # Optional: For models with feature importances
        def get_feature_importances(self) -> Optional[Dict[str, float]]:
            # return dict(zip(self.processed_feature_names_, self.model.feature_importances_))
            return None
    ```
    Refer to existing models like `XGBoost.py` or `decision_tree.py` for examples of implementing preprocessing, scaling, cross-validation, and handling different input types within the `_prepare_data_for_model` helper.

2.  **Register the Model:**
    Add your new model to `EXACT/Backend/ML_models/get_model.py`:
    ```python
    from ML_models import my_new_model # Import your new model file

    def get_model(model_name_str: str, **model_params: Any) -> model_interface.ModelInterface:
        match model_name_str:
            # ... existing cases ...
            case "MyNewModelName": # This name will be used in frontend/API
                instance = my_new_model.MyNewModel(**model_params)
                return instance
            case _:
                raise ValueError(f"Unknown model type requested: {model_name_str}")
    ```

3.  **Update Frontend:**
    * **Parameters & Descriptions:** Add your model and its tunable hyperparameters to `EXACT/Frontend/ml_model_hyperparameters.py` in the `HYPERPARAMETER_DESCRIPTIONS` dictionary.
        ```python
        HYPERPARAMETER_DESCRIPTIONS = {
            # ... existing models ...
            "MyNewModelName": {
                "my_param": "Description of my_param.",
                "another_param": "Description of another_param."
                # Add wrapper params like imputer_strategy, n_splits, auto_tune if applicable
            }
        }
        ```
    * **UI Components:** In `EXACT/Frontend/callbacks.py`, modify the `update_model_settings_panel` callback to include UI elements (e.g., `dcc.Input`, `dcc.Dropdown`) for your model's parameters when "MyNewModelName" is selected in the `detection-model-dropdown`. Pattern-matching IDs should be used for these components.

4.  **Update Backend Interpretation (If Necessary):**
    The `ModelWrapperForXAI` in `EXACT/Backend/ML_models/model_wrapper.py` standardizes how XAI tools interact with models. It uses a `score_interpretation` argument (`'lower_is_anomaly'` or `'higher_is_anomaly'`).
    In `EXACT/Backend/execute_calls.py`, within the `run_batch` function, there's an `interpretation_list` (e.g., `interpretation_list = ['lstm', 'XGBoost', 'decision_tree']`). Models in this list are assumed to have `higher_is_anomaly` interpretation for their scores.
    * If your new model's raw anomaly scores mean "lower is anomaly", add its registered name (e.g., "MyNewModelName") to this `interpretation_list`.
    * If its scores mean "higher is anomaly", ensure it's **not** in the list (or the logic is adapted if the default changes).
    This ensures the `ModelWrapperForXAI` correctly interprets your model's scores for XAI methods that rely on probabilities or score ranking.

### Adding a new Explainable AI (XAI) Method

1.  **Implement the XAI Interface:**
    Create a new Python file in `EXACT/Backend/XAI_methods/methods/`. Your new XAI class must inherit from `ExplainerMethodAPI` (`EXACT/Backend/XAI_methods/explainer_method_api.py`).
    ```python
    from XAI_methods.explainer_method_api import ExplainerMethodAPI
    import numpy as np
    from typing import Any, List, Dict

    class MyNewXAIMethod(ExplainerMethodAPI):
        def __init__(self, model: Any, background_data: np.ndarray, **params: Any):
            # model: The ModelWrapperForXAI instance
            # background_data: Often 3D NumPy array (samples, seq_len, features)
            # params: Can include 'mode', 'feature_names', 'sequence_length', etc.
            self.model_wrapper = model
            self.background_data = background_data
            self.mode = params.get('mode')
            self.base_feature_names = params.get('feature_names')
            self.sequence_length = params.get('sequence_length')
            # ... initialize your explainer (e.g., a SHAP explainer instance) ...

        def explain(self, instances_to_explain: np.ndarray, **kwargs: Any) -> Any:
            # instances_to_explain: 3D NumPy array (n_instances, seq_len, n_features)
            # kwargs: Runtime parameters for this explanation (e.g., num_features for LIME)
            # ... logic to generate explanations ...
            # Return the explanation in a format suitable for visualization/processing
            # (e.g., SHAP values as NumPy array, LIME explanation object, DiCE counterfactuals object)
            pass
    ```
    Refer to `ShapExplainer.py`, `LimeExplainer.py`, or `DiceExplainer.py` for examples.

2.  **Register in XAI Factory:**
    Add your method to `EXACT/Backend/XAI_methods/xai_factory.py`:
    ```python
    from XAI_methods.methods.my_new_xai_method import MyNewXAIMethod # Import

    def xai_factory(...) -> ExplainerMethodAPI:
        match method_key:
            # ... existing cases ...
            case "mynewxaimethodname": # Lowercase key
                explainer_instance = MyNewXAIMethod(
                    ml_model=ml_model,
                    background_data=background_data,
                    **kwargs
                )
                return explainer_instance
            case _:
                # ...
    ```

3.  **Update Frontend:**
    * **Parameters & Descriptions:** Add your XAI method and its tunable parameters to `EXACT/Frontend/ml_model_hyperparameters.py` in the `XAI_METHOD_DESCRIPTIONS` dictionary.
        ```python
        XAI_METHOD_DESCRIPTIONS = {
            # ... existing methods ...
            "MyNewXAIMethodName": { # Match the name used in get_xai_methods()
                "description": "Description of MyNewXAIMethod.",
                "capabilities": "...",
                "limitations": "...",
                "parameters": {
                    "xai_param1": "Description of xai_param1."
                }
            }
        }
        ```
    * **UI Components:** In `EXACT/Frontend/callbacks.py`, modify the `update_xai_settings_panel` callback to include UI elements for your XAI method's parameters when "MyNewXAIMethodName" is selected in the `xai-method-dropdown`.

4.  **Add Visualization Logic:**
    * In `EXACT/Backend/XAI_methods/xai_visualizations.py`, create a new function (e.g., `process_and_plot_mynewxai`) to handle the output of your XAI method and generate plots (e.g., saving images or HTML files to the job's output directory: `output_dir/job_name/MyNewXAIMethodName/`).
    * In `EXACT/Backend/XAI_methods/xai_runner.py`, within the `run_explanations` method:
        * Import your new visualization function.
        * Add your method's name to the `plot_handlers` dictionary, mapping it to your visualization function.
        * If your method requires specific runtime arguments for its `explain` call (beyond `instances_to_explain`), ensure these are collected from `settings` and passed correctly when calling `ts_explainer.explain(..., method_name="MyNewXAIMethodName", **specific_runtime_kwargs)`.

### Existing ML Models

The EXACT tool comes with several pre-integrated ML models. Their parameters can be configured via the frontend.

* **LSTM Autoencoder (`lstm.py`):**
    * **Purpose:** An unsupervised neural network model for detecting anomalies based on reconstruction error. Sequences that are difficult to reconstruct are flagged as anomalous.
    * **Key Parameters:** `units` (LSTM layer units), `activation`, `optimizer`, `learning_rate`, `loss` (for AE training), `epochs`, `batch_size`, `time_steps` (sequence length).
    * **Interpretation:** Higher reconstruction error implies anomaly.

* **Isolation Forest (`isolation_forest.py`):**
    * **Purpose:** An unsupervised ensemble method that isolates observations by randomly selecting a feature and then randomly selecting a split value. Anomalies are typically isolated in fewer splits.
    * **Key Parameters:** `n_estimators` (number of trees), `contamination` (expected proportion of outliers), `max_samples`, `max_features`, `bootstrap`.
    * **Interpretation:** Lower decision_function scores (more negative) indicate higher anomaly likelihood.

* **One-Class SVM with Autoencoder Preprocessing (`svm.py`):**
    * **Purpose:** An unsupervised model. First, an autoencoder reduces data dimensionality. Then, a One-Class SVM is trained on these encoded representations to find a boundary that encloses normal data.
    * **Key AE Parameters:** `encoding_dim`, `ae_activation`, `ae_output_activation`, `optimizer`, `learning_rate`, `loss` (AE), `epochs`, `batch_size`.
    * **Key SVM Parameters:** `svm_nu`, `svm_kernel`, `svm_gamma`.
    * **Interpretation:** Lower decision_function scores from the SVM (more negative) indicate higher anomaly likelihood.

* **XGBoost (`XGBoost.py`):**
    * **Purpose:** A supervised gradient boosting model for classification. Requires labeled data.
    * **Key Parameters:** `n_estimators`, `learning_rate`, `max_depth`, `objective` (e.g., 'binary:logistic'), `eval_metric`. Wrapper parameters include `n_splits` for CV, `early_stopping_rounds`.
    * **Interpretation:** Higher probability for the anomaly class (typically class 1) indicates anomaly.

* **Decision Tree (`decision_tree.py`):**
    * **Purpose:** A supervised model that creates a tree-like structure for classification. Requires labeled data.
    * **Key Parameters:** `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`. Wrapper parameters include `imputer_strategy`, `n_splits` for CV, `auto_tune` options.
    * **Interpretation:** Higher probability for the anomaly class indicates anomaly.

* **SGDClassifier (`SGDClassifier.py`):**
    * **Purpose:** A supervised linear model (e.g., SVM, Logistic Regression) trained with Stochastic Gradient Descent. Requires labeled data.
    * **Key Parameters:** `loss` (e.g., 'hinge' for SVM, 'log_loss' for logistic regression), `penalty`, `alpha` (regularization strength), `max_iter`, `class_weight`, `learning_rate`. Wrapper parameters include `scaler_type`, `imputer_strategy`, `n_splits` for CV, `auto_tune` options, `calibrate_probabilities`.
    * **Interpretation:** Higher probability for the anomaly class indicates anomaly (especially if calibrated or using a probabilistic loss).

### Existing XAI Methods

EXACT integrates the following XAI methods:

* **SHAP (SHapley Additive exPlanations) (`ShapExplainer.py`):**
    * **Purpose:** Assigns an importance value (SHAP value) to each feature for a particular prediction, based on game theory. Explains how features contribute to pushing the model output from a base value.
    * **Needs:** A trained model (wrapped by `ModelWrapperForXAI`), background data for reference.
    * **Key Parameters:** `shap_method` ('kernel', 'tree'), `nsamples` (for KernelSHAP), `l1_reg_k_features` (for KernelSHAP feature selection).
    * **Output:** SHAP values for each feature, for each instance explained. Visualized as summary plots, waterfall plots, force plots, etc.

* **LIME (Local Interpretable Model-agnostic Explanations) (`LimeExplainer.py`):**
    * **Purpose:** Explains individual predictions by learning a simpler, interpretable linear model locally around the prediction.
    * **Needs:** A trained model (wrapped), background data for sampling statistics.
    * **Key Parameters:** `num_features` (in explanation), `num_samples` (perturbations), `kernel_width`, `feature_selection`, `discretize_continuous`.
    * **Output:** A list of feature weights for a specific instance. Visualized as an HTML report.

* **DiCE (Diverse Counterfactual Explanations) (`DiceExplainer.py`):**
    * **Purpose:** Generates counterfactual examples – minimal changes to feature values that flip the model's prediction to a desired outcome.
    * **Needs:** A trained model (wrapped), background data (including outcomes) to initialize DiCE's data interface.
    * **Key Parameters:** `total_CFs` (number of counterfactuals), `desired_class` (for classification), `features_to_vary`, `backend` (ML framework), `dice_method` ('random', 'genetic', 'kdtree').
    * **Output:** A set of counterfactual instances. Visualized by showing the original instance and the counterfactuals, highlighting changes.

### Adding an Anomaly Injection Method

To add a new anomaly injection method:

1.  **Create Method File:**
    Add a new Python file in `EXACT/Backend/Simulator/AnomalyInjector/InjectionMethods/`.
    Define a class with an `inject_anomaly` method.
    ```python
    # Example: EXACT/Backend/Simulator/AnomalyInjector/InjectionMethods/my_injector.py
    class MyNewAnomalyInjector:
        def inject_anomaly(self, data_series_to_modify, rng, data_range, mean, settings_for_method):
            # data_series_to_modify: pd.Series of the data for one column at specific indices
            # rng: np.random.Generator for randomness
            # data_range, mean: statistics of the original data series (or relevant segment)
            # settings_for_method: AnomalySetting object containing magnitude, etc.
            # ... your logic to modify data_series_to_modify ...
            return modified_series
    ```
    Refer to existing injectors like `spike.py` or `lowered.py`.

2.  **Register in AnomalyInjector:**
    In `EXACT/Backend/Simulator/AnomalyInjector/anomalyinjector.py`:
    * Import your new class: `from Simulator.AnomalyInjector.InjectionMethods.my_injector import MyNewAnomalyInjector`
    * Add an `elif` block in the `_apply_anomaly` method:
        ```python
        elif anomaly_type == 'MyNewAnomalyName': # Match the 'anomaly_type' string
            injector = MyNewAnomalyInjector()
            return injector.inject_anomaly(data_series_to_modify, rng, data_range, mean, settings_for_method)
        ```

3.  **Update Frontend (Optional):**
    If your new injection method has unique parameters not covered by the existing UI (anomaly_type, timestamp, magnitude, percentage, duration, columns), you would need to:
    * Update `EXACT/Frontend/pages/index.py` to include input fields for these new parameters within the "injection-panel" when your method is selected.
    * Ensure these new parameters are collected in the `start_job_handler` callback in `EXACT/Frontend/callbacks.py` and included in the `inj_params` sent to the backend.
    The backend API (`EXACT/Backend/api/api.py`) and `execute_calls.py` should generally handle passing `inj_params` through, as long as the structure is a list of dictionaries. The `AnomalySetting` class in `EXACT/Backend/Simulator/DBAPI/type_classes.py` might need new attributes if your method requires fundamentally new types of settings.

### Backend API

The system's backend functionalities are exposed through an API defined in `EXACT/Backend/api/api.py`. This API is used by the frontend and the CLI-tool to send requests (e.g., start job, get data, list models) to the backend engine (`EXACT/Backend/engine.py`). The `API_README.md` (`EXACT/Backend/API_README.md`) provides an overview of the original API methods, though some details might have evolved with XAI integration. Key interactions involve sending JSON-formatted requests over a socket.

### Database API

EXACT uses a database interface defined in `EXACT/Backend/Simulator/DBAPI/db_interface.py` (the version in `EXACT/Database/db_interface.py` appears to be an older or alternative version, while `EXACT/Backend/Simulator/DBAPI/db_interface.py` is more aligned with the simulator's usage) and implemented for TimescaleDB in `EXACT/Database/timescaledb_api.py`. This modular design allows for potential adaptation to other time-series databases by implementing the `DBInterface`. The interface defines methods for creating tables, inserting data, reading data, dropping tables, and checking table existence.

## 🔍 Troubleshooting

* **Docker Build/Compose Failures:**
    * Ensure Docker Desktop (or Docker daemon) is running.
    * Check for port conflicts (e.g., if `FRONTEND_PORT` or `DATABASE_PORT` are already in use). Modify your `.env` if necessary.
    * Verify internet connectivity for pulling base images.
    * Look for specific error messages in the Docker build output.
* **Container Not Starting:**
    * Use `docker ps -a` to see the status of containers.
    * Check container logs: `docker logs <container_name>` (e.g., `docker logs Backend`, `docker logs Frontend`, `docker logs TSdatabase`).
    * Ensure `.env` file is correctly configured and present in the `Docker` directory.
* **Frontend Not Connecting to Backend:**
    * Verify `BACKEND_HOST` and `BACKEND_PORT` are correctly set in the frontend's environment (usually managed by `docker-compose.yml` linking or environment variables passed to the frontend container).
    * Check backend container logs for errors during startup or request handling.
* **Database Connection Issues (from Backend):**
    * Ensure the `TSdatabase` container is running.
    * Verify `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER`, `DATABASE_PASSWORD`, `DATABASE_NAME` in the backend's environment (via `.env` and `docker-compose.yml`) match the TimescaleDB container's settings.
* **XAI Plots Not Appearing:**
    * Verify the `XAI_PLOT_OUTPUT_PATH` in your `.env` file points to a valid path on your host machine that can be mounted as a volume.
    * Ensure this volume is correctly mounted in both the `backend` and `dash-frontend` services in `docker-compose.yml`.
    * Check backend logs for any errors during XAI plot generation or saving.
    * Check frontend (Dash app) logs and browser developer console for errors related to serving static assets from `/xai-assets/`.
* **Job Failures:**
    * The primary source of information will be the backend container logs (`docker logs Backend`). Look for Python tracebacks or error messages related to data processing, model training/detection, or XAI execution.
    * The "Job Metadata" section on the job results page might show a "Failed" status or incomplete information.
* **"No displayable XAI results" or "Method subdirectory not found":**
    * This means the backend either didn't run the XAI method, encountered an error during XAI, or didn't save output files to the expected location (`XAI_PLOT_OUTPUT_PATH/job_name/MethodName/`). Check backend logs.


## 📄 License

This project is licensed under Creative Commons Attribution 4.0 International. See `LICENCE` for more details.

## ✍ Authors

* [TedBoman](https://github.com/TedBoman)
* [TheoGould/SlightlyRoasted](https://github.com/SlightlyRoasted)

## 🙏 Acknowledgements

EXACT builds upon the foundational work of the AnomDet tool. We would like to acknowledge the original developers of AnomDet:

* [TedBoman](https://github.com/TedBoman) - AnomDet
* [SlightlyRoasted](https://github.com/SlightlyRoasted) - AnomDet
* [MarcusHammarstrom](https://github.com/MarcusHammarstrom) - AnomDet
* [Liamburberry](https://github.com/Liamburberry) - AnomDet
* [MaxStrang](https://github.com/MaxStrang) - AnomDet
- [valens-twiringiyimana](https://github.com/valens-twiringiyimana) - AnomDet
- [Seemihh](https://github.com/Seemihh) - AnomDet
