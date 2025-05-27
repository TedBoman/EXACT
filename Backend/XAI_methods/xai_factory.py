from typing import Any, Dict, List
import warnings
import numpy as np

# Import the specific explainer CLASSES
from XAI_methods.methods.ShapExplainer import ShapExplainer
from XAI_methods.methods.LimeExplainer import LimeExplainer
from XAI_methods.methods.DiceExplainer import DiceExplainer

# Import the base API
try:
    from explainer_method_api import ExplainerMethodAPI
except ImportError:
    # Define a dummy 
    import abc
    class ExplainerMethodAPI(abc.ABC): pass

def xai_factory(
    method_name: str,               # Input: "shap", "lime", etc.
    ml_model: Any,                  # Input: The trained ML model instance
    background_data: np.ndarray,    # Input: The 3D NumPy background data
    **kwargs: Any                    # Input: Catch-all for other params like mode, feature_names
) -> ExplainerMethodAPI:             # Output: An initialized explainer object
    """
    Factory function to get an initialized XAI explainer object.

    Args:
        method_name (str): The requested explanation method (e.g., 'shap').
        ml_model (Any): The trained machine learning model instance.
        background_data (np.ndarray): The 3D NumPy background data sample.
        **kwargs (Any): Additional keyword arguments needed by specific explainers
                       (e.g., mode='classification', feature_names=['f1', 'f2']).

    Returns:
        ExplainerMethodAPI: An initialized explainer object ready for use.

    Raises:
        ValueError: If method_name is unsupported or required kwargs are missing.
        RuntimeError: If instantiation fails.
    """
    method_key = method_name.lower()
    #print(f"get_explainer factory called for: '{method_key}'")

    match method_key:
        case "shapexplainer":
            # print(f"Attempting to instantiate ShapExplainer...")
            try:
                # Pass the received arguments TO the ShapExplainer constructor
                explainer_instance = ShapExplainer(
                    model=ml_model,
                    background_data=background_data,
                    # Pass necessary params from kwargs. 'mode' is used by ShapExplainer.
                    **kwargs
                )
                # Quick check (optional): Verify it matches the expected API type
                if not isinstance(explainer_instance, ExplainerMethodAPI):
                     warnings.warn(f"Instantiated object for '{method_key}' may not fully implement ExplainerMethodAPI.", RuntimeWarning)

                return explainer_instance
            except KeyError as e:
                #  print(f"Error instantiating ShapExplainer: Missing required parameter in kwargs: {e}")
                 raise ValueError(f"Missing required parameter for ShapExplainer: {e}") from e
            except Exception as e:
                #  print(f"Error instantiating ShapExplainer: {e}")
                 raise RuntimeError(f"Failed to instantiate ShapExplainer for method '{method_key}'") from e

        case "limeexplainer":
            # print(f"Attempting to instantiate LimeExplainer...")
            try:
                # Pass the received arguments TO the LimeExplainer constructor
                explainer_instance = LimeExplainer(
                    model=ml_model,
                    background_data=background_data,
                    **kwargs
                )
                # Quick check (optional): Verify it matches the expected API type
                if not isinstance(explainer_instance, ExplainerMethodAPI):
                     warnings.warn(f"Instantiated object for '{method_key}' may not fully implement ExplainerMethodAPI.", RuntimeWarning)

                return explainer_instance
            
            except Exception as e:
                # print(f"Error instantiating LimeExplainer: {e}")
                raise RuntimeError(f"Failed to instantiate LimeExplainer for method '{method_key}'") from e

        case "diceexplainer":
            # print(f"Attempting to instantiate DiceExplainer...")
            try:
                # Pass the received arguments TO the LimeExplainer constructor
                explainer_instance = DiceExplainer(
                    model=ml_model,
                    background_data=background_data,
                    **kwargs
                )
                # Quick check (optional): Verify it matches the expected API type
                if not isinstance(explainer_instance, ExplainerMethodAPI):
                     warnings.warn(f"Instantiated object for '{method_key}' may not fully implement ExplainerMethodAPI.", RuntimeWarning)

                return explainer_instance
            
            except Exception as e:
                # print(f"Error instantiating DiceExplainer: {e}")
                raise RuntimeError(f"Failed to instantiate DiceExplainer for method '{method_key}'") from e

        case _:
            # Handle unsupported method names
            supported = ["ShapExplainer", "LimeExplainer", "DiceExplainer"] # Add when implemented
            raise ValueError(f"Unsupported explanation method requested: '{method_key}'. Supported: {supported}")