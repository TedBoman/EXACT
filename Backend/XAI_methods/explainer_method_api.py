import abc
import numpy as np
from typing import Any, List

class ExplainerMethodAPI(abc.ABC):
    """
    Abstract Base Class defining the interface for a specific XAI method implementation
    (e.g., a SHAP explainer wrapper, a LIME explainer wrapper).
    """

    @abc.abstractmethod
    def explain(self, instances_to_explain: np.ndarray, **kwargs: Any) -> Any:
        """
        Execute the explanation process for the given instances.
        """
        raise NotImplementedError