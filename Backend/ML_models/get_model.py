from ML_models import isolation_forest
from ML_models import lstm 
from ML_models import svm
from ML_models import XGBoost
from ML_models import decision_tree
from ML_models import SGDClassifier
def get_model(model, **model_params):
    match model:
        case "lstm":
            # Pass unpacked model_params to the constructor
            lstm_instance = lstm.LSTMModel(**model_params)
            return lstm_instance

        case "isolation_forest":
            # Pass unpacked model_params to the constructor
            if_instance = isolation_forest.IsolationForestModel(**model_params)
            return if_instance

        case "svm":
            # Pass unpacked model_params to the constructor
            svm_instance = svm.SVMModel(**model_params)
            return svm_instance

        case "XGBoost":
            # Pass unpacked model_params to the constructor
            XGBoost_instance = XGBoost.XGBoostModel(**model_params)
            return XGBoost_instance

        case "decision_tree":
            # Pass unpacked model_params to the constructor
            DT_instance = decision_tree.DecisionTreeModel(**model_params)
            return DT_instance
         
        case "SGDClassifier":
            # Pass unpacked model_params to the constructor
            svc_instance = SGDClassifier.SGDLinearModel(**model_params)
            return svc_instance

        # Add a default case to handle unknown model types gracefully
        case _:
             raise ValueError(f"Unknown model type requested: {model}")
            
    