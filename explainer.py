# import file directory of the class_based_models 
from class_based_models import lung_cancer_mlp
import shap

# Outline (code from random forest)

explainer = shap.Explainer(model, X_train_resampled)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)