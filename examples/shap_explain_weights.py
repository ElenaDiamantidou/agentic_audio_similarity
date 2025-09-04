import xgboost as xgb
import shap
import numpy as np
import pandas as pd

from src.utils.dataframe_utils import explode_dict_columns
from data.create_train_set_for_aggregator_model import load_data
from data.loader import DaTACOSDataLoader

# Initialize loader
loader = DaTACOSDataLoader(
    meta_path="../data/da-tacos/da-tacos_metadata/da-tacos_benchmark_subset_metadata.json",
    features_cens_path="../data/da-tacos/da-tacos_benchmark_subset_cens",
    features_hpcp_path="../data/da-tacos/da-tacos_benchmark_subset_hpcp"
)
sampled_dataset = pd.read_csv("../data/da-tacos/train_set_for_aggregator_model.csv")
feature_cols = ["sim_hpcp", "sim_centroid", "sim_metadata"]

data = sampled_dataset[feature_cols].rename(columns={
    "sim_hpcp": "hpcp_audio_similarity",
    "sim_centroid": "cen_audio_similarity",
    "sim_metadata": "metadata_audio_similarity"
})

labels = 0.35*data["hpcp_audio_similarity"] + 0.35*data["cen_audio_similarity"] + 0.3*data["metadata_audio_similarity"]

# Train model
model = xgb.XGBRegressor()
model.fit(data, labels)

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(data)

# Summary plot (global feature importance)
shap.summary_plot(shap_values, data)

# Bar plot version
shap.summary_plot(shap_values, data, plot_type="bar")

# Force plot (local explanation for one prediction)
i = 5
shap.force_plot(explainer.expected_value, shap_values[i,:], data.iloc[i,:])
