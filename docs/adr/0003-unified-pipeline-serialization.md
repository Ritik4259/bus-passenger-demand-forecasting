# 0003: Unified End-to-End Pipeline Serialization in Joblib

Serving ML models often breaks due to feature drift between offline feature scripts and online request handling. We decided to encapsulate all preprocessing (cyclical time encoders, categorical encoders, scaling) and the regression estimator inside a unified scikit-learn `Pipeline` saved to `.joblib`. This ensures that FastAPI inference receives raw parameters and transforms them identically to the training pipeline, maintaining modularity for the live debugging task.
