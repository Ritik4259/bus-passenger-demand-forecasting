# 0001: Unconstrained Latent Demand as the Prediction Target

In transit operations, observed passenger boardings are censored whenever demand exceeds vehicle capacity, hiding true passenger volume spikes. We decided to model and forecast true unconstrained passenger demand (`passenger_demand`) rather than observed boardings, using vehicle capacity strictly as an evaluation and action threshold. This ensures the ML models can accurately detect and trigger `OVERFLOW_RISK` actions when passenger demand outstrips vehicle capacity.
