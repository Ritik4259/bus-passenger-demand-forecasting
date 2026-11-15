# 0004: Excluding Vehicle Capacity from Exogenous Demand Features

In a decision-support system, dispatchers need to evaluate "what-if" counterfactuals (e.g., "what happens to occupancy if I assign a 30-passenger bus instead of a 60-passenger bus?"). We decided to strictly exclude vehicle capacity from the regression model's input feature set, treating passenger demand as an exogenous property of route, time, weather, and calendar factors. This prevents reverse causality where the model would artificially depress predicted passenger demand whenever a smaller bus is assigned.
