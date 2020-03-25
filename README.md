## Ensemble-Pytorch
Implementation of scikit-learn like ensemble methods using Pytorch.

### Supported methods
* **VotingClassifier**: [Completed]
* **BaggingClassifier**: [Completed]
* **GradientBoostingClassifier**: [Completed]
* **StackingClassifier**
* **VotingRegressor**
* **BaggingRegressor**
* **GradientBoostingRegressor**
* **StackingRegressor**

### How to use
```python
''' Please see examples in ./script for detailed introduction '''

# Import base learner class and ensmeble method class
from base_learner_module import base_learner
from ensemble.method_module import ensemble_method

# Define arguments for ensemble method (e.g., n_estimator) and base learner, separately
ensemble_args = {}
base_learner_args = {}

# Initialize model
model = VotingClassifier(ensemble_args, base_learner, base_learner_args)

# Training and Evaluating
model.fit(train_loader)
model.evaluate(test_loader)
```

### Experiments

### Package dependency