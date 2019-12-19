## Ensemble-Pytorch
Implementation of ensemble methods using Pytorch. Some parallel ensemble methods can be accelerated by exploiting the differentiable property of base learners.

### Supported methods
* **VotingClassifier**
* **BaggingClassifier**
* **GradientBoostingClassifier**
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