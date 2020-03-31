## Ensemble-Pytorch
Implementation of scikit-learn like ensemble methods in Pytorch.

### Methods
* **VotingClassifier**: [Completed]
* **BaggingClassifier**: [Completed]
* **GradientBoostingClassifier**: [Completed]

### How to use
```python
''' 
  Please see examples in ./script for details 
'''

# Base learner, ensmeble method
from base_learner_module import base_learner
from ensemble.method_module import ensemble_method

# Load train/test loader
train_loader = DataLoader(...)
test_loader = DataLoader(...)

# Arguments for ensemble method (e.g., n_estimator) and base learner
ensemble_args = {}
base_learner_args = {}

# Train/Evaluate
model = ensemble_method(ensemble_args, base_learner, base_learner_args)
model.fit(train_loader)
model.evaluate(test_loader)
```

### Experiment
* The table below presents the performance of different ensemble methods on CIFAR-10 dataset
* Each of them uses 10 LeNet-5 model (with RELU activation and Dropout) as base learners
* Results can be reproduced by running ``./scripts/cifar_cnn_demo.py``

| Model Name | Params (MB) | Testing Acc (%) | Improvement (%) |
| ------ | ------ | ------  | ------ |
| **Single LeNet-5 (Baseline)** | 0.07 | 72.89 | - |
| **Single AlexNet** | 2.47 | 77.22 | + 4.33 |
| **VotingClassifier (10)** | 3.17 | 78.25 | + 5.36 |
| **BaggingClassifier (10)** | 3.17 | 77.32 | + 4.43 |
| **GradientBoostingClassifier (10)** | 3.17 | 80.96 | + 8.07 |

### Reference
1. Zhou, Zhi-Hua. "Ensemble methods: foundations and algorithms." CRC press (2012).
2. Friedman, Jerome H. "Greedy function approximation: a gradient boosting machine." Annals of statistics (2001): 1189-1232.
