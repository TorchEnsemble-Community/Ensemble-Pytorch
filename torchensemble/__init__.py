from .fusion import FusionClassifier
from .fusion import FusionRegressor
from .voting import VotingClassifier
from .voting import VotingRegressor
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .gradient_boosting import GradientBoostingClassifier
from .gradient_boosting import GradientBoostingRegressor
from .snapshot_ensemble import SnapshotEnsembleClassifier
from .snapshot_ensemble import SnapshotEnsembleRegressor


__all__ = ["FusionClassifier",
           "FusionRegressor",
           "VotingClassifier",
           "VotingRegressor",
           "BaggingClassifier",
           "BaggingRegressor",
           "GradientBoostingClassifier",
           "GradientBoostingRegressor",
           "SnapshotEnsembleClassifier",
           "SnapshotEnsembleRegressor"]
