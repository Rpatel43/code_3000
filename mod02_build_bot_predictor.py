# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.02,
        n_estimators=800,
        max_depth=2,
        subsample=1.0,
        min_samples_leaf=10,
        random_state=seed
    )
    model.fit(X, y)
    return model