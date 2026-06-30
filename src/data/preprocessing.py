import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

class DomainAwareImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer based on feature groups.
    - Trajectory/Flags/Counts -> 0
    - days_since -> max_train + 1
    - Ratios/Amounts -> median
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.medians = {}
        self.max_days_since = {}
        
    def fit(self, X, y=None):
        # Flatten all customer portfolios to compute global statistics
        # X is a list of 2D numpy arrays (n_loans, n_features)
        all_loans = np.vstack(X)
        
        for i, col in enumerate(self.feature_names):
            col_data = all_loans[:, i]
            
            # Learn medians for amounts and ratios
            if "AMNT" in col or "RATIO" in col:
                valid_data = col_data[~np.isnan(col_data)]
                self.medians[i] = np.median(valid_data) if len(valid_data) > 0 else 0
                
            # Learn max + 1 for days_since
            elif col.startswith("DAYS_SINCE"):
                valid_data = col_data[~np.isnan(col_data)]
                self.max_days_since[i] = np.max(valid_data) + 1 if len(valid_data) > 0 else 9999
                
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        X_out = []
        for instance in X:
            # Copy to avoid modifying original
            inst = np.copy(instance)
            
            for i, col in enumerate(self.feature_names):
                mask = np.isnan(inst[:, i])
                if not np.any(mask):
                    continue
                    
                if "AMNT" in col or "RATIO" in col:
                    inst[mask, i] = self.medians.get(i, 0)
                elif col.startswith("DAYS_SINCE"):
                    inst[mask, i] = self.max_days_since.get(i, 9999)
                else:
                    # Trajectory, flags, counts default to 0
                    inst[mask, i] = 0
            
            X_out.append(inst)
        return X_out


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips features at 1st and 99th percentile."""
    def __init__(self, feature_names, binary_features):
        self.feature_names = feature_names
        self.binary_features = binary_features
        self.percentiles = {}
        
    def fit(self, X, y=None):
        all_loans = np.vstack(X)
        
        for i, col in enumerate(self.feature_names):
            if col in self.binary_features:
                continue
            
            col_data = all_loans[:, i]
            if "RATIO" in col:
                self.percentiles[i] = (0.0, 1.0)
            else:
                p1 = np.percentile(col_data, 1)
                p99 = np.percentile(col_data, 99)
                self.percentiles[i] = (p1, p99)
                
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        X_out = []
        for instance in X:
            inst = np.copy(instance)
            for i, bounds in self.percentiles.items():
                inst[:, i] = np.clip(inst[:, i], bounds[0], bounds[1])
            X_out.append(inst)
        return X_out


class PortfolioRobustScaler(BaseEstimator, TransformerMixin):
    """Applies RobustScaler to non-binary columns."""
    def __init__(self, feature_names, binary_features):
        self.feature_names = feature_names
        self.binary_features = binary_features
        
        self.scale_indices = [
            i for i, col in enumerate(self.feature_names) 
            if col not in self.binary_features
        ]
        self.scaler = RobustScaler()
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        if not self.scale_indices:
            return self
            
        all_loans = np.vstack(X)
        self.scaler.fit(all_loans[:, self.scale_indices])
        return self
        
    def transform(self, X):
        if not self.scale_indices:
            return X
            
        X_out = []
        for instance in X:
            inst = np.copy(instance)
            # transform takes 2D array, we scale just the selected columns
            inst[:, self.scale_indices] = self.scaler.transform(inst[:, self.scale_indices])
            X_out.append(inst)
        return X_out


def create_preprocessing_pipeline(feature_names, binary_features):
    """Creates the full sklearn Pipeline for preprocessing portfolio instances."""
    return Pipeline([
        ('imputer', DomainAwareImputer(feature_names)),
        ('clipper', OutlierClipper(feature_names, binary_features)),
        ('scaler', PortfolioRobustScaler(feature_names, binary_features))
    ])
