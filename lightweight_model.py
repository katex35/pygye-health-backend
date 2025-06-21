import numpy as np
import joblib
from collections import Counter

class SimpleDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                
                left_gini = self._gini_impurity(y[left_mask])
                right_gini = self._gini_impurity(y[right_mask])
                
                weighted_gini = (np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini) / len(y)
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def _predict_sample(self, sample, tree):
        if not isinstance(tree, dict):
            return tree
        
        if sample[tree['feature']] <= tree['threshold']:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])
    
    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

class SimpleRandomForest:
    def __init__(self, n_estimators=20, max_depth=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            final_predictions.append(Counter(votes).most_common(1)[0][0])
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        probabilities = []
        
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            prob_1 = np.sum(votes == 1) / len(votes)
            prob_0 = 1 - prob_1
            probabilities.append([prob_0, prob_1])
        
        return np.array(probabilities)

class SimpleStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1, self.std)
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

if __name__ == "__main__":
    # Test with simple data
    print("Testing lightweight model implementation...")
    
    # Create some test data
    np.random.seed(42)
    X_test = np.random.randn(100, 5)
    y_test = np.random.randint(0, 2, 100)
    
    # Test scaler
    scaler = SimpleStandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    print(f"Scaler test - Mean: {np.mean(X_scaled, axis=0)}")
    print(f"Scaler test - Std: {np.std(X_scaled, axis=0)}")
    
    # Test model
    model = SimpleRandomForest(n_estimators=5, max_depth=3)
    model.fit(X_scaled, y_test)
    predictions = model.predict(X_scaled[:10])
    probabilities = model.predict_proba(X_scaled[:10])
    
    print(f"Model test - Predictions: {predictions}")
    print(f"Model test - Probabilities shape: {probabilities.shape}")
    print("Lightweight model implementation successful!") 