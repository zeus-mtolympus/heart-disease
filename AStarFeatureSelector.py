import numpy as np
import heapq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class AStarFeatureSelector:
    def __init__(self, max_features=None, max_expansions=1110999):
        self.max_features = max_features
        self.max_expansions = max_expansions

    def fit(self, X, y, bin_feature_names):
        self.bin_feature_names = bin_feature_names
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features

        print("\n=== A* Feature Selection Started ===")
        pq = []
        heapq.heappush(pq, (0, []))  # (f_score, subset)
        visited = set()
        best_score = -1
        self.best_subset = []
        expansions = 0

        while pq and expansions < self.max_expansions:
            _, subset = heapq.heappop(pq)
            expansions += 1

            key = tuple(sorted(subset))
            if key in visited:
                continue
            visited.add(key)

            score = self.score_subset(X, y, subset)
            if score > best_score:
                best_score = score
                self.best_subset = subset[:]
                print(f"New best: {len(subset)} features, score={best_score:.4f}, features: {[bin_feature_names[i] for i in subset]}")
                print(expansions)

            if len(subset) >= self.max_features:
                continue

            for feat in range(n_features):
                if feat in subset:
                    continue
                new_subset = subset + [feat]
                g = len(new_subset)
                h = 1.0 - score
                f_new = g + 5 * h  # heuristic weight
                heapq.heappush(pq, (f_new, new_subset))

        print(f"\nA* finished after {expansions} expansions. Best score: {best_score:.4f}\n")
        return self

    def score_subset(self, X, y, subset):
        if not subset:
            return 0
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        return np.mean(cross_val_score(clf, X[:, subset], y, cv=3, scoring='accuracy'))

    def transform(self, X):
        return X[:, self.best_subset] if hasattr(self, 'best_subset') else X
