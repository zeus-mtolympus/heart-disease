import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class GreedyLADSelector:
    def __init__(self):
        self.selected = []

    def fit(self, X, y, bin_feature_names):
        self.bin_feature_names = bin_feature_names
        n = X.shape[1]
        remaining = list(range(n))
        self.selected = []
        best_overall = 0

        print("\n=== Greedy Feature Selection Started ===")
        while remaining:
            best_f = None
            best_score = best_overall

            for f in remaining:
                subset = self.selected + [f]
                score = self.evaluate_subset(X, y, subset)
                if score > best_score:
                    best_score = score
                    best_f = f

            if best_f is None:
                print("No improvement. Stopping.")
                break

            self.selected.append(best_f)
            remaining.remove(best_f)
            print(f"Added: {self.bin_feature_names[best_f]} → CV Score = {best_score:.4f}")

        print(f"Final selected {len(self.selected)} features.\n")
        return self

    def evaluate_subset(self, X, y, subset):
        if len(subset) == 0:
            return 0
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        return np.mean(cross_val_score(clf, X[:, subset], y, cv=3, scoring='accuracy'))

    def transform(self, X):
        return X[:, self.selected]

    def fit_transform(self, X, y, bin_feature_names):
        self.fit(X, y, bin_feature_names)
        return self.transform(X)