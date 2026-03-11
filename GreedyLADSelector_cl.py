import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from ConsistencyChecker_cl import check_consistency, remove_conflicting_rows


class GreedyLADSelector:
    def __init__(self):
        self.best_subset = []
        # Cleaned training data after fallback conflict removal — available to pipeline
        self.X_clean = None
        self.y_clean = None

    def fit(self, X, y, bin_feature_names):
        self.bin_feature_names = bin_feature_names
        n = X.shape[1]
        remaining = list(range(n))
        self.best_subset = []
        best_overall = 0

        # Track best discriminating subset found so far
        best_consistent_score = -1
        best_consistent_subset = None

        print("\n=== Greedy Feature Selection Started ===")

        while remaining:
            best_f = None
            best_score = best_overall

            for f in remaining:
                subset = self.best_subset + [f]
                score = self.evaluate_subset(X, y, subset)
                if score > best_score:
                    best_score = score
                    best_f = f

            if best_f is None:
                print("No improvement. Stopping.")
                break

            self.best_subset.append(best_f)
            remaining.remove(best_f)
            best_overall = best_score
            print(f"[Overall]    Added: {bin_feature_names[best_f]} → CV Score = {best_score:.4f}")

            # Check consistency independently — note it but never stop early
            if check_consistency(X, y, self.best_subset):
                if best_score > best_consistent_score:
                    best_consistent_score = best_score
                    best_consistent_subset = self.best_subset[:]
                    print(f"[Consistent] New best discriminating: {len(self.best_subset)} feature(s), "
                          f"score={best_consistent_score:.4f}")

        # --- Final selection ---
        if best_consistent_subset is not None:
            self.best_subset = best_consistent_subset
            self.X_clean = X
            self.y_clean = y
            print(f"\n[Result] Using best discriminating subset: "
                  f"{len(self.best_subset)} features, score={best_consistent_score:.4f}")
        else:
            print(
                f"\n[ConsistencyWarning] No fully discriminating subset found. "
                f"Falling back to best overall subset "
                f"({len(self.best_subset)} features, score={best_overall:.4f}) "
                f"and removing conflicting rows."
            )
            # Fix #1: actually capture the returned cleaned data
            self.X_clean, self.y_clean, _ = remove_conflicting_rows(
                X, y, self.best_subset, verbose=True
            )

        print(f"Final selected {len(self.best_subset)} features.\n")
        return self

    def evaluate_subset(self, X, y, subset):
        if len(subset) == 0:
            return 0
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        return np.mean(cross_val_score(clf, X[:, subset], y, cv=3, scoring='accuracy'))

    def transform(self, X):
        return X[:, self.best_subset]

    def fit_transform(self, X, y, bin_feature_names):
        self.fit(X, y, bin_feature_names)
        return self.transform(X)