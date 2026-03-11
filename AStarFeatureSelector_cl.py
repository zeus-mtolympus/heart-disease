import numpy as np
import heapq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from ConsistencyChecker_cl import check_consistency, remove_conflicting_rows


class AStarFeatureSelector:
    def __init__(self, max_features=None, max_expansions=10000): #1110999
        self.max_features = max_features
        self.max_expansions = max_expansions
        # Cleaned training data after fallback conflict removal — available to pipeline
        self.X_clean = None
        self.y_clean = None

    def fit(self, X, y, bin_feature_names):
        self.bin_feature_names = bin_feature_names
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features

        print("\n=== A* Feature Selection Started ===")
        pq = []
        heapq.heappush(pq, (0.0, []))
        visited = set()

        # Track best overall (highest accuracy, any subset)
        best_score = -1
        best_subset_overall = []

        # Track best discriminating subset (consistent + highest accuracy)
        best_consistent_score = -1
        best_consistent_subset = None

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

            # Update best overall
            if score > best_score:
                best_score = score
                best_subset_overall = subset[:]
                print(f"[Overall]      New best: {len(subset)} features, score={best_score:.4f}, "
                      f"features: {[bin_feature_names[i] for i in subset]}")

            # Check consistency — track independently, never stop early
            if subset and check_consistency(X, y, subset):
                if score > best_consistent_score:
                    best_consistent_score = score
                    best_consistent_subset = subset[:]
                    print(f"[Consistent]   New best discriminating: {len(subset)} features, "
                          f"score={best_consistent_score:.4f}, "
                          f"features: {[bin_feature_names[i] for i in subset]}")

            if len(subset) >= self.max_features:
                continue

            for feat in range(n_features):
                if feat in subset:
                    continue
                new_subset = subset + [feat]
                # Fix heuristic: normalise g so both terms are in [0, 1]
                g = len(new_subset) / n_features
                h = 1.0 - score
                f_new = g + 5 * h
                heapq.heappush(pq, (f_new, new_subset))

        # --- Final selection ---
        if best_consistent_subset is not None:
            self.best_subset = best_consistent_subset
            self.X_clean = X
            self.y_clean = y
            print(f"\n[Result] Using best discriminating subset: "
                  f"{len(self.best_subset)} features, score={best_consistent_score:.4f}")
        else:
            self.best_subset = best_subset_overall
            print(
                f"\n[ConsistencyWarning] No fully discriminating subset found after "
                f"{expansions} expansions. Falling back to best overall subset "
                f"({len(self.best_subset)} features, score={best_score:.4f}) "
                f"and removing conflicting rows."
            )
            # Fix #1: actually capture the returned cleaned data
            self.X_clean, self.y_clean, _ = remove_conflicting_rows(
                X, y, self.best_subset, verbose=True
            )

        print(f"A* finished after {expansions} expansions.\n")
        return self

    def score_subset(self, X, y, subset):
        if not subset:
            return 0
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        return np.mean(cross_val_score(clf, X[:, subset], y, cv=3, scoring='accuracy'))

    def transform(self, X):
        return X[:, self.best_subset] if hasattr(self, 'best_subset') else X