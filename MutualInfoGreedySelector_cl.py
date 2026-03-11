import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from ConsistencyChecker_cl import check_consistency, remove_conflicting_rows


class MutualInfoGreedySelector:
    def __init__(self, cv=5):
        self.cv = cv
        # Cleaned training data after fallback conflict removal — available to pipeline
        self.X_clean = None
        self.y_clean = None

    def fit(self, X, y):
        # Step 1: Rank all features by mutual information with target
        mi = mutual_info_classif(X, y, discrete_features=True)
        ranked = np.argsort(mi)[::-1]

        self.best_subset = []
        best_overall = 0

        # Track best discriminating subset found so far
        best_consistent_score = -1
        best_consistent_subset = None

        print("\n=== MutualInfo Greedy Feature Selection Started ===")
        print(f"Feature ranking by MI (best first): {ranked.tolist()}\n")

        # Step 2: Walk down ranked list, keep feature if accuracy improves
        for f in ranked:
            trial = self.best_subset + [f]

            clf = LogisticRegression(max_iter=1000)
            score = np.mean(cross_val_score(clf, X[:, trial], y, cv=self.cv))

            if score >= best_overall:
                self.best_subset.append(f)
                best_overall = score
                print(f"[Overall]    Added feature {f} → CV Score = {best_overall:.4f}")

                # Check consistency independently — note it but never stop early
                if check_consistency(X, y, self.best_subset):
                    if score > best_consistent_score:
                        best_consistent_score = score
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

    def transform(self, X):
        return X[:, self.best_subset]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)