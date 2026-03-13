import numpy as np
import heapq
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from ConsistencyChecker_cl import check_consistency, remove_conflicting_rows


class MutualInfoAStarSelector:
    """
    Two-phase feature selector:
      Phase 1 — Rank all binary features by Mutual Information with the target.
                This gives a focused candidate pool ordered by individual relevance.
      Phase 2 — Run A* search over that ranked pool.
                A* explores combinations guided by accuracy, so it can find
                synergistic subsets that pure MI ranking or greedy would miss.

    Tracks best_overall and best_consistent subsets independently throughout
    the search — never stops early. At the end picks best_consistent if found,
    otherwise falls back to best_overall and removes conflicting rows.
    """

    def __init__(self, max_expansions=100000, cv=3): #1110999
        self.max_expansions = max_expansions
        self.cv = cv
        # Cleaned training data after fallback conflict removal — available to pipeline
        self.X_clean = None
        self.y_clean = None

    def fit(self, X, y, bin_feature_names):
        self.bin_feature_names = bin_feature_names
        n_features = X.shape[1]

        # ── Phase 1: MI ranking ──────────────────────────────────────────────
        print("\n=== MutualInfo A* Feature Selection Started ===")
        print("Phase 1: Computing Mutual Information rankings...")

        mi = mutual_info_classif(X, y, discrete_features=True)
        ranked = np.argsort(mi)[::-1]   # indices sorted best MI first

        print(f"Top 10 features by MI: {ranked[:10].tolist()}")
        print(f"Candidate pool: all {n_features} features, A* explores in MI-ranked order\n")

        # ── Phase 2: A* over MI-ranked pool ─────────────────────────────────
        print("Phase 2: Running A* search...")

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

            score = self._score_subset(X, y, subset)

            # Update best overall
            if score > best_score:
                best_score = score
                best_subset_overall = subset[:]
                print(f"[Overall]    New best: {len(subset)} features, score={best_score:.4f}, "
                      f"features: {[bin_feature_names[i] for i in subset]}")

            # Check consistency — track independently, never stop early
            if subset and check_consistency(X, y, subset):
                if score > best_consistent_score:
                    best_consistent_score = score
                    best_consistent_subset = subset[:]
                    print(f"[Consistent] New best discriminating: {len(subset)} features, "
                          f"score={best_consistent_score:.4f}, "
                          f"features: {[bin_feature_names[i] for i in subset]}")

            if len(subset) >= n_features:
                continue

            # Expand in MI-ranked order; MI rank used as tie-breaker
            for rank, feat in enumerate(ranked):
                if feat in subset:
                    continue
                new_subset = subset + [feat]
                # Fix heuristic: normalise g so both terms are in [0, 1]
                g = len(new_subset) / n_features
                h = 1.0 - score
                rank_penalty = rank / n_features * 0.1
                f_new = g + 5 * h + rank_penalty
                heapq.heappush(pq, (f_new, new_subset))

        # ── Final selection ──────────────────────────────────────────────────
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

        print(f"MutualInfo A* finished after {expansions} expansions.\n")
        return self

    def _score_subset(self, X, y, subset):
        if not subset:
            return 0.0
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        return np.mean(cross_val_score(clf, X[:, subset], y, cv=self.cv, scoring='accuracy'))

    def transform(self, X):
        return X[:, self.best_subset] if self.best_subset else X

    def fit_transform(self, X, y, bin_feature_names):
        self.fit(X, y, bin_feature_names)
        return self.transform(X)