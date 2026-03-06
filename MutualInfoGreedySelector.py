import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class MutualInfoGreedySelector:
    def __init__(self, max_features=None, cv=5):
        self.max_features = max_features
        self.cv = cv

    def fit(self, X, y):
        n_features = X.shape[1]

        # Step 1: Mutual information with target
        mi = mutual_info_classif(X, y, discrete_features=True)

        # Step 2: Sort features by MI
        ranked = np.argsort(mi)[::-1]

        selected = []
        best_score = 0

        for f in ranked:
            trial = selected + [f]

            clf = LogisticRegression(max_iter=1000)
            score = np.mean(
                cross_val_score(clf, X[:, trial], y, cv=self.cv)
            )

            # Keep feature only if accuracy improves
            if score >= best_score:
                selected.append(f)
                best_score = score

            if self.max_features and len(selected) >= self.max_features:
                break

        self.selected_features_ = selected
        self.best_subset = selected      # compatibility with LAD pipeline
        self.selected = selected         # extra compatibility (optional but safer)
        return self


    def transform(self, X):
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
