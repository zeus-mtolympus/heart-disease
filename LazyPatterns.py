# LazyPatterns.py
import numpy as np


class LazyPatterns:
    """
    Lazy Rule Learner – per-test-instance rule generation using all binarized features.
    Completely lazy: no rules are mined in advance.
    """
    def __init__(self, binarizer, purity=0.75, min_support=3, verbose=False):
        self.binarizer = binarizer
        self.purity = purity
        self.min_support = min_support
        self.verbose = verbose

    def fit(self, X_train_bin, y_train, original_feature_names):
        self.X_train_bin = X_train_bin.astype(np.int8)
        self.y_train = y_train.astype(np.int8)
        self.original_feature_names = original_feature_names

        # Human-readable names for every binary column
        self.bin_names = [
            f"{original_feature_names[f_idx]} <= {thr:.4f}"
            for cut_id, (f_idx, thr) in self.binarizer.cutpoints.items()
        ]

        if self.verbose:
            print(f"[LazyPatterns] Stored {self.X_train_bin.shape[0]} training rows "
                  f"and {self.X_train_bin.shape[1]} binary features.")
        return self

    def _make_readable(self, row_bin, mask):
        """Turn a binary row + mask into a list of readable conditions."""
        conds = []
        for i in np.where(mask)[0]:
            if row_bin[i] == 1:
                conds.append(self.bin_names[i])
            else:
                conds.append(f"NOT {self.bin_names[i]}")
        return conds

    def predict_single(self, test_row_bin):
        # 1. Exact match first (fast path)
        matches = np.all(self.X_train_bin == test_row_bin, axis=1)
        idx_exact = np.where(matches)[0]

        if len(idx_exact) >= self.min_support:
            labels, counts = np.unique(self.y_train[idx_exact], return_counts=True)
            best_label = labels[np.argmax(counts)]
            pur = counts.max() / len(idx_exact)
            if pur >= self.purity:
                return {
                    "label": int(best_label),
                    "purity": float(pur),
                    "support": int(len(idx_exact)),
                    "exact_match": True,
                    "rule": self._make_readable(test_row_bin, np.ones_like(test_row_bin, dtype=bool))
                }

        # 2. Bottom-up greedy pattern building
        active_cols = np.arange(len(test_row_bin))
        current_cols = []
        current_vals = []
        best_purity = 0.0

        while len(current_cols) < len(active_cols):
            best_col = None
            best_trial_purity = best_purity

            for col in active_cols:
                if col in current_cols:
                    continue
                trial_cols = current_cols + [col]
                trial_vals = current_vals + [test_row_bin[col]]
                mask = np.all(self.X_train_bin[:, trial_cols] == trial_vals, axis=1)
                covered = np.where(mask)[0]

                if len(covered) < self.min_support:
                    continue

                y_cov = self.y_train[covered]
                trial_purity = np.max(np.bincount(y_cov)) / len(covered)

                if trial_purity > best_trial_purity and trial_purity >= self.purity:
                    best_trial_purity = trial_purity
                    best_col = col
                    best_label_temp = np.argmax(np.bincount(y_cov))
                    best_support_temp = len(covered)

            if best_col is None:
                break

            current_cols.append(best_col)
            current_vals.append(test_row_bin[best_col])
            best_purity = best_trial_purity
            best_label = best_label_temp
            best_support = best_support_temp

        if best_purity >= self.purity:
            mask = np.zeros(len(test_row_bin), dtype=bool)
            mask[current_cols] = True
            return {
                "label": int(best_label),
                "purity": float(best_purity),
                "support": int(best_support),
                "exact_match": len(current_cols) == len(active_cols),
                "rule": self._make_readable(test_row_bin, mask)
            }

        # 3. Fallback – majority class
        maj = int(np.argmax(np.bincount(self.y_train)))
        return {
            "label": maj,
            "purity": 0.5,
            "support": len(self.y_train),
            "exact_match": False,
            "rule": ["<majority class fallback>"]
        }

    def predict(self, X_test_bin):
        """Predict on the whole test matrix."""
        return [self.predict_single(row) for row in X_test_bin]