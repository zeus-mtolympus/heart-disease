import numpy as np
import itertools

class Eager1:
    def __init__(self, binarizer=None, selector=None, purity=0.6, verbose=True, threshold=0):
        self.min_purity = purity
        self.verbose = verbose
        self.binarizer = binarizer
        self.selector = selector
        self.rules = []
        self.threshold = threshold

    def fit(self, Xsel, y, original_feature_names):
        self.rules.clear()
        self.original_feature_names = original_feature_names

        if self.verbose:
            print("\n=== Eager1 (Paper-Style) Rule Mining Started ===")
            print(f"Selected binary matrix shape: {Xsel.shape}")

        Xn = Xsel.astype(int)

        # Build readable binary names
        selected_cut_ids = self.selector.selected if hasattr(self.selector, 'selected') else self.selector.best_subset
        bin_names = []
        for cut_id in selected_cut_ids:
            feat_idx, thresh = self.binarizer.cutpoints[cut_id]
            bin_names.append(f"{self.original_feature_names[feat_idx]} <= {thresh:.4f}")

        n_features = Xn.shape[1]
        label_support = {}
        candidate_rules = []

        # 🔥 TOP-DOWN SEARCH: small subsets first
        for size in range(1, n_features + 1):

            for attrs in itertools.combinations(range(n_features), size):

                # Check both value patterns: 0/1 combinations from data
                unique_rows = np.unique(Xn[:, attrs], axis=0)

                for pattern in unique_rows:

                    mask = np.all(Xn[:, attrs] == pattern, axis=1)
                    covered_idx = np.where(mask)[0]
                    repet = len(covered_idx)

                    if repet <= self.threshold:
                        continue

                    labels, counts = np.unique(y[covered_idx], return_counts=True)
                    label = labels[np.argmax(counts)]
                    purity = counts.max() / repet

                    if purity >= self.min_purity:

                        rule = {
                            "label": int(label),
                            "attrs": list(attrs),
                            "values": list(pattern),
                            "purity": float(purity),
                            "repet": int(repet),
                            "readable": [
                                bin_names[a] if v == 1 else f"NOT ({bin_names[a]})"
                                for a, v in zip(attrs, pattern)
                            ]
                        }

                        candidate_rules.append(rule)
                        label_support[label] = label_support.get(label, 0) + repet

        # Remove non-minimal rules (keep minimal pure ones)
        final_rules = []
        for r in candidate_rules:
            is_minimal = True
            for other in candidate_rules:
                if r == other:
                    continue
                if (set(other["attrs"]).issubset(set(r["attrs"])) and
                        other["label"] == r["label"] and
                        len(other["attrs"]) < len(r["attrs"])):
                    is_minimal = False
                    break
            if is_minimal:
                final_rules.append(r)

        # Compute weights
        self.rules = []
        for r in final_rules:
            denom = max(1, label_support.get(r["label"], 1))
            r["weight"] = r["repet"] / denom
            self.rules.append(r)

        # Sort
        self.rules.sort(key=lambda x: (-x["weight"], -x["purity"], x["label"]))

        if self.verbose:
            print(f"Generated {len(self.rules)} minimal pure rules.\n")

    def print_rules(self, top_n=20):
        print(f"\n=== TOP {top_n} RULES (sorted by weight) ===")
        for i, r in enumerate(self.rules[:top_n], 1):
            print(f"Rule {i:2d}: IF {' AND '.join(r['readable'])} "
                  f"THEN class = {r['label']} "
                  f"| Purity={r['purity']:.3f} | Support={r['repet']} | Weight={r['weight']:.3f}")
