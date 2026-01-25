import numpy as np

class MaxPatterns:
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
            print("\n=== MaxPatterns Rule Mining Started ===")
            print(f"Selected binary matrix shape: {Xsel.shape}")

        Xn = Xsel.astype(int)
        # Create readable names for selected binary columns
        selected_cut_ids = self.selector.selected if hasattr(self.selector, 'selected') else self.selector.best_subset
        bin_names = []
        for cut_id in selected_cut_ids:
            feat_idx, thresh = self.binarizer.cutpoints[cut_id]
            bin_names.append(f"{self.original_feature_names[feat_idx]} <= {thresh:.4f}")

        rules_raw = []
        label_support = {}

        unique_instances, indices = np.unique(Xn, axis=0, return_index=True)

        for inst in unique_instances:
            attrs = list(range(len(inst)))
            repet, max_count, purity, label, _ = self._stats(Xn, y, inst, attrs)

            # Greedy attribute deletion
            while len(attrs) > 1 and purity >= self.min_purity:
                best_remove = None
                best_purity = purity

                for a in attrs:
                    trial_attrs = [t for t in attrs if t != a]
                    _, _, p2, _, _ = self._stats(Xn, y, inst, trial_attrs)
                    if p2 >= self.min_purity and p2 > best_purity:
                        best_purity = p2
                        best_remove = a

                if best_remove is None:
                    break
                attrs.remove(best_remove)
                repet, max_count, purity, label, _ = self._stats(Xn, y, inst, attrs)

            if purity < self.min_purity or len(attrs) == 0:
                continue

            rule = {
                "label": int(label),
                "attrs": attrs.copy(),
                "values": [int(inst[a]) for a in attrs],
                "purity": float(purity),
                "repet": int(repet),
                "readable": [bin_names[i] if v == 1 else f"NOT ({bin_names[i]})" for i, v in zip(attrs, [int(inst[a]) for a in attrs])]
            }
            if rule["repet"] > self.threshold:
                rules_raw.append(rule)
            label_support[label] = label_support.get(label, 0) + repet

        # Dedup & weight
        seen = {}
        for r in rules_raw:
            key = (r["label"], tuple(r["attrs"]), tuple(r["values"]))
            if key not in seen:
                seen[key] = r
            else:
                seen[key]["repet"] += r["repet"]

        self.rules = []
        for r in seen.values():
            denom = max(1, label_support.get(r["label"], 1))
            r["weight"] = r["repet"] / denom
            self.rules.append(r)

        self.rules.sort(key=lambda x: (-x["weight"], -x["purity"], x["label"]))

        if self.verbose:
            print(f"Generated {len(self.rules)} high-quality rules.\n")

    def _stats(self, Xn, y, inst, attrs):
        if len(attrs) == 0:
            return 0, 0, 0.0, 0, 1.0
        mask = np.all(Xn[:, attrs] == [inst[a] for a in attrs], axis=1)
        covered = np.where(mask)[0]
        repet = len(covered)
        if repet == 0:
            return 0, 0, 0.0, 0, 1.0
        labels, counts = np.unique(y[covered], return_counts=True)
        label = labels[np.argmax(counts)]
        purity = counts.max() / repet
        return repet, counts.max(), purity, int(label), 0.0

    def print_rules(self, top_n=20):
        print(f"\n=== TOP {top_n} RULES (sorted by weight) ===")
        for i, r in enumerate(self.rules[:top_n], 1):
            conds = r["readable"]
            print(f"Rule {i:2d}: IF {' AND '.join(conds)} "
                  f"THEN class = {r['label']} "
                  f"| Purity={r['purity']:.3f} | Support={r['repet']} | Weight={r['weight']:.3f}")
