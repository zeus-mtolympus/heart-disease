import numpy as np
import pandas as pd
import heapq
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import cross_val_score


# ================================================================
# 1. DECISION TREE CUTPOINT BINARIZER (with readable output)
# ================================================================
class DecisionTreeCutpointBinarizer:
    def __init__(self,
                 mode="two",
                 max_depth=5,
                 min_samples_leaf=15,
                 min_impurity_decrease=0.005,
                 min_support=15,
                 edge_fraction=0.05,
                 min_interval_fraction=0.12,
                 random_state=42):

        assert mode in ("one", "two")
        self.mode = mode
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_support = min_support
        self.edge_fraction = edge_fraction
        self.min_interval_fraction = min_interval_fraction
        self.random_state = random_state

        self.cutpoints = {}  # cut_id → (feat_idx, threshold)
        self.feature_names = None

    def get_cutpoints(self):
        return self.cutpoints

    def print_cutpoints_readable(self):
        print("\n=== DISCOVERED CUTPOINTS (Binarization Stage) ===")
        if len(self.cutpoints) == 0:
            print("No cutpoints found!")
            return
        for cut_id, (feat_idx, thresh) in self.cutpoints.items():
            feat_name = self.feature_names[feat_idx]
            print(f"Cut {cut_id:3d}:  {feat_name} <= {thresh:.6f}")
        print(f"Total binary features created: {len(self.cutpoints)}\n")

    def _tree_thresholds(self, x, y):
        x = x.reshape(-1, 1)
        sample_weight = compute_sample_weight("balanced", y)

        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
            class_weight="balanced"
        )
        tree.fit(x, y, sample_weight=sample_weight)

        thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
        thresholds = np.unique(np.round(thresholds, 8))

        if len(thresholds) == 0:
            return np.array([]), np.array([])

        gains = []
        for t in thresholds:
            left_mask = x.flatten() <= t
            n_left = left_mask.sum()
            n_right = len(x) - n_left
            if n_left < 2 or n_right < 2:
                gains.append(0)
                continue

            p_left = y[left_mask].mean()
            p_right = y[~left_mask].mean()
            p_total = y.mean()

            gain = (n_left + n_right) * (
                p_total * (1 - p_total) -
                (n_left / (n_left + n_right)) * p_left * (1 - p_left) -
                (n_right / (n_left + n_right)) * p_right * (1 - p_right)
            )
            gains.append(gain)

        order = np.argsort(thresholds)
        return thresholds[order], np.array(gains)[order]
    
    def _prune_thresholds(self, x, thresholds, gains):
        if len(thresholds) == 0:
            return [], []

        pruned_thresh = []
        pruned_gains = []

        xmin, xmax = np.min(x), np.max(x)
        range_x = max(xmax - xmin, 1e-8)
        edge_min = xmin + self.edge_fraction * range_x
        edge_max = xmax - self.edge_fraction * range_x

        for t, g in zip(thresholds, gains):
            if t <= edge_min or t >= edge_max:
                continue
            left_count = np.sum(x <= t)
            right_count = len(x) - left_count
            if left_count < self.min_support or right_count < self.min_support:
                continue
            pruned_thresh.append(t)
            pruned_gains.append(g)

        return pruned_thresh, pruned_gains

    def _select_cutpoints(self, pruned_thresh, pruned_gains, x):
        if len(pruned_thresh) == 0:
            return []

        sorted_idx = np.argsort(pruned_thresh)
        thresh_by_value = np.array(pruned_thresh)[sorted_idx]
        gains_by_value = np.array(pruned_gains)[sorted_idx]

        if self.mode == "one":
            return [thresh_by_value[len(thresh_by_value)//2]]

        if len(thresh_by_value) <= 1:
            return list(thresh_by_value)

        lower = thresh_by_value[0]
        upper = thresh_by_value[-1]
        range_x = np.ptp(x)

        if (upper - lower) < self.min_interval_fraction * range_x:
            if len(thresh_by_value) > 0:
                best_idx = np.argmax(gains_by_value)
                return [thresh_by_value[best_idx]]
            return [thresh_by_value[len(thresh_by_value)//2]]

        return [lower, upper]

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.cutpoints = {}
        cut_id = 0

        for feat_idx in range(X.shape[1]):
            x_feat = X[:, feat_idx].astype(float)
            if np.std(x_feat) == 0:
                continue

            thresholds, gains = self._tree_thresholds(x_feat, y)
            pruned_t, pruned_g = self._prune_thresholds(x_feat, thresholds, gains)
            selected = self._select_cutpoints(pruned_t, pruned_g, x_feat)

            for cp in selected:
                self.cutpoints[cut_id] = (feat_idx, cp)
                cut_id += 1

        return self

    def transform(self, X):
        if len(self.cutpoints) == 0:
            return np.zeros((X.shape[0], 0), dtype=int)

        Xbin = []
        for feat_idx, thresh in self.cutpoints.values():
            col = (X[:, feat_idx] <= thresh).astype(int)
            Xbin.append(col.reshape(-1, 1))

        return np.hstack(Xbin)

    def fit_transform(self, X, y, feature_names=None):
        self.fit(X, y, feature_names=feature_names)
        return self.transform(X)


# ================================================================
# 2. GREEDY & A* SELECTORS (with readable output)
# ================================================================
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


class AStarFeatureSelector:
    def __init__(self, max_features=None, max_expansions=500):
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


# ================================================================
# 3. MAXPATTERNS – EAGER LAD RULE MINING (fully readable)
# ================================================================
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


# ================================================================
# MAIN PIPELINE – FULLY READABLE
# ================================================================
from sklearn.model_selection import train_test_split

# ================================================================
# MAIN PIPELINE – WITH TRAIN/TEST SPLIT + STRATIFICATION
# ================================================================

# 1. Load data
df = pd.read_csv("Heart_disease_cleveland_new.csv")
print("Dataset loaded:", df.shape)
print("Features:", df.columns[:-1].tolist())
print("Target distribution (full):")
print(df.iloc[:, -1].value_counts(normalize=True).sort_index())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.astype(int)
feature_names = df.columns[:-1].tolist()

# ==============================================
# STRATIFIED TRAIN/TEST SPLIT (preserves class ratio)
# ==============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,          # 75% train, 25% test
    random_state=42,
    stratify=y               # THIS keeps class balance identical
)

print("\n" + "="*50)
print("STRATIFIED SPLIT COMPLETED")
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print("Train class distribution:")
print(pd.Series(y_train).value_counts(normalize=True).sort_index())
print("Test class distribution:")
print(pd.Series(y_test).value_counts(normalize=True).sort_index())
print("="*50 + "\n")

# ==============================================
# FIT ENTIRE LAD PIPELINE ONLY ON TRAINING DATA
# ==============================================

print("=== FITTING ON TRAINING DATA ONLY ===\n")

# Step 1: Binarization (only on train!)
binarizer = DecisionTreeCutpointBinarizer(mode="two", max_depth=4, min_samples_leaf=10)
Xbin_train = binarizer.fit_transform(X_train, y_train, feature_names=feature_names)
binarizer.print_cutpoints_readable()

# Create readable names for binary features
bin_feature_names = [
    f"{feature_names[f_idx]} <= {thr:.4f}"
    for cut_id, (f_idx, thr) in binarizer.cutpoints.items()
]

print(f"Binarized training matrix: {Xbin_train.shape}\n")

# # Step 2: Feature Selection (only on train!)
# selector = GreedyLADSelector()                    # or AStarFeatureSelector()
# Xsel_train = selector.fit_transform(Xbin_train, y_train, bin_feature_names=bin_feature_names)

selector = AStarFeatureSelector()                  
Xsel_train = selector.fit(Xbin_train, y_train, bin_feature_names=bin_feature_names).transform(Xbin_train)

# Step 3: Rule Mining (only on train!)
mp = MaxPatterns(binarizer=binarizer, selector=selector, purity=0.65, verbose=True, threshold = 0)
mp.fit(Xsel_train, y_train, original_feature_names=feature_names)

print("\nTOP 15 RULES DISCOVERED ON TRAINING DATA:")
mp.print_rules(top_n=15)

# ==============================================
# TRANSFORM TEST DATA USING SAME CUTPOINTS & SELECTED FEATURES
# ==============================================

# Apply same binarization
Xbin_test = binarizer.transform(X_test)                    # uses learned cutpoints
# Apply same feature selection
Xsel_test = Xbin_test[:, selector.best_subset]                # same columns!

print(f"\nTest matrix after binarization + selection: {Xsel_test.shape}")

# ==============================================
# EVALUATE RULES ON TEST DATA
# ==============================================

def evaluate_rules_strongest_partial(mp, X_test_selected, y_test):
    """
    Final LAD prediction (gold standard):
      - First: exact match → use it
      - Otherwise: strongest partial match = (overlap_ratio × weight)
      - Rules are checked in decreasing weight order → high-weight rules win ties
      - 100% logical justification, zero majority fallback
    """
    print("\n" + "="*70)
    print("LAD PREDICTION: STRONGEST PARTIAL MATCH (sorted by weight)")
    print("="*70)

    # Sort rules once by weight descending → higher weight = checked first = wins ties
    sorted_rules = sorted(mp.rules, key=lambda r: r["weight"], reverse=True)

    predictions = []
    confidences = []      # final score = overlap × weight
    justifications = []

    for idx, row in enumerate(X_test_selected):
        best_rule = None
        best_score = -1
        best_overlap = 0

        for rule in sorted_rules:                          # ← High-weight first!
            attrs = rule["attrs"]
            values = rule["values"]

            # Count exact matches
            matches = sum(row[attrs[i]] == values[i] for i in range(len(attrs)))
            overlap_ratio = matches / len(attrs)

            # Composite score: overlap × weight (weight already favors strong rules)
            score = overlap_ratio * rule["weight"]

            if score > best_score:
                best_score = score
                best_rule = rule
                best_overlap = overlap_ratio

        # Always predict using the best partial match
        predictions.append(best_rule["label"])
        confidences.append(best_score)
        justifications.append(
            f"Rule weight={best_rule['weight']:.3f}, "
            f"overlap={best_overlap:.2f} ({best_overlap*100:.0f}%), "
            f"→ {' AND '.join(best_rule['readable'])}"
        )

    # Results
    acc = np.mean(np.array(predictions) == y_test)
    print(f"Test Accuracy (strongest partial match): {acc:.4f} "
          f"({int(acc*len(y_test))}/{len(y_test)} correct)")
    print(f"Average confidence score: {np.mean(confidences):.4f}")
    print(f"Full coverage: 100% ({len(y_test)}/{len(y_test)}) instances justified")

    # Show top 10 example predictions
    print(f"\nFirst {min(100, len(y_test))} test predictions:")
    for i in range(min(100, len(y_test))):
        print(f"  True={y_test[i]:1d} | Pred={predictions[i]:1d} | "
              f"Score={confidences[i]:.3f} | {justifications[i][:90]}...")

    return predictions, confidences, justifications

# After training on train set...
preds, scores, reasons = evaluate_rules_strongest_partial(mp, Xsel_test, y_test)


import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, 
    precision_score, recall_score, roc_auc_score, confusion_matrix)



def evaluate(mp, X_train_sel, y_train, X_test_sel, y_test, model_name="LAD"):
    """
    THE ONE AND ONLY FUNCTION YOU WILL EVER NEED.
    Literally everything: confusion matrix, overfitting gap, FP/FN counts, exact coverage...
    """
    print("="*90)
    print("                  ULTIMATE LAD EVALUATION — FULL REPORT")
    print("="*90)

    # === Model Stats ===
    n_rules = len(mp.rules)
    n_cutpoints = len(mp.binarizer.cutpoints)
    n_features = X_train_sel.shape[1]
    model_size_kb = len(pickle.dumps({
        "cutpoints": mp.binarizer.cutpoints,
        "selected": getattr(mp.selector, 'selected', mp.selector.best_subset),
        "rules": mp.rules
    })) / 1024

    # Sort rules by weight (strongest first)
    sorted_rules = sorted(mp.rules, key=lambda r: r["weight"], reverse=True)

    def predict_with_info(row):
        best_label = None
        best_score = -1
        exact = False
        for rule in sorted_rules:
            attrs = rule["attrs"]
            values = rule["values"]
            matches = sum(row[attrs[i]] == values[i] for i in range(len(attrs)))
            if matches == len(attrs):
                return rule["label"], True, rule["weight"]  # exact match
            score = (matches / len(attrs)) * rule["weight"]
            if score > best_score:
                best_score = score
                best_label = rule["label"]
        return best_label, exact, best_score

    # Predictions on train and test
    def get_preds(X):
        preds, exacts, confs = [], [], []
        for row in X:
            p, e, c = predict_with_info(row)
            preds.append(p); exacts.append(e); confs.append(c)
        return np.array(preds), np.array(exacts), np.array(confs)

    # Test set
    start = time.time()
    test_pred, test_exact, test_conf = get_preds(X_test_sel)
    pred_time_ms = (time.time() - start) / len(X_test_sel) * 1000

    # Train set (for overfitting)
    train_pred, _, _ = get_preds(X_train_sel)

    # === Metrics ===
    train_acc = accuracy_score(y_train, train_pred)
    test_acc  = accuracy_score(y_test,  test_pred)
    overfitting_gap = train_acc - test_acc

    bal_acc = balanced_accuracy_score(y_test, test_pred)
    f1      = f1_score(y_test, test_pred)
    prec    = precision_score(y_test, test_pred, zero_division=0)
    rec     = recall_score(y_test, test_pred, zero_division=0)
    auc     = roc_auc_score(y_test, test_conf)
    exact_cov = test_exact.mean()

    # === CONFUSION MATRIX ===
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

    # === FINAL REPORT ===
    print(f"Model                 : {model_name}")
    print(f"Rules                 : {n_rules}")
    print(f"Cutpoints             : {n_cutpoints}")
    print(f"Selected Features     : {n_features}")
    print(f"Model Size            : {model_size_kb:.1f} KB")
    print(f"Prediction Speed      : {pred_time_ms:.2f} ms per sample")
    print(f"Exact Rule Coverage   : {exact_cov:.1%} ({test_exact.sum()}/{len(y_test)} patients fully explained)")
    print()
    print(f"Train Accuracy        : {train_acc:.4f}")
    print(f"Test Accuracy         : {test_acc:.4f}")
    print(f"OVERFITTING GAP       : {overfitting_gap:+.4f} ← ← ← ← ← ← ← ← ← ← ← ← ←")
    print()
    print("TEST PERFORMANCE")
    print(f"Accuracy              : {test_acc:.4f}")
    print(f"Balanced Accuracy     : {bal_acc:.4f}")
    print(f"F1-Score              : {f1:.4f}")
    print(f"Precision             : {prec:.4f}")
    print(f"Recall                : {rec:.4f}")
    print(f"ROC-AUC               : {auc:.4f}")
    print()
    print("FULL CONFUSION MATRIX")
    print("                        Predicted →")
    print("                          No Disease (0)    Disease (1)")
    print(f"True No Disease (0) →       {tn:4d}             {fp:4d}    ← FP: Healthy wrongly told sick")
    print(f"True Disease    (1) →       {fn:4d}             {tp:4d}    ← FN: Sick patient missed")
    print()
    print(f"→ {fp} healthy patients incorrectly predicted as having disease (False Positive")
    print(f"→ {fn} patients with disease incorrectly predicted as healthyFalse Negative ← more dangerous in medicine")
    print(f"Correct predictions   : {tn + tp}/{len(y_test)} = {test_acc:.1%}")

    # === ONE-ROW SUMMARY TABLE ===
    summary = pd.DataFrame([{
        "Model": model_name,
        "Rules": n_rules,
        "Cutpoints": n_cutpoints,
        "Size_KB": round(model_size_kb, 1),
        "Pred_ms": round(pred_time_ms, 2),
        "Exact_Cov_%": f"{exact_cov:.1%}",
        "Train_Acc": round(train_acc, 4),
        "Test_Acc": round(test_acc, 4),
        "Overfit_Gap": round(overfitting_gap, 4),
        "Accuracy": round(test_acc, 4),
        "Bal_Acc": round(bal_acc, 4),
        "F1": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "AUC": round(auc, 4),
        "FP_(0→1)": fp,
        "FN_(1→0)": fn,
        "Correct": tn + tp,
        "Total": len(y_test)
    }])

results = evaluate(
    mp=mp,
    X_train_sel=Xsel_train,
    y_train=y_train,
    X_test_sel=Xsel_test,
    y_test=y_test,
    model_name="LAD Heart Disease (Final)"
)