#astar +eager+lazy


import time
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
    def __init__(self, max_features=None, max_expansions=200):
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
        print(label_support)
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
# 4. LazyPatterns (LAZY LAD) – fully compatible with Greedy & A*
# ================================================================
class LazyPatterns:
    def __init__(self, purity=0.8):
        self.min_purity = purity
        self.Xsel = None
        self.y = None
        self.colnames = None

    def fit(self, Xsel, y):
        """
        Xsel must be the SELECTED binary matrix (after Greedy/A*)
        It can be a NumPy array or DataFrame.
        """
        if isinstance(Xsel, pd.DataFrame):
            self.Xsel = Xsel.values.astype(int)
            self.colnames = list(Xsel.columns)
        else:
            self.Xsel = Xsel.astype(int)
            self.colnames = [f"f{i}" for i in range(Xsel.shape[1])]

        self.y = y.astype(int)
        return self

    def predict_instance(self, x, print_rule=False):
        attributes = list(range(len(x)))
        label_votes = {}
        matched_rules = []

        def recurse(attrs):
            if not attrs:
                return
            mask = (self.Xsel[:, attrs] == x[attrs]).all(axis=1)
            covered = np.where(mask)[0]
            if len(covered) == 0:
                return

            unique, counts = np.unique(self.y[covered], return_counts=True)
            purities = counts / len(covered)
            best = np.argmax(purities)
            label = int(unique[best])
            purity = purities[best]

            # good rule
            if purity >= self.min_purity:
                label_votes[label] = label_votes.get(label, 0) + len(covered)

                if print_rule:
                    conds = [
                        f"{self.colnames[a]}={x[a]}"
                        for a in attrs
                    ]
                    matched_rules.append((label, conds, purity, len(covered)))
                return

            # otherwise, recurse by removing attributes
            for a in attrs:
                new_attrs = [t for t in attrs if t != a]
                recurse(new_attrs)

        recurse(attributes)

        # majority vote fallback
        if not label_votes:
            pred = int(np.bincount(self.y).argmax())
        else:
            pred = max(label_votes, key=label_votes.get)

        # print matched rules
        if print_rule and matched_rules:
            print("\nMatched Lazy Rules:")
            for lbl, conds, purity, support in matched_rules:
                print(f"  Label={lbl}, Conditions: {' AND '.join(conds)}, "
                      f"Purity={purity:.3f}, Support={support}")

        return pred

    def predict(self, Xsel, print_rules=False):
        X = Xsel.values.astype(int) if isinstance(Xsel, pd.DataFrame) else Xsel
        return np.array([self.predict_instance(x, print_rule=print_rules) for x in X])

    # ================================================================
    # 5. Real-time Lazy LAD Predictor (fixed & compatible)
    # ================================================================
    @staticmethod
    def lazy_real_time_input_predict(lazy, binarizer, selector, X_train_df, original_feature_names):
        """
        lazy: LazyPatterns trained object
        binarizer: fitted DecisionTreeCutpointBinarizer
        selector: Greedy or A* selector object
        X_train_df: original X train (before binarization)
        original_feature_names: list of names BEFORE selection
        """

        # detect selected feature indices
        if hasattr(selector, "selected"):
            selected_idx = selector.selected
        elif hasattr(selector, "best_subset"):
            selected_idx = selector.best_subset
        else:
            raise ValueError("Selector missing 'selected' or 'best_features'")

        


        print("\n--- Real-time Lazy LAD prediction ---")
        print("Selected features used:", original_feature_names)

        while True:
            print("\nEnter feature values (or 'exit'):")
            user_vals = {}

            for f in original_feature_names:
                val = input(f"{f}: ").strip()
                if val.lower() == "exit":
                    print("Exiting real-time mode.")
                    return
                if val == "":
                    user_vals[f] = np.nan
                else:
                    try:
                        user_vals[f] = float(val)
                    except:
                        print("Invalid value, using NaN")
                        user_vals[f] = np.nan

            # fill missing with median
            filled = {}
            for f in original_feature_names:
                if np.isnan(user_vals[f]):
                    filled[f] = X_train_df[f].median()
                else:
                    filled[f] = user_vals[f]

            user_df = pd.DataFrame([filled])

            # binarize one-row input
            user_bin = binarizer.transform(user_df.values)

            # select only important features
            user_sel = user_bin[:, selected_idx]

            # Lazy LAD prediction
            pred = lazy.predict_instance(user_sel[0], print_rule=True)
            print(f"\nPredicted label: {pred}")

# ================================================================
# MAIN PIPELINE – FULLY READABLE
# ================================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score, confusion_matrix
import time

# ================================================================
# MAIN PIPELINE – WITH TRAIN/TEST SPLIT + FULL EVALUATION
# ================================================================

# 1. Load data
df = pd.read_csv("Heart_disease_cleveland_new.csv")
print("Dataset loaded:", df.shape)
print("Features:", df.columns[:-1].tolist())
print("Target distribution:\n", df.iloc[:, -1].value_counts())

feature_names = df.columns[:-1].tolist()
target_name = df.columns[-1]

X_raw = df[feature_names].copy()
y = df[target_name].values.astype(int)

# --- Train/Test Split ---
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train_df.shape[0]}, Test size: {X_test_df.shape[0]}")
print(f"Train class dist: {np.bincount(y_train)}")
print(f"Test class dist:  {np.bincount(y_test)}")

# Convert train to numpy for binarizer
X_train = X_train_df.values
X_test = X_test_df.values

print("\n=== STARTING FULL READABLE LAD PIPELINE (with Train/Test Evaluation) ===\n")

# Step 1: Binarization (fit on train only)
binarizer = DecisionTreeCutpointBinarizer(mode="two", max_depth=4, min_samples_leaf=10)
Xbin_train = binarizer.fit_transform(X_train, y_train, feature_names=feature_names)
binarizer.print_cutpoints_readable()

# Create readable names for all binary features
bin_feature_names = []
for cut_id, (f_idx, thr) in binarizer.cutpoints.items():
    bin_feature_names.append(f"{feature_names[f_idx]} <= {thr:.4f}")

print(f"Binarized train matrix shape: {Xbin_train.shape}")

# Transform test set using the same binarizer
Xbin_test = binarizer.transform(X_test)

# Step 2: Feature Selection (fit on train binarized)
# Choose one: Greedy or A*

# selector = GreedyLADSelector()
# Xsel_train = selector.fit_transform(Xbin_train, y_train, bin_feature_names=bin_feature_names)

selector = AStarFeatureSelector(max_features=15, max_expansions=300)
selector.fit(Xbin_train, y_train, bin_feature_names=bin_feature_names)
Xsel_train = selector.transform(Xbin_train)

# Apply same selection to test
if hasattr(selector, "selected"):
    selected_idx = selector.selected
else:
    selected_idx = selector.best_subset

Xsel_test = Xbin_test[:, selected_idx]

# Readable column names for selected features
selected_bin_names = [bin_feature_names[i] for i in selected_idx]
Xsel_train_df = pd.DataFrame(Xsel_train, columns=selected_bin_names)
Xsel_test_df = pd.DataFrame(Xsel_test, columns=selected_bin_names)

print(f"Selected {len(selected_idx)} binary features for modeling.\n")

# ================================================================
# 3. EAGER LAD: MaxPatterns (Rule Mining on Train)
# ================================================================
mp = MaxPatterns(binarizer=binarizer, selector=selector, purity=0.65, verbose=True, threshold=0)
start_time = time.time()
mp.fit(Xsel_train, y_train, original_feature_names=feature_names)
eager_fit_time = time.time() - start_time

mp.print_rules(top_n=20)

# Eager Prediction on Test Set
print("\n=== EAGER LAD PREDICTION ON TEST SET ===")
start_time = time.time()
# For eager: we need to implement a simple predict using best rules per class
# Simple voting: each rule votes with its weight, take majority
def eager_predict(X_test_sel, rules):
    preds = []
    for x in X_test_sel:
        votes = {}
        for rule in rules:
            match = all(x[attr_idx] == val for attr_idx, val in zip(rule["attrs"], rule["values"]))
            if match:
                lbl = rule["label"]
                votes[lbl] = votes.get(lbl, 0) + rule["weight"]
        if not votes:
            pred = 0  # fallback to majority class
        else:
            pred = max(votes.items(), key=lambda x: x[1])[0]
        preds.append(pred)
    return np.array(preds)

y_pred_eager = eager_predict(Xsel_test, mp.rules)
eager_pred_time = time.time() - start_time

# ================================================================
# 4. LAZY LAD: LazyPatterns
# ================================================================
lazy = LazyPatterns(purity=0.80)
start_time = time.time()
lazy.fit(Xsel_train_df, y_train)
lazy_fit_time = time.time() - start_time

print("\n=== LAZY LAD PREDICTION ON TEST SET ===")
start_time = time.time()
y_pred_lazy = lazy.predict(Xsel_test_df, print_rules=False)
lazy_pred_time = time.time() - start_time

# ================================================================
# 5. PERFORMANCE EVALUATION
# ================================================================
def print_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n=== {model_name} PERFORMANCE ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y_true, y_pred)
        print(f"AUC:       {auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

print_metrics(y_test, y_pred_eager, "EAGER LAD (MaxPatterns)")
print_metrics(y_test, y_pred_lazy,  "LAZY LAD")

# Time summary
print(f"\n=== TIME SUMMARY ===")
print(f"Eager Fit Time:     {eager_fit_time:.4f}s")
print(f"Eager Predict Time: {eager_pred_time:.4f}s")
print(f"Lazy Fit Time:      {lazy_fit_time:.4f}s")
print(f"Lazy Predict Time:  {lazy_pred_time:.4f}s")

# ================================================================
# Optional: Real-time prediction (uncomment if you want interactive input later)
# ================================================================
# print("\nStarting real-time prediction mode... (type 'exit' to stop)")
# LazyPatterns.lazy_real_time_input_predict(
#     lazy=lazy,
#     binarizer=binarizer,
#     selector=selector,
#     X_train_df=X_train_df,
#     original_feature_names=feature_names
# )

print("\n=== FULL PIPELINE WITH TRAIN/TEST EVALUATION COMPLETED SUCCESSFULLY ===")