from sklearn.model_selection import train_test_split
from MaxPatterns import *
from AStarFeatureSelector import *
from GreedyLADSelector import *
from DecisionTreeCutpointBinarizer import *
from DecisionTreeCutpointBinarizerV2 import *
from LazyPatterns import *
from Eager1 import *
import time
import pickle
import numpy as np
import pandas as pd
from MutualInfoGreedySelector import *
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix
)

# Report Function
def evaluate(mp, X_train_sel, y_train, X_test_sel, y_test, model_name="LAD"):
    print("="*90)
    print("                 LAD EVALUATION")
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

# Evaluation Function

def evaluate_rules_strongest_partial(mp, X_test_selected, y_test):
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


# MAIN PIPELINE – WITH TRAIN/TEST SPLIT + STRATIFICATION

# 1. Load data
df = pd.read_csv("Heart_disease_cleveland_new.csv")
print("Dataset loaded:", df.shape)
print("Features:", df.columns[:-1].tolist())
print("Target distribution (full):")
print(df.iloc[:, -1].value_counts(normalize=True).sort_index())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.astype(int)
feature_names = df.columns[:-1].tolist()

# STRATIFIED TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,          # 75% train, 25% test
    random_state=42,            #so that everytime the same data is selected for training when u run program again and again 
    stratify=y               # THIS keeps class balance in both train and test split
)

print("\n" + "="*50)
print("STRATIFIED SPLIT COMPLETED")
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print("Train class distribution:")
print(pd.Series(y_train).value_counts(normalize=True).sort_index())
print("Test class distribution:")
print(pd.Series(y_test).value_counts(normalize=True).sort_index())
print("="*50 + "\n")

# FIT ENTIRE LAD PIPELINE ONLY ON TRAINING DATA

print("=== FITTING ON TRAINING DATA ONLY ===\n")

# Step 1: Binarization (only on train!)
binarizer = DecisionTreeCutpointBinarizerV2(mode="greedy", max_depth=4, min_samples_leaf=10)
Xbin_train = binarizer.fit_transform(X_train, y_train, feature_names=feature_names)
binarizer.print_cutpoints_readable()

# Create readable names for binary features
bin_feature_names = [
    f"{feature_names[f_idx]} <= {thr:.4f}"
    for cut_id, (f_idx, thr) in binarizer.cutpoints.items()
]

print(f"Binarized training matrix: {Xbin_train.shape}\n")

# # Step 2: Feature Selection (only on train!)

selector = AStarFeatureSelector()                  
Xsel_train = selector.fit(Xbin_train, y_train, bin_feature_names=bin_feature_names).transform(Xbin_train)
# Paper-based minimal feature selection
# selector = MutualInfoGreedySelector(max_features=15, cv=5)

# Xsel_train = selector.fit_transform(Xbin_train, y_train)

print("\nSelected feature indices:", selector.selected_features_)
print("Total selected:", len(selector.selected_features_))
# Step 3: Rule Mining (only on train!)
mp = MaxPatterns(binarizer=binarizer, selector=selector, purity=1.0, verbose=True, threshold = 0)
# mp = Eager1(binarizer=binarizer, selector=selector, purity=1.0, verbose=True, threshold=0)

mp.fit(Xsel_train, y_train, original_feature_names=feature_names)

print(f"\nTOP {len(mp.rules)} RULES DISCOVERED ON TRAINING DATA:")
mp.print_rules(top_n=len(mp.rules))

# TRANSFORM TEST DATA USING SAME CUTPOINTS & SELECTED FEATURES

# Apply same binarization
Xbin_test = binarizer.transform(X_test)                    # uses learned cutpoints
# Apply same feature selection
Xsel_test = Xbin_test[:, selector.best_subset]                # same columns!

print(f"\nTest matrix after binarization + selection: {Xsel_test.shape}")

preds, scores, reasons = evaluate_rules_strongest_partial(mp, Xsel_test, y_test)

results = evaluate(
    mp=mp,
    X_train_sel=Xsel_train,
    y_train=y_train,
    X_test_sel=Xsel_test,
    y_test=y_test,
    model_name="LAD Heart Disease (Final)"
)