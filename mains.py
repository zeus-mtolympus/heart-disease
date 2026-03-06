# main.py

from sklearn.model_selection import train_test_split
from MaxPatterns import *
from AStarFeatureSelector import *
from DecisionTreeCutpointBinarizerV2 import *
from LazyPatterns import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

# --------------------------- DATA LOADING ---------------------------
df = pd.read_csv("Heart_disease_cleveland_new.csv")
print("Dataset loaded:", df.shape)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.astype(int)
feature_names = df.columns[:-1].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# --------------------------- BINARIZATION ---------------------------
print("\n=== BINARIZATION ===")
binarizer = DecisionTreeCutpointBinarizerV2(mode="greedy", max_depth=4, min_samples_leaf=10)
Xbin_train = binarizer.fit_transform(X_train, y_train, feature_names=feature_names)
binarizer.print_cutpoints_readable()

Xbin_test = binarizer.transform(X_test)
print(f"Binarized shapes -> train {Xbin_train.shape} | test {Xbin_test.shape}\n")

# --------------------------- 1. LAZY MODEL ---------------------------
print("="*70)
print("1. LAZY MODEL (instance-specific rules)")
print("="*70)

lazy = LazyPatterns(binarizer=binarizer, purity=0.75, min_support=3, verbose=True)
lazy.fit(Xbin_train, y_train, original_feature_names=feature_names)

lazy_results = lazy.predict(Xbin_test)

lazy_preds = [r["label"] for r in lazy_results]
print(f"\nLAZY TEST ACCURACY : {accuracy_score(y_test, lazy_preds):.4f}")
print(f"Balanced Accuracy   : {balanced_accuracy_score(y_test, lazy_preds):.4f}")
print(f"F1-score            : {f1_score(y_test, lazy_preds):.4f}")
print(f"Exact-match rules   : {sum(r['exact_match'] for r in lazy_results)/len(lazy_results):.1%}\n")

# Show some explanations
print("First 15 lazy justifications:")
for i, r in enumerate(lazy_results[:15]):
    mark = "CORRECT" if r["label"] == y_test[i] else "WRONG"
    rule_str = " AND ".join(r["rule"][:7]) + (" ..." if len(r["rule"]) > 7 else "")
    print(f"{i+1:2d}. True={y_test[i]} Pred={r['label']} [{mark}] | "
          f"Purity={r['purity']:.3f} | Supp={r['support']} -> {rule_str}")

# --------------------------- 2. EAGER MODEL ---------------------------

# ============================ LAZY PATTERNS FULL EVALUATION ============================

import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix
)

def evaluate_lazy(lazy_model, X_train_bin, y_train, X_test_bin, y_test, model_name="LazyPatterns"):
    print("="*90)
    print("                 LAZY PATTERNS EVALUATION")
    print("="*90)

    # === Model "size" – we store the whole training set + binarizer → realistic size
    model_pickle = pickle.dumps({
        "binarizer": lazy_model.binarizer,
        "X_train_bin": lazy_model.X_train_bin,
        "y_train": lazy_model.y_train,
        "bin_names": lazy_model.bin_names
    })
    model_size_kb = len(model_pickle) / 1024

    # === Prediction on test (with timing & confidence) ===
    start = time.time()
    results = lazy_model.predict(X_test_bin)                  # list of dicts
    pred_time_ms = (time.time() - start) / len(X_test_bin) * 1000

    y_pred = np.array([r["label"] for r in results])
    confidences = np.array([r["purity"] for r in results])    # purity = confidence
    exact_matches = np.array([r["exact_match"] for r in results])

    # Train predictions (to measure overfitting)
    train_results = lazy_model.predict(X_train_bin)
    y_train_pred = np.array([r["label"] for r in train_results])

    # === Metrics ===
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test,  y_pred)
    overfitting_gap = train_acc - test_acc

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    auc     = roc_auc_score(y_test, confidences)   # purity as probability-like score
    exact_cov = exact_matches.mean()

    # === Confusion Matrix ===
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # === FINAL REPORT ===
    print(f"Model                 : {model_name}")
    print(f"Stored Training Rows  : {len(y_train)}")
    print(f"Binary Features       : {X_train_bin.shape[1]}")
    print(f"Model Size (pickle)   : {model_size_kb:.1f} KB   ← includes full training data!")
    print(f"Prediction Speed      : {pred_time_ms:.2f} ms per sample")
    print(f"Exact Rule Coverage   : {exact_cov:.1%} ({exact_matches.sum()}/{len(y_test)} patients fully explained)")
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
    print(f"ROC-AUC (purity)      : {auc:.4f}")
    print()
    print("FULL CONFUSION MATRIX")
    print("                        Predicted →")
    print("                          No Disease (0)    Disease (1)")
    print(f"True No Disease (0) →       {tn:4d}             {fp:4d}    ← FP: Healthy wrongly told sick")
    print(f"True Disease    (1) →       {fn:4d}             {tp:4d}    ← FN: Sick patient missed")
    print()
    print(f"→ {fp} healthy patients incorrectly predicted as having disease (False Positive)")
    print(f"→ {fn} patients with disease incorrectly predicted as healthy (False Negative) ← more dangerous in medicine")
    print(f"Correct predictions   : {tn + tp}/{len(y_test)} = {test_acc:.1%}")

    # === ONE-ROW SUMMARY TABLE (same format as eager model) ===
    summary = pd.DataFrame([{
        "Model": model_name,
        "Rules": "Lazy (per-instance)",
        "Cutpoints": len(lazy_model.binarizer.cutpoints),
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

    print("\n" + "="*90)
    print("ONE-ROW SUMMARY")
    print("="*90)
    print(summary.to_string(index=False))

    # Optional: Show first 20 explanations
    print(f"\nFirst 20 Lazy Justifications:")
    for i, r in enumerate(results[:20]):
        mark = "CORRECT" if r["label"] == y_test[i] else "WRONG"
        rule_str = " AND ".join(r["rule"]) if isinstance(r["rule"], list) else str(r["rule"])
        if len(rule_str) > 100:
            rule_str = rule_str[:97] + "..."
        print(f"{i+1:2d}. True={y_test[i]} Pred={r['label']} [{mark}] | "
              f"Purity={r['purity']:.3f} | Supp={r['support']:3d} | {rule_str}")

    return summary

# =====================================================================================
# === INSERT THIS AFTER YOU TRAIN THE LAZY MODEL ===
# =====================================================================================

print("\n" + "="*70)
print("LAZY PATTERNS – FULL DETAILED EVALUATION")
print("="*70)

lazy = LazyPatterns(binarizer=binarizer, purity=0.75, min_support=3, verbose=True)
lazy.fit(Xbin_train, y_train, original_feature_names=feature_names)

lazy_summary = evaluate_lazy(
    lazy_model=lazy,
    X_train_bin=Xbin_train,
    y_train=y_train,
    X_test_bin=Xbin_test,
    y_test=y_test,
    model_name="LazyPatterns Heart Disease"
)