"""
LAD Heart Disease Pipeline
==========================
Configure your choices below, then run. All pipeline logic lives in functions.
"""

# ================================================================
# CONFIGURATION — change these to switch algorithms
# ================================================================

DATA_PATH       = "Heart_disease_cleveland_new.csv"
TEST_SIZE       = 0.25
RANDOM_STATE    = 42

# Binarizer mode: "one" | "two" | "all" | "dense" | "greedy"
BINARIZER_MODE          = "greedy"
BINARIZER_MAX_DEPTH     = 4
BINARIZER_MIN_SAMPLES   = 10

# Feature Selector: "greedy" | "astar" | "mutualinfo" | "mutualinfo_astar"
SELECTOR = "astar"

# Pattern Miner: "maxpatterns" | "eager" | "genetic"
# Use PATTERN_MINER = "lazy" to run the lazy path instead (separate evaluate)
PATTERN_MINER   = "maxpatterns"
PURITY          = 0.95
THRESHOLD       = 0

# Genetic algorithm settings (only used when PATTERN_MINER = "genetic")
GA_GENERATIONS  = 1000
GA_POP_SIZE     = 200

# Lazy settings (only used when PATTERN_MINER = "lazy")
# LAZY_PURITY is separate from PURITY so eager and lazy can be tuned independently
# 1.0 purity is too strict for lazy — recommended range: 0.65–0.80
LAZY_PURITY      = 0.75
LAZY_MIN_SUPPORT = 3

# ================================================================
# IMPORTS
# ================================================================

import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix
)

from MaxPatterns_cl import MaxPatterns
from Eager_cl import Eager
from GeneticRuleMiner_cl import GeneticRuleMiner
from LazyPatterns_cl import LazyPatterns
from AStarFeatureSelector_cl import AStarFeatureSelector
from GreedyLADSelector_cl import GreedyLADSelector
from DecisionTreeCutpointBinarizerV2 import DecisionTreeCutpointBinarizerV2
from MutualInfoGreedySelector_cl import MutualInfoGreedySelector
from MutualInfoAStarSelector_cl import MutualInfoAStarSelector

# ================================================================
# PIPELINE FUNCTIONS
# ================================================================

def load_and_split_data(path, test_size, random_state):
    df = pd.read_csv(path)
    print("Dataset loaded:", df.shape)
    print("Features:", df.columns[:-1].tolist())
    print("Target distribution (full):")
    print(df.iloc[:, -1].value_counts(normalize=True).sort_index())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)
    feature_names = df.columns[:-1].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\n" + "="*50)
    print("STRATIFIED SPLIT COMPLETED")
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print("Train class distribution:")
    print(pd.Series(y_train).value_counts(normalize=True).sort_index())
    print("Test class distribution:")
    print(pd.Series(y_test).value_counts(normalize=True).sort_index())
    print("="*50 + "\n")

    return X_train, X_test, y_train, y_test, feature_names


def binarize(X_train, y_train, X_test, feature_names, mode, max_depth, min_samples):
    binarizer = DecisionTreeCutpointBinarizerV2(
        mode=mode, max_depth=max_depth, min_samples_leaf=min_samples
    )
    Xbin_train = binarizer.fit_transform(X_train, y_train, feature_names=feature_names)
    binarizer.print_cutpoints_readable()

    bin_feature_names = [
        f"{feature_names[f_idx]} <= {thr:.4f}"
        for cut_id, (f_idx, thr) in binarizer.cutpoints.items()
    ]
    print(f"Binarized training matrix: {Xbin_train.shape}\n")

    Xbin_test = binarizer.transform(X_test)
    return binarizer, Xbin_train, Xbin_test, bin_feature_names


def select_features(selector_name, Xbin_train, y_train, bin_feature_names):
    if selector_name == "greedy":
        selector = GreedyLADSelector()
        selector.fit(Xbin_train, y_train, bin_feature_names)
    elif selector_name == "astar":
        selector = AStarFeatureSelector()
        selector.fit(Xbin_train, y_train, bin_feature_names)
    elif selector_name == "mutualinfo":
        selector = MutualInfoGreedySelector()
        selector.fit(Xbin_train, y_train)
    elif selector_name == "mutualinfo_astar":
        selector = MutualInfoAStarSelector()
        selector.fit(Xbin_train, y_train, bin_feature_names)
    else:
        raise ValueError(
            f"Unknown selector: '{selector_name}'. "
            f"Choose from: greedy, astar, mutualinfo, mutualinfo_astar"
        )

    # Use cleaned data the selector already produced — no second consistency check
    Xsel_train     = selector.X_clean[:, selector.best_subset]
    y_train_clean  = selector.y_clean

    print(f"\nSelected feature indices: {selector.best_subset}")
    print(f"Total selected: {len(selector.best_subset)}")
    return selector, Xsel_train, y_train_clean


def mine_patterns(miner_name, binarizer, selector, Xsel_train, y_train,
                  feature_names, purity, threshold,
                  ga_generations=300, ga_pop_size=200):
    if miner_name == "maxpatterns":
        miner = MaxPatterns(binarizer=binarizer, selector=selector,
                            purity=purity, verbose=True, threshold=threshold)
    elif miner_name == "eager":
        miner = Eager(binarizer=binarizer, selector=selector,
                       purity=purity, verbose=True, threshold=threshold)
    elif miner_name == "genetic":
        miner = GeneticRuleMiner(binarizer=binarizer, selector=selector,
                                 purity=purity, verbose=True, threshold=threshold,
                                 n_generations=ga_generations, pop_size=ga_pop_size)
    else:
        raise ValueError(
            f"Unknown miner: '{miner_name}'. "
            f"Choose from: maxpatterns, eager1, genetic"
        )

    miner.fit(Xsel_train, y_train, original_feature_names=feature_names)

    if miner.rules:
        print(f"\nTOP {len(miner.rules)} RULES DISCOVERED ON TRAINING DATA:")
        miner.print_rules(top_n=len(miner.rules))
    else:
        print("\n[Warning] No rules were generated. "
              "Try lowering PURITY or THRESHOLD in the config.")

    return miner


def _predict_row(row, sorted_rules):
    """
    Single shared prediction function used by both print_predictions and evaluate.
    Checks exact match first (returns immediately), then falls back to strongest
    partial match scored as overlap_ratio * weight.
    Returns (label, is_exact, confidence_score).
    """
    best_label = None
    best_score = -1
    is_exact   = False

    for rule in sorted_rules:
        attrs, values = rule["attrs"], rule["values"]
        matches = sum(row[attrs[i]] == values[i] for i in range(len(attrs)))

        if matches == len(attrs):
            return rule["label"], True, rule["weight"]

        score = (matches / len(attrs)) * rule["weight"]
        if score > best_score:
            best_score = score
            best_label = rule["label"]

    return best_label, is_exact, best_score


def predict_all(miner, X_selected):
    """Run prediction on a full matrix. Returns arrays of labels, exact flags, scores."""
    if not miner.rules:
        raise RuntimeError("Miner has no rules. Cannot predict.")

    sorted_rules = sorted(miner.rules, key=lambda r: r["weight"], reverse=True)
    labels, exacts, scores = [], [], []

    for row in X_selected:
        label, exact, score = _predict_row(row, sorted_rules)
        labels.append(label)
        exacts.append(exact)
        scores.append(score)

    return np.array(labels), np.array(exacts), np.array(scores)


def print_predictions(miner, X_test_selected, y_test):
    """Print per-instance prediction results with justification."""
    print("\n" + "="*70)
    print("LAD PREDICTION: STRONGEST PARTIAL MATCH (sorted by weight)")
    print("="*70)

    sorted_rules = sorted(miner.rules, key=lambda r: r["weight"], reverse=True)
    predictions, confidences, justifications = [], [], []

    for row in X_test_selected:
        label, exact, score = _predict_row(row, sorted_rules)
        best_rule = next(
            (r for r in sorted_rules
             if r["label"] == label and
             sum(row[r["attrs"][i]] == r["values"][i]
                 for i in range(len(r["attrs"]))) / len(r["attrs"]) * r["weight"] == score),
            sorted_rules[0]
        )
        overlap = sum(
            row[best_rule["attrs"][i]] == best_rule["values"][i]
            for i in range(len(best_rule["attrs"]))
        ) / len(best_rule["attrs"])

        predictions.append(label)
        confidences.append(score)
        justifications.append(
            f"Rule weight={best_rule['weight']:.3f}, "
            f"overlap={overlap:.2f} ({overlap*100:.0f}%), "
            f"→ {' AND '.join(best_rule['readable'])}"
        )

    acc = np.mean(np.array(predictions) == y_test)
    print(f"Test Accuracy (strongest partial match): {acc:.4f} "
          f"({int(acc*len(y_test))}/{len(y_test)} correct)")
    print(f"Average confidence score: {np.mean(confidences):.4f}")
    print(f"Full coverage: 100% ({len(y_test)}/{len(y_test)}) instances justified")

    print(f"\nFirst {min(100, len(y_test))} test predictions:")
    for i in range(min(100, len(y_test))):
        print(f"  True={y_test[i]:1d} | Pred={predictions[i]:1d} | "
              f"Score={confidences[i]:.3f} | {justifications[i][:90]}...")

    return np.array(predictions), np.array(confidences), justifications


def evaluate(miner, X_train_sel, y_train, X_test_sel, y_test, model_name,
             selector_name, miner_name, purity, threshold):
    print("="*90)
    print("                 LAD EVALUATION")
    print("="*90)

    print("PIPELINE SETTINGS")
    print(f"  Selector          : {selector_name}")
    print(f"  Pattern Miner     : {miner_name}")
    print(f"  Purity threshold  : {purity}")
    print(f"  Support threshold : {threshold}")
    print()

    if not miner.rules:
        print("[Error] No rules available. Cannot evaluate.")
        return

    n_rules       = len(miner.rules)
    n_cutpoints   = len(miner.binarizer.cutpoints)
    n_features    = X_train_sel.shape[1]
    model_size_kb = len(pickle.dumps({
        "cutpoints": miner.binarizer.cutpoints,
        "selected":  miner.selector.best_subset,
        "rules":     miner.rules
    })) / 1024

    start = time.time()
    test_pred, test_exact, test_conf = predict_all(miner, X_test_sel)
    pred_time_ms = (time.time() - start) / len(X_test_sel) * 1000
    train_pred, _, _ = predict_all(miner, X_train_sel)

    train_acc       = accuracy_score(y_train, train_pred)
    test_acc        = accuracy_score(y_test,  test_pred)
    overfitting_gap = train_acc - test_acc
    bal_acc         = balanced_accuracy_score(y_test, test_pred)
    f1              = f1_score(y_test, test_pred)
    prec            = precision_score(y_test, test_pred, zero_division=0)
    rec             = recall_score(y_test, test_pred, zero_division=0)
    auc             = roc_auc_score(y_test, test_conf)
    exact_cov       = test_exact.mean()
    tn, fp, fn, tp  = confusion_matrix(y_test, test_pred).ravel()

    print(f"Model                 : {model_name}")
    print(f"Rules                 : {n_rules}")
    print(f"Cutpoints             : {n_cutpoints}")
    print(f"Selected Features     : {n_features}")
    print(f"Model Size            : {model_size_kb:.1f} KB")
    print(f"Prediction Speed      : {pred_time_ms:.2f} ms per sample")
    print(f"Exact Rule Coverage   : {exact_cov:.1%} "
          f"({test_exact.sum()}/{len(y_test)} patients fully explained)")
    print()
    print(f"Train Accuracy        : {train_acc:.4f}")
    print(f"Test Accuracy         : {test_acc:.4f}")
    print(f"OVERFITTING GAP       : {overfitting_gap:+.4f} ← ← ← ← ← ← ← ← ← ← ← ← ←")
    print()
    print("TEST PERFORMANCE")
    print(f"  Accuracy            : {test_acc:.4f}")
    print(f"  Balanced Accuracy   : {bal_acc:.4f}")
    print(f"  F1-Score            : {f1:.4f}")
    print(f"  Precision           : {prec:.4f}")
    print(f"  Recall              : {rec:.4f}")
    print(f"  ROC-AUC (ranking)   : {auc:.4f}  ← scores are overlap×weight, not probabilities")
    print()
    print("FULL CONFUSION MATRIX")
    print("                        Predicted →")
    print("                          No Disease (0)    Disease (1)")
    print(f"True No Disease (0) →       {tn:4d}             {fp:4d}"
          f"    ← FP: Healthy wrongly told sick")
    print(f"True Disease    (1) →       {fn:4d}             {tp:4d}"
          f"    ← FN: Sick patient missed")
    print()
    print(f"→ {fp} healthy patients incorrectly predicted as having disease (False Positive)")
    print(f"→ {fn} patients with disease incorrectly predicted as healthy "
          f"(False Negative ← more dangerous in medicine)")
    print(f"Correct predictions   : {tn + tp}/{len(y_test)} = {test_acc:.1%}")


def evaluate_lazy(lazy, X_train_sel, y_train, X_test_sel, y_test,
                  selector_name, purity, min_support):
    """
    Separate evaluation path for LazyPatterns.
    Includes full stage diagnostics: how many instances were resolved by
    exact match, greedy rule, or fell back to majority class.
    """
    print("="*90)
    print("                 LAD EVALUATION  —  LAZY PATTERNS")
    print("="*90)

    print("PIPELINE SETTINGS")
    print(f"  Selector          : {selector_name}")
    print(f"  Pattern Miner     : lazy")
    print(f"  Purity threshold  : {purity}")
    print(f"  Min support       : {min_support}")
    print()

    def run_lazy(X):
        results  = lazy.predict(X)
        labels   = np.array([r["label"]       for r in results])
        purities = np.array([r["purity"]       for r in results])
        supports = np.array([r["support"]      for r in results])
        exacts   = np.array([r["exact_match"]  for r in results])
        stages   = np.array([r["stage"]        for r in results])
        return labels, purities, supports, exacts, stages, results

    start = time.time()
    test_pred, test_pur, test_sup, test_exact, test_stages, test_results = run_lazy(X_test_sel)
    pred_time_ms = (time.time() - start) / len(X_test_sel) * 1000

    train_pred, _, _, _, _, _ = run_lazy(X_train_sel)

    # ── Stage breakdown ──────────────────────────────────────────────────────
    n_exact    = int(np.sum(test_stages == "exact"))
    n_pruned   = int(np.sum(test_stages == "pruned"))
    n_fallback = int(np.sum(test_stages == "fallback"))
    n_test     = len(y_test)

    # Accuracy per stage — tells you where errors are coming from
    exact_mask    = test_stages == "exact"
    pruned_mask   = test_stages == "pruned"
    fallback_mask = test_stages == "fallback"

    acc_exact    = (accuracy_score(y_test[exact_mask],    test_pred[exact_mask])
                    if n_exact    > 0 else float("nan"))
    acc_pruned   = (accuracy_score(y_test[pruned_mask],   test_pred[pruned_mask])
                    if n_pruned   > 0 else float("nan"))
    acc_fallback = (accuracy_score(y_test[fallback_mask], test_pred[fallback_mask])
                    if n_fallback > 0 else float("nan"))

    train_acc       = accuracy_score(y_train, train_pred)
    test_acc        = accuracy_score(y_test,  test_pred)
    overfitting_gap = train_acc - test_acc
    bal_acc         = balanced_accuracy_score(y_test, test_pred)
    f1              = f1_score(y_test, test_pred)
    prec            = precision_score(y_test, test_pred, zero_division=0)
    rec             = recall_score(y_test, test_pred, zero_division=0)
    tn, fp, fn, tp  = confusion_matrix(y_test, test_pred).ravel()

    # ── Stage diagnostics ────────────────────────────────────────────────────
    print("STAGE DIAGNOSTICS  (tells you where predictions are coming from)")
    print(f"  Stage 1 — Exact match : {n_exact:4d}/{n_test} "
          f"({n_exact/n_test:.1%})  accuracy={acc_exact:.4f}"
          if n_exact > 0 else
          f"  Stage 1 — Exact match : {n_exact:4d}/{n_test} ({n_exact/n_test:.1%})")
    print(f"  Stage 2 — Pruned rule : {n_pruned:4d}/{n_test} "
          f"({n_pruned/n_test:.1%})  accuracy={acc_pruned:.4f}"
          if n_pruned > 0 else
          f"  Stage 2 — Pruned rule : {n_pruned:4d}/{n_test} ({n_pruned/n_test:.1%})")
    print(f"  Stage 3 — Fallback    : {n_fallback:4d}/{n_test} "
          f"({n_fallback/n_test:.1%})  accuracy={acc_fallback:.4f}"
          if n_fallback > 0 else
          f"  Stage 3 — Fallback    : {n_fallback:4d}/{n_test} ({n_fallback/n_test:.1%})")
    print()

    # Diagnostic hint if fallback rate is high
    if n_fallback / n_test > 0.3:
        print(f"  [DiagnosticHint] {n_fallback/n_test:.1%} fallback rate is high.")
        print(f"  Suggestions:")
        print(f"    — Lower LAZY_PURITY (currently {purity}) — try 0.65 or 0.60")
        print(f"    — Lower LAZY_MIN_SUPPORT (currently {min_support}) — try 2")
        print(f"    — Use a binarizer with more features (denser cutpoints)")
        print()

    print(f"Prediction Speed      : {pred_time_ms:.2f} ms per sample")
    print(f"Avg Rule Purity       : {test_pur.mean():.4f}")
    print(f"Avg Rule Support      : {test_sup.mean():.1f}")
    print()
    print(f"Train Accuracy        : {train_acc:.4f}")
    print(f"Test Accuracy         : {test_acc:.4f}")
    print(f"OVERFITTING GAP       : {overfitting_gap:+.4f} ← ← ← ← ← ← ← ← ← ← ← ← ←")
    print()
    print("TEST PERFORMANCE")
    print(f"  Accuracy            : {test_acc:.4f}")
    print(f"  Balanced Accuracy   : {bal_acc:.4f}")
    print(f"  F1-Score            : {f1:.4f}")
    print(f"  Precision           : {prec:.4f}")
    print(f"  Recall              : {rec:.4f}")
    print()
    print("FULL CONFUSION MATRIX")
    print("                        Predicted →")
    print("                          No Disease (0)    Disease (1)")
    print(f"True No Disease (0) →       {tn:4d}             {fp:4d}"
          f"    ← FP: Healthy wrongly told sick")
    print(f"True Disease    (1) →       {fn:4d}             {tp:4d}"
          f"    ← FN: Sick patient missed")
    print()
    print(f"→ {fp} healthy patients incorrectly predicted as having disease (False Positive)")
    print(f"→ {fn} patients with disease incorrectly predicted as healthy "
          f"(False Negative ← more dangerous in medicine)")
    print(f"Correct predictions   : {tn + tp}/{len(y_test)} = {test_acc:.1%}")

    # ── Per-instance breakdown (reuse already-computed results, no second predict call) ──
    print(f"\nFirst {min(100, n_test)} lazy predictions:")
    for i in range(min(100, n_test)):
        r        = test_results[i]
        rule_str = " AND ".join(r["rule"])[:80]
        print(f"  True={y_test[i]} | Pred={r['label']} | Stage={r['stage']:8s} | "
              f"Purity={r['purity']:.2f} | Support={r['support']:4d} | "
              f"Rule: {rule_str}...")


# ================================================================
# ENTRY POINT — only function calls below
# ================================================================

def run_pipeline():
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data(
        DATA_PATH, TEST_SIZE, RANDOM_STATE
    )

    binarizer, Xbin_train, Xbin_test, bin_feature_names = binarize(
        X_train, y_train, X_test, feature_names,
        BINARIZER_MODE, BINARIZER_MAX_DEPTH, BINARIZER_MIN_SAMPLES
    )

    selector, Xsel_train, y_train = select_features(
        SELECTOR, Xbin_train, y_train, bin_feature_names
    )

    Xsel_test = Xbin_test[:, selector.best_subset]
    print(f"\nTest matrix after binarization + selection: {Xsel_test.shape}")

    # ── Lazy path ────────────────────────────────────────────────────────────
    if PATTERN_MINER == "lazy":
        lazy = LazyPatterns(
            binarizer=binarizer,
            selector=selector,
            purity=LAZY_PURITY,
            min_support=LAZY_MIN_SUPPORT,
            verbose=True
        )
        lazy.fit(Xsel_train, y_train, original_feature_names=feature_names)

        evaluate_lazy(
            lazy=lazy,
            X_train_sel=Xsel_train,
            y_train=y_train,
            X_test_sel=Xsel_test,
            y_test=y_test,
            selector_name=SELECTOR,
            purity=LAZY_PURITY,
            min_support=LAZY_MIN_SUPPORT
        )

    # ── Eager path (MaxPatterns / Eager1 / Genetic) ──────────────────────────
    else:
        miner = mine_patterns(
            PATTERN_MINER, binarizer, selector, Xsel_train, y_train,
            feature_names, PURITY, THRESHOLD,
            ga_generations=GA_GENERATIONS,
            ga_pop_size=GA_POP_SIZE
        )

        print_predictions(miner, Xsel_test, y_test)

        evaluate(
            miner=miner,
            X_train_sel=Xsel_train,
            y_train=y_train,
            X_test_sel=Xsel_test,
            y_test=y_test,
            model_name="LAD Heart Disease (Final)",
            selector_name=SELECTOR,
            miner_name=PATTERN_MINER,
            purity=PURITY,
            threshold=THRESHOLD
        )


run_pipeline()