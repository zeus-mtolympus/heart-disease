from flask import Flask, render_template, request

app = Flask(__name__)

# Rules from your output (adjusted for clarity and consistency)
rules = [
    ({"cp": 0, "ca": 0}, 1, 0.946, 56, 0.243, "NOT (cp ≤ 2.5800) AND NOT (ca ≤ 0.5800)"),
    ({"cp": 1, "thal": 0, "chol": 0}, 1, 0.933, 45, 0.196, "cp ≤ 2.5800 AND thal > 1.5800 AND chol > 219.5800"),
    ({"ca": 0, "thal": 0}, 1, 0.890, 91, 0.182, "ca ≤ 0.5800 AND thal > 1.5800"),
    ({"ca": 0, "thal": 0, "chol": 0}, 1, 0.927, 41, 0.178, "NOT (ca ≤ 0.5800) AND NOT (thal ≤ 1.5800) AND NOT (chol ≤ 219.5800)"),
    ({"cp": 1, "thal": 1, "thalach": 1}, 0, 0.902, 82, 0.164, "cp ≤ 2.5800 AND thal ≤ 1.5800 AND thalach > 132.5800"),
    ({"cp": 1, "oldpeak": 0}, 0, 0.897, 78, 0.156, "cp ≤ 2.5800 AND oldpeak > 0.897"),
    ({"cp": 1, "ca": 1, "thalach": 1}, 0, 0.897, 78, 0.156, "cp ≤ 2.5800 AND ca ≤ 0.5800 AND thalach > 132.5800"),
    ({"cp": 1, "ca": 1, "thal": 1, "thalach": 1}, 0, 0.948, 58, 0.116, "cp ≤ 2.5800 AND ca ≤ 0.5800 AND thal ≤ 1.5800 AND thalach > 132.5800"),
    ({"cp": 0, "ca": 0, "thal": 0, "thalach": 0}, 1, 0.962, 26, 0.113, "NOT (cp ≤ 2.5800) AND NOT (ca ≤ 0.5800) AND NOT (thal ≤ 1.5800) AND thalach ≤ 132.5800"),
    ({"ca": 0, "thal": 0, "thalach": 0}, 1, 0.952, 21, 0.091, "NOT (ca ≤ 0.5800) AND NOT (thal ≤ 1.5800) AND thalach ≤ 132.5800"),
]

def evaluate_rules(features):
    best_score = -1
    best_class = None
    best_rule = "No strong matching rule found"
    best_is_exact = False

    for conds, label, purity, support, weight, desc in rules:
        matches = sum(1 for k, v in conds.items() if features.get(k) == v)
        total = len(conds)
        score = (matches / total) * weight if total > 0 else 0

        if score > best_score:
            best_score = score
            best_class = label
            best_rule = f"Rule: {desc} (Purity: {purity:.1%}, Support: {support}, Weight: {weight:.3f})"
            best_is_exact = (matches == total)

    if best_class is None:
        best_class = 0
        best_rule = "Fallback: majority class (low risk)"

    return best_class, round(best_score, 3), best_rule, best_is_exact


@app.route('/', methods=['GET', 'POST'])
def home():
    result = {}
    if request.method == 'POST':
        try:
            form = request.form
            raw = {
                'age': float(form.get('age', 0)),
                'sex': int(form.get('sex', -1)),
                'cp': int(form.get('cp', -1)),
                'trestbps': float(form.get('trestbps', 0)),
                'chol': float(form.get('chol', 0)),
                'fbs': int(form.get('fbs', -1)),
                'restecg': int(form.get('restecg', -1)),
                'thalach': float(form.get('thalach', 0)),
                'exang': int(form.get('exang', -1)),
                'oldpeak': float(form.get('oldpeak', 0)),
                'slope': int(form.get('slope', -1)),
                'ca': float(form.get('ca', -1)),
                'thal': int(form.get('thal', -1)),
            }

            # Convert to binary literals used by rules
            binarized = {
                'cp':      1 if raw['cp']     <= 2.58 else 0,
                'ca':      1 if raw['ca']     <= 0.58 else 0,
                'thal':    1 if raw['thal']   <= 1.58 else 0,
                'chol':    1 if raw['chol']   <= 219.58 else 0,
                'thalach': 1 if raw['thalach'] <= 132.58 else 0,
                'oldpeak': 1 if raw['oldpeak'] <= 0.897 else 0,
            }

            cls, conf, rule, exact = evaluate_rules(binarized)

            result = {
                'class': "High risk (disease likely)" if cls == 1 else "Low risk (disease unlikely)",
                'confidence': conf,
                'rule': rule,
                'exact': exact,
                'success': True
            }
        except Exception as e:
            result = {'success': False, 'error': str(e)}

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)