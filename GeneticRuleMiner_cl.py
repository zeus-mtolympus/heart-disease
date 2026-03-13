import numpy as np


class GeneticRuleMiner:
    """
    Genetic Algorithm Rule Miner for LAD.

    Each individual is a candidate rule:
        - attrs  : list of feature indices used in the rule
        - values : binary value (0 or 1) for each attr
        - label  : predicted class (0 or 1)

    Fitness = purity * coverage_fraction
        - purity          : fraction of covered training rows with the correct label
        - coverage_fraction: fraction of same-label training rows covered by the rule

    This directly optimises the LAD objective — rules should be pure AND cover
    as many same-label instances as possible.

    Pipeline interface:
        - Produces self.rules in the same format as MaxPatterns/Eager1
        - Plugs into evaluate() and predict_all() unchanged
    """

    def __init__(self,
                 binarizer=None,
                 selector=None,
                 purity=0.6,
                 n_generations=300,
                 pop_size=200,
                 min_attrs=1,
                 max_attrs=None,
                 tournament_size=5,
                 crossover_rate=0.7,
                 mutation_rate=0.15,
                 elite_frac=0.1,
                 threshold=0,
                 verbose=True,
                 random_state=42):

        self.binarizer       = binarizer
        self.selector        = selector
        self.purity          = purity
        self.n_generations   = n_generations
        self.pop_size        = pop_size
        self.min_attrs       = min_attrs
        self.max_attrs       = max_attrs
        self.tournament_size = tournament_size
        self.crossover_rate  = crossover_rate
        self.mutation_rate   = mutation_rate
        self.elite_frac      = elite_frac
        self.threshold       = threshold
        self.verbose         = verbose
        self.random_state    = random_state
        self.rules           = []

    # ─────────────────────────────────────────────
    # Individual encoding / decoding
    # ─────────────────────────────────────────────

    def _random_individual(self, n_features):
        """
        An individual is encoded as a dict:
            attrs  : sorted list of feature indices (1..max_attrs of them)
            values : binary value per attr
            label  : 0 or 1
        """
        max_a = self.max_attrs or max(1, n_features // 2)
        size  = self.rng.integers(self.min_attrs, max_a + 1)
        attrs = sorted(self.rng.choice(n_features, size=size, replace=False).tolist())
        values = self.rng.integers(0, 2, size=len(attrs)).tolist()
        label  = int(self.rng.integers(0, 2))
        return {"attrs": attrs, "values": values, "label": label}

    # ─────────────────────────────────────────────
    # Fitness
    # ─────────────────────────────────────────────

    def _fitness(self, ind, Xn, y, label_counts):
        attrs, values, label = ind["attrs"], ind["values"], ind["label"]
        if not attrs:
            return 0.0

        mask    = np.all(Xn[:, attrs] == values, axis=1)
        covered = np.where(mask)[0]
        repet   = len(covered)

        if repet <= self.threshold:
            return 0.0

        correct  = np.sum(y[covered] == label)
        purity   = correct / repet

        if purity < self.purity:
            return 0.0

        # coverage_fraction: how much of the same-label pool does this rule cover?
        total_label = label_counts.get(label, 1)
        coverage    = correct / total_label

        return purity * coverage

    # ─────────────────────────────────────────────
    # Selection
    # ─────────────────────────────────────────────

    def _tournament_select(self, population, fitnesses):
        idxs = self.rng.choice(len(population), size=self.tournament_size, replace=False)
        best = max(idxs, key=lambda i: fitnesses[i])
        return population[best]

    # ─────────────────────────────────────────────
    # Crossover
    # ─────────────────────────────────────────────

    def _crossover(self, p1, p2, n_features):
        if self.rng.random() > self.crossover_rate:
            return dict(p1), dict(p2)

        # Combine attr pools from both parents, then split randomly
        all_attrs  = sorted(set(p1["attrs"]) | set(p2["attrs"]))
        if len(all_attrs) < 2:
            return dict(p1), dict(p2)

        split = self.rng.integers(1, len(all_attrs))
        attrs1 = all_attrs[:split]
        attrs2 = all_attrs[split:]

        def make_child(attrs, ref1, ref2, label):
            # Inherit value from whichever parent had that attr; random if neither
            ref_vals = {}
            for a, v in zip(ref1["attrs"], ref1["values"]):
                ref_vals[a] = v
            for a, v in zip(ref2["attrs"], ref2["values"]):
                if a not in ref_vals:
                    ref_vals[a] = v
            values = [ref_vals.get(a, int(self.rng.integers(0, 2))) for a in attrs]
            return {"attrs": attrs, "values": values, "label": label}

        label1 = p1["label"] if self.rng.random() < 0.5 else p2["label"]
        label2 = p2["label"] if self.rng.random() < 0.5 else p1["label"]

        c1 = make_child(attrs1 or [all_attrs[0]], p1, p2, label1)
        c2 = make_child(attrs2 or [all_attrs[-1]], p1, p2, label2)
        return c1, c2

    # ─────────────────────────────────────────────
    # Mutation
    # ─────────────────────────────────────────────

    def _mutate(self, ind, n_features):
        ind = {"attrs": ind["attrs"][:], "values": ind["values"][:], "label": ind["label"]}
        max_a = self.max_attrs or max(1, n_features // 2)

        for i in range(len(ind["attrs"])):
            if self.rng.random() < self.mutation_rate:
                # Flip this value
                ind["values"][i] = 1 - ind["values"][i]

        if self.rng.random() < self.mutation_rate:
            # Swap one attr for a new random one
            all_attrs = set(range(n_features))
            unused    = list(all_attrs - set(ind["attrs"]))
            if unused:
                swap_pos = int(self.rng.integers(0, len(ind["attrs"])))
                new_attr = int(self.rng.choice(unused))
                ind["attrs"][swap_pos]  = new_attr
                ind["attrs"]  = sorted(ind["attrs"])

        if self.rng.random() < self.mutation_rate:
            # Add or remove one attr
            if len(ind["attrs"]) > self.min_attrs and self.rng.random() < 0.5:
                remove_pos = int(self.rng.integers(0, len(ind["attrs"])))
                ind["attrs"].pop(remove_pos)
                ind["values"].pop(remove_pos)
            elif len(ind["attrs"]) < max_a:
                unused = list(set(range(n_features)) - set(ind["attrs"]))
                if unused:
                    new_attr = int(self.rng.choice(unused))
                    insert_pos = int(np.searchsorted(ind["attrs"], new_attr))
                    ind["attrs"].insert(insert_pos, new_attr)
                    ind["values"].insert(insert_pos, int(self.rng.integers(0, 2)))

        if self.rng.random() < self.mutation_rate:
            ind["label"] = 1 - ind["label"]

        return ind

    # ─────────────────────────────────────────────
    # Rule conversion
    # ─────────────────────────────────────────────

    def _to_rule(self, ind, Xn, y, bin_names, label_counts):
        attrs, values, label = ind["attrs"], ind["values"], ind["label"]
        mask    = np.all(Xn[:, attrs] == values, axis=1)
        covered = np.where(mask)[0]
        repet   = len(covered)

        if repet == 0:
            return None

        correct = int(np.sum(y[covered] == label))
        purity  = correct / repet

        if purity < self.purity or repet <= self.threshold:
            return None

        total_label = label_counts.get(label, 1)
        weight      = correct / total_label

        readable = [
            bin_names[a] if v == 1 else f"NOT ({bin_names[a]})"
            for a, v in zip(attrs, values)
        ]

        return {
            "label":    label,
            "attrs":    attrs,
            "values":   values,
            "purity":   float(purity),
            "repet":    repet,
            "weight":   float(weight),
            "readable": readable
        }

    # ─────────────────────────────────────────────
    # Main fit
    # ─────────────────────────────────────────────

    def fit(self, Xsel, y, original_feature_names):
        self.rules.clear()
        self.rng = np.random.default_rng(self.random_state)

        Xn         = Xsel.astype(int)
        n_features = Xn.shape[1]
        if self.verbose:
            print("\n=== Genetic Rule Miner Started ===")
            print(f"Selected binary matrix: {Xn.shape} | "
                  f"Generations: {self.n_generations} | Pop size: {self.pop_size}")

        # Readable names for selected columns
        selected_cut_ids = self.selector.best_subset
        bin_names = []
        for cut_id in selected_cut_ids:
            feat_idx, thresh = self.binarizer.cutpoints[cut_id]
            bin_names.append(f"{original_feature_names[feat_idx]} <= {thresh:.4f}")

        # Per-label counts for fitness and weight calculation
        labels_unique, counts_unique = np.unique(y, return_counts=True)
        label_counts = dict(zip(labels_unique.tolist(), counts_unique.tolist()))

        # ── Initialise population ────────────────────────────────────────────
        population = np.array(
            [self._random_individual(n_features) for _ in range(self.pop_size)],
            dtype=object
        )
        fitnesses  = np.array([self._fitness(ind, Xn, y, label_counts)
                                for ind in population])

        n_elite = max(1, int(self.elite_frac * self.pop_size))
        best_fitness_ever = -1.0
        best_gen          = 0

        # ── Evolution loop ───────────────────────────────────────────────────
        for gen in range(self.n_generations):
            elite_idx  = np.argsort(fitnesses)[-n_elite:]
            elites     = population[elite_idx].tolist()

            new_population = elites[:]

            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2, n_features)
                c1 = self._mutate(c1, n_features)
                c2 = self._mutate(c2, n_features)
                new_population.extend([c1, c2])

            population = np.array(new_population[:self.pop_size], dtype=object)
            fitnesses  = np.array([self._fitness(ind, Xn, y, label_counts)
                                    for ind in population])

            gen_best = fitnesses.max()
            if gen_best > best_fitness_ever:
                best_fitness_ever = gen_best
                best_gen          = gen

            if self.verbose and (gen + 1) % 50 == 0:
                print(f"  Gen {gen+1:4d}/{self.n_generations} | "
                      f"Best fitness: {gen_best:.4f} | "
                      f"Mean fitness: {fitnesses.mean():.4f} | "
                      f"Non-zero: {(fitnesses > 0).sum()}/{self.pop_size}")

        if self.verbose:
            print(f"\nEvolution complete. Best fitness {best_fitness_ever:.4f} "
                  f"first seen at generation {best_gen + 1}.")

        # ── Convert final population to rules ────────────────────────────────
        seen  = {}
        for ind in population:
            if self._fitness(ind, Xn, y, label_counts) <= 0:
                continue
            rule = self._to_rule(ind, Xn, y, bin_names, label_counts)
            if rule is None:
                continue
            key = (rule["label"], tuple(rule["attrs"]), tuple(rule["values"]))
            if key not in seen or rule["weight"] > seen[key]["weight"]:
                seen[key] = rule

        self.rules = sorted(seen.values(),
                            key=lambda r: (-r["weight"], -r["purity"], r["label"]))

        if self.verbose:
            print(f"Generated {len(self.rules)} unique valid rules.\n")

    def print_rules(self, top_n=20):
        if not self.rules:
            print("No rules to display.")
            return
        print(f"\n=== TOP {top_n} RULES (sorted by weight) ===")
        for i, r in enumerate(self.rules[:top_n], 1):
            print(f"Rule {i:2d}: IF {' AND '.join(r['readable'])} "
                  f"THEN class = {r['label']} "
                  f"| Purity={r['purity']:.3f} | Support={r['repet']} "
                  f"| Weight={r['weight']:.3f}")