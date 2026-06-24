# Structural Alternatives — 4 paradigms corroborating DUSTED

> Personal methodology-usage reference. See
> `diagnostics/structural_alternatives/{rqa,matrix_profile_motifs,probabilistic_suffix_tree,transition_network}.py`,
> BENCHMARK §33–§34.

## What it is

Four **firewall-compliant, independent-mathematics** techniques that corroborate DUSTED from
outside DTW. They are NOT a replacement for DUSTED — they exist to prove DUSTED's regime
classifications are a property of the vocalisations, not an idiosyncrasy of the DTW metric.
Each is pure-stdlib or lazy-imported, deterministic, and operates on acoustic-similarity token
ids (not learned semantics).

| Paradigm | Module | Mathematics | Best lens for |
|---|---|---|---|
| **RQA** | `rqa.py` (pure numpy) | recurrence plot → %DET / %LAM / L / TT | graded dynamics |
| **Matrix Profile** | `matrix_profile_motifs.py` (stumpy, lazy) | STUMP continuous subsequence matching | degeneracy pre-filter |
| **PST** | `probabilistic_suffix_tree.py` (pure stdlib) | variable-order Markov syntax | discrete-syntax richness |
| **Transition Network** | `transition_network.py` (networkx, lazy) | directed weighted graph topology | global modularity |

## API

```python
from bioacoustic_pipeline.diagnostics.structural_alternatives import (
    rqa, matrix_profile_motifs, probabilistic_suffix_tree as pst, transition_network as tn,
)

# sequences = list[list[int]] of k-means token ids (same tokenization DUSTED uses)
pst_sum = pst.fit_pst(sequences, max_depth=4, pmin=0.1, alpha=0.5, min_count=5)
net     = tn.transition_network_metrics(sequences)          # nodes/edges/density/clustering/communities/MPL
g       = tn.build_transition_graph(sequences)              # DiGraph — top weighted edges = bigram candidates

rp  = rqa.recurrence_plot(X, target_rec=0.05)               # X = (n_frames, D) trajectory
rqa_metrics = rqa.rqa_metrics(rp, ...)                      # REC, DET, LAM, mean_diag_length, trapping_time
mp  = matrix_profile_motifs.matrix_profile(X, m=8, n_components=1)  # recurrence_fraction + motif_distance
```

Key implementation notes:
- **RQA** holds recurrence rate (~REC) constant across calls of different length by choosing a
  per-call distance threshold → DET and LAM (the *structural* metrics) become length-invariant.
  `_runs()` does the RLE for diagonal/vertical line counting.
- **Matrix Profile**: `stumpy.stump` returns an `mparray` (2-D **object** array) — coerce
  `[:,0]` to `float64`; PCA to 1 component (PC1) before STUMP. Reports `recurrence_fraction`
  + `motif_distance` distribution (bimodal = a second motif family).
- **PST** keeps a context only if `P(a|s) ≥ pmin` AND `P(a|s) ≥ (1+α)·P(a|suffix(s))` AND
  `count ≥ min_count` → `n_significant_nodes` is the syntax-richness number.
- **Transition Network**: greedy-modularity communities on the weighted undirected projection;
  mean harmonic path length on the largest weakly-connected component.

## Commands

```bash
# Per-dataset DUSTED-vs-alternatives comparison
python scripts/compare_structure_methods.py --n-files 20
python scripts/bat_rqa_by_context.py --per-context 30 --top-k 6      # RQA DET/LAM by Context
python scripts/marmoset_matrix_profile_sweep.py --n-calls 200        # token-free degeneracy check
python scripts/bengalese_network_topology.py --n-files 100           # full transition-network topology
```

## ⚠️ Gotchas

1. **They corroborate, they do not replace.** Promotion rule: a *decisive* regime-win becomes a
   router branch (e.g. `graded → RQA`), never a wholesale DUSTED replacement. Keep DUSTED as
   the headline method.
2. **PST / transition-network need DISCRETE tokens** — weak on graded repertoires. RQA is the
   graded-bout lens (it runs on the continuous trajectory, not tokens).
3. **Matrix Profile is the token-free degeneracy check.** Use it to prove a "degenerate
   inventory" claim (e.g. infant marmoset) is not a k-means artifact — MP on the raw
   trajectory agrees (87.5% stutter-dominant, unimodal).
4. **`stumpy.stump` dtype trap** — see API; the `[:,0]` object-array must be coerced to float64
   or `np.isfinite` rejects it.
5. **All firewall-clean / meaning-free.** Tokens are acoustic-similarity cluster ids; metrics
   describe structure, not function.

## Empirical headline (§33–§34 — DUSTED corroborated 4 ways)

| Paradigm | Dataset | Result | Verdict vs DUSTED |
|---|---|---|---|
| PST + Graph | Bengalese | 247 contexts, 5 communities, clustering 0.686 | **Corroborates** discrete syntax |
| RQA | Bat | %LAM context-dependent (**p=0.016**); %DET invariant | **Complements** (dynamics DTW missed) |
| Matrix Profile | Marmoset | 87.5% stutter-dominant, unimodal | **Confirms** degeneracy token-free |
| Transition Network | LibriSpeech | 639 nodes, 19 communities, path 1.26 | **Upper bound** (scales to human diversity) |

## Related
- `dusted.md` — the method these corroborate.
- §37 compositional mining reuses PST + transition_network for bigram-candidate detection.
