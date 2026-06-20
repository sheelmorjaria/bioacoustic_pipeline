"""Tests for the speaker-controlled functional-alias null.

Pins: genuine aliases (tokens with identical following-context) are detected and
SURVIVE the within-file / frequency / within-emitter nulls; and the within-group
shuffle preserves each group's token multiset + bout lengths.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import collections
import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.alias_detector import (
    alias_density,
    controlled_alias_tests,
    shuffle_sequences_within_group,
)


def _genuine_alias_corpus(seed: int = 7, n: int = 300):
    """Tokens 10 & 11 are always followed by token 5 -> identical following-context."""
    rng = np.random.RandomState(seed)
    seqs, emit = [], []
    for i in range(n):
        emit.append(i % 4)
        length = rng.randint(10, 16)
        s = []
        for _ in range(length):
            x = int(rng.choice([10, 11, 2, 3, 4]))
            s.append(x)
            s.append(5 if x in (10, 11) else int(rng.choice([2, 3, 4, 5])))
        seqs.append(s)
    return seqs, emit


class TestControlledAlias(unittest.TestCase):
    def test_genuine_aliases_detected_and_survive_nulls(self):
        seqs, emit = _genuine_alias_corpus()
        self.assertGreater(alias_density(seqs), 0.5)
        res = controlled_alias_tests(seqs, emitter=emit, n_permutations=100, seed=0)
        for name in ("within_file", "frequency", "within_emitter"):
            self.assertIn(name, res)
            # Observed structure must exceed each structure-destroying null.
            self.assertGreater(res[name].observed, res[name].null_mean, name)
            self.assertLess(res[name].p_value, 0.1, name)

    def test_controls_only_run_when_group_supplied(self):
        seqs, _emit = _genuine_alias_corpus()
        base = controlled_alias_tests(seqs, n_permutations=50, seed=0)
        self.assertEqual(set(base.keys()), {"within_file", "frequency"})
        with_emit = controlled_alias_tests(seqs, emitter=[0] * len(seqs), n_permutations=50, seed=0)
        self.assertIn("within_emitter", with_emit)

    def test_shuffle_preserves_group_marginals_and_lengths(self):
        rng = np.random.RandomState(0)
        seqs = [[int(x) for x in rng.randint(0, 5, rng.randint(3, 8))] for _ in range(40)]
        group = [i % 3 for i in range(40)]
        shuffled = shuffle_sequences_within_group(seqs, group, rng)
        # bout lengths identical
        self.assertEqual([len(s) for s in seqs], [len(s) for s in shuffled])
        # per-group token totals preserved
        before = collections.defaultdict(int)
        for s, g in zip(seqs, group):
            before[g] += len(s)
        after = collections.defaultdict(int)
        for s, g in zip(shuffled, group):
            after[g] += len(s)
        self.assertEqual(dict(before), dict(after))


if __name__ == "__main__":
    unittest.main()
