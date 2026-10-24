"""Verify speaker weighting and paired schedule without fitting any model."""

import importlib.util
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path("scripts").resolve()))
spec = importlib.util.spec_from_file_location("p10_training", Path("scripts/rclat_p10_train.py"))
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def test_macro_counts_speakers_equally_not_utterances():
    correct = {
        "speaker": "a",
        "reference_characters": 10,
        "character_errors": 0,
        "reference_words": 2,
        "word_errors": 0,
        "hypothesis": "A WORD",
    }
    empty = {
        "speaker": "b",
        "reference_characters": 10,
        "character_errors": 10,
        "reference_words": 2,
        "word_errors": 2,
        "hypothesis": "",
    }
    result = training.aggregate([correct] * 9 + [empty])
    assert result["speaker_macro_cer"] == 0.5
    assert result["pooled_cer"] == 0.1
    assert result["nonempty_fraction"] == 0.9


def test_paired_order_and_schedule_fixed_without_outcomes():
    rows = [{"uid": str(i)} for i in range(20)]
    assert training.epoch_order(rows, 1) == training.epoch_order(list(reversed(rows)), 1)
    assert training.epoch_order(rows, 1) != training.epoch_order(rows, 2)
    assert training.learning_rate(500, 5000) == pytest.approx(1e-3)
    assert training.learning_rate(5000, 5000) == pytest.approx(1e-5)
