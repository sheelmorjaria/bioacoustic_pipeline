"""Prospective diagnostic fixtures for missing-data semantics; no production changes."""

import torch

PATTERNS = [
    "finite",
    "observed_nan",
    "observed_posinf",
    "observed_neginf",
    "masked_nan",
    "masked_posinf",
    "masked_neginf",
    "empty_first_origin",
    "empty_all",
    "sparse_one_per_origin",
    "mixed_observed_nan_masked_inf",
]
EXPECTED_VALID = {
    "finite": True,
    "masked_nan": True,
    "masked_posinf": True,
    "masked_neginf": True,
    "sparse_one_per_origin": True,
    "observed_nan": False,
    "observed_posinf": False,
    "observed_neginf": False,
    "empty_first_origin": False,
    "empty_all": False,
    "mixed_observed_nan_masked_inf": False,
}
SHAPES = [(4, 3, 64), (128, 3, 64)]
LAYOUTS = ["contiguous", "strided", "offset7"]
OPERATIONS = ["and", "where"]


def guard_fixture(shape, pattern):
    if shape not in SHAPES or pattern not in PATTERNS:
        raise ValueError("Unregistered guard fixture")
    x = torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.float32).reshape(shape) / 1024
    mask = torch.ones_like(x, dtype=torch.bool)
    if pattern.startswith(("observed_", "masked_")):
        suffix = pattern.split("_", 1)[1]
        value = {"nan": float("nan"), "posinf": float("inf"), "neginf": float("-inf")}[suffix]
        x.flatten()[0] = value
        if pattern.startswith("masked_"):
            mask.flatten()[0] = False
    elif pattern == "empty_first_origin":
        mask[0] = False
        x[0] = float("nan")
    elif pattern == "empty_all":
        mask.fill_(False)
        x.fill_(float("nan"))
    elif pattern == "sparse_one_per_origin":
        mask.fill_(False)
        x.fill_(float("nan"))
        mask.reshape(shape[0], -1)[:, 0] = True
        x.reshape(shape[0], -1)[:, 0] = torch.arange(shape[0], dtype=torch.float32) / 100
    elif pattern == "mixed_observed_nan_masked_inf":
        x.flatten()[0] = float("nan")
        x.flatten()[1] = float("inf")
        mask.flatten()[1] = False
    return x, mask


def schedule():
    return [
        dict(
            name=f"B{shape[0]}.{pattern}.{layout}.{operation}",
            shape=shape,
            pattern=pattern,
            layout=layout,
            operation=operation,
            expected_valid=EXPECTED_VALID[pattern],
        )
        for shape in SHAPES
        for pattern in PATTERNS
        for layout in LAYOUTS
        for operation in OPERATIONS
    ]
