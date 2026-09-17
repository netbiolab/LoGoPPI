"""Calibrate Global and Maxsim logits and combine them equally."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.special import expit


def _vector(value: Any, name: str) -> np.ndarray:
    """Convert input values to a finite one-dimensional array."""
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def nll(labels: np.ndarray, logits: np.ndarray) -> float:
    """Compute the mean binary negative log-likelihood."""
    labels = _vector(labels, "labels")
    logits = _vector(logits, "logits")
    if labels.shape != logits.shape:
        raise ValueError("labels and logits lengths differ")
    if np.any((labels < 0) | (labels > 1)):
        raise ValueError("labels must be between 0 and 1")
    return float((np.logaddexp(0.0, logits) - labels * logits).mean())


def validate_calibration(state: Mapping[str, Any]) -> None:
    """Validate a LoGoPPI calibration file."""
    if state.get("format") not in {"x-species", "bernett"}:
        raise ValueError("format must be 'x-species' or 'bernett'")

    required_sections = {"maxsim_train", "global", "maxsim", "blend"}
    missing = required_sections.difference(state)
    extra = set(state).difference(required_sections | {"format"})
    if missing:
        raise ValueError(f"missing calibration section: {min(missing)}")
    if extra:
        raise ValueError(f"unexpected calibration field: {min(extra)}")

    expected_keys = {
        "maxsim_train": {"mean", "std"},
        "global": {"temperature", "bias"},
        "maxsim": {"scale", "bias"},
        "blend": {"global_weight", "maxsim_weight"},
    }
    for section, keys in expected_keys.items():
        actual = set(state[section])
        if actual != keys:
            raise ValueError(
                f"{section} must contain exactly: {', '.join(sorted(keys))}"
            )
        for name in keys:
            if not math.isfinite(float(state[section][name])):
                raise ValueError(f"{section}.{name} must be finite")

    for section, name in (
        ("maxsim_train", "std"),
        ("global", "temperature"),
        ("maxsim", "scale"),
    ):
        if float(state[section][name]) <= 0:
            raise ValueError(f"{section}.{name} must be positive")

    weights = (
        float(state["blend"]["global_weight"]),
        float(state["blend"]["maxsim_weight"]),
    )
    if weights != (0.5, 0.5):
        raise ValueError("LoGoPPI requires fixed Global and Maxsim weights of 0.5")


def calibrated_logits(
    global_logits: Any,
    maxsim_scores: Any,
    state: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return the calibrated branch logits and their equal-weight average."""
    validate_calibration(state)
    global_values = _vector(global_logits, "global_logits")
    maxsim_values = _vector(maxsim_scores, "maxsim_scores")
    if global_values.shape != maxsim_values.shape:
        raise ValueError("global_logits and maxsim_scores lengths differ")

    global_logit = (
        global_values / float(state["global"]["temperature"])
        + float(state["global"]["bias"])
    )
    standardized_maxsim = (
        maxsim_values - float(state["maxsim_train"]["mean"])
    ) / float(state["maxsim_train"]["std"])
    maxsim_logit = (
        float(state["maxsim"]["scale"]) * standardized_maxsim
        + float(state["maxsim"]["bias"])
    )
    final_logit = (
        float(state["blend"]["global_weight"]) * global_logit
        + float(state["blend"]["maxsim_weight"]) * maxsim_logit
    )
    return {
        "global_logit": global_logit,
        "maxsim_logit": maxsim_logit,
        "final_logit": final_logit,
    }


def apply_calibration(
    global_logits: Any,
    maxsim_scores: Any,
    state: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return calibrated logits and their sigmoid probabilities."""
    logits = calibrated_logits(global_logits, maxsim_scores, state)
    return {
        **logits,
        "p_global": expit(logits["global_logit"]),
        "p_maxsim": expit(logits["maxsim_logit"]),
        "score": expit(logits["final_logit"]),
    }
