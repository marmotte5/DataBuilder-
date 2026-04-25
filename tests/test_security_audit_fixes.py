"""Audit-fix regression tests.

Locks in the P0/P1 audit fixes so they can't silently regress:
- CRIT-1: training_state.py uses weights_only=True
- MAJ-1: TRUST_REMOTE_CODE_MODELS lives in constants.py and matches the
  legacy generate_worker definition (no drift between training/generate).
- MIN-7: .gitignore covers training session files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_training_state_uses_weights_only_true():
    """The RNG-state load path must use weights_only=True (RCE mitigation)."""
    src = (REPO_ROOT / "dataset_sorter" / "training_state.py").read_text()
    # Any torch.load on the RNG file must explicitly pass weights_only=True
    assert "weights_only=True" in src, (
        "training_state.py must use weights_only=True on RNG-state load"
    )
    assert "weights_only=False" not in src, (
        "training_state.py must NOT use weights_only=False anywhere"
    )


def test_trust_remote_code_models_lives_in_constants():
    """Single source of truth: TRUST_REMOTE_CODE_MODELS in constants.py."""
    from dataset_sorter.constants import TRUST_REMOTE_CODE_MODELS
    assert isinstance(TRUST_REMOTE_CODE_MODELS, set)
    # The architectures that need trust_remote_code in diffusers as of 2026
    expected = {"zimage", "flux2", "chroma", "hidream"}
    assert expected.issubset(TRUST_REMOTE_CODE_MODELS), (
        f"Missing trust-required archs: {expected - TRUST_REMOTE_CODE_MODELS}"
    )


def test_generate_worker_imports_trust_set_from_constants():
    """generate_worker must consume the central TRUST_REMOTE_CODE_MODELS set."""
    src = (REPO_ROOT / "dataset_sorter" / "generate_worker.py").read_text()
    assert "from dataset_sorter.constants import" in src
    assert "TRUST_REMOTE_CODE_MODELS" in src
    # And it must NOT redefine it locally
    assert "TRUST_REMOTE_CODE_MODELS = {" not in src, (
        "generate_worker.py shouldn't redefine TRUST_REMOTE_CODE_MODELS — "
        "it must import the set from constants.py"
    )


def test_gitignore_covers_session_files():
    """Training session files must be git-ignored to avoid leaking paths."""
    gi = (REPO_ROOT / ".gitignore").read_text()
    for pattern in (".last_session.json", "random_states.pt",
                     "random_states_aux.json", "checkpoint-*/"):
        assert pattern in gi, f"Missing .gitignore pattern: {pattern!r}"


@pytest.mark.skipif(
    importlib.util.find_spec("PyQt6") is None,
    reason="PyQt6 not installed",
)
def test_security_prompts_module_imports():
    """The security_prompts helper must import cleanly."""
    from dataset_sorter.ui.security_prompts import (
        confirm_trust_remote_code, reset_trust_decisions,
    )
    assert callable(confirm_trust_remote_code)
    assert callable(reset_trust_decisions)
