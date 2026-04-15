"""
Shared environment and dataset utilities for the cyber RL project.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


class CyberKillChainEnv:
    """
    Simplified cyber kill-chain environment with a linear 5-state progression.

    States:
        0: Start
        1: Scanned
        2: Vulns_Found
        3: Exploited
        4: Exfiltrated (terminal)

    Actions:
        0: scan_network
        1: search_vulns
        2: run_exploit
        3: steal_data
    """

    def __init__(
        self,
        max_steps: int = 20,
        transition_probabilities: Optional[Dict[int, float]] = None,
    ) -> None:
        # Problem constants: tiny discrete MDP for fast CPU experiments.
        self.n_states = 5
        self.n_actions = 4
        self.terminal_state = 4
        self.max_steps = max_steps
        # Per-state success chance used when the correct action is chosen.
        self.transition_probabilities = transition_probabilities or {
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 1.0,
        }
        self.state = 0
        self.steps_taken = 0

    def reset(self) -> int:
        """Reset episode and return initial state."""
        # Always restart from initial kill-chain stage.
        self.state = 0
        self.steps_taken = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Execute one environment step."""
        if self.state == self.terminal_state:
            return self.state, 0.0, True, {"message": "Episode already terminal."}

        self.steps_taken += 1
        done = False
        # In this simplified design, each state has exactly one "correct" action.
        correct_action = self.state

        wrong_action = False
        successful_progress = False

        if action == correct_action:
            # Dataset-derived probability makes the environment less deterministic.
            success_prob = float(self.transition_probabilities.get(self.state, 1.0))
            if random.random() <= success_prob:
                successful_progress = True
                if self.state == 3:
                    # Final successful stage transition.
                    self.state = 4
                    reward = 100.0
                    done = True
                else:
                    self.state += 1
                    reward = 1.0
            else:
                reward = -1.0
        else:
            wrong_action = True
            reward = -1.0

        if self.steps_taken >= self.max_steps and self.state != self.terminal_state:
            # Episode timeout avoids infinite loops under repeated bad actions.
            done = True

        info = {
            "wrong_action": wrong_action,
            "successful_progress": successful_progress,
            "is_terminal_success": self.state == self.terminal_state and done,
        }
        return self.state, reward, done, info


def _safe_mean(values: List[float]) -> float:
    """Return clipped mean in [0.05, 0.95]."""
    if not values:
        return 0.5
    value = float(np.mean(values))
    return float(np.clip(value, 0.05, 0.95))


def infer_transition_probabilities_from_rows(
    rows: List[Dict[str, str]]
) -> Dict[int, float]:
    """Infer phase transition probabilities from CSV rows."""
    transition_probs = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    if not rows:
        return transition_probs

    column_names = list(rows[0].keys())
    lower_cols = {col.lower(): col for col in column_names}

    # Map kill-chain stages to broad column-name cues in the dataset.
    phase_keywords = {
        0: ("scan", "recon", "discover"),
        1: ("vuln", "cve", "weak", "enumer"),
        2: ("exploit", "rce", "payload", "shell"),
        3: ("exfil", "steal", "data", "leak"),
    }

    for phase_idx, keywords in phase_keywords.items():
        # Gather columns that appear to represent this kill-chain phase.
        matched_cols = [
            original_col
            for lower, original_col in lower_cols.items()
            if any(keyword in lower for keyword in keywords)
        ]
        if not matched_cols:
            continue

        phase_values: List[float] = []
        for col in matched_cols:
            numeric_values: List[float] = []
            for row in rows:
                raw_value = row.get(col, "")
                try:
                    numeric_values.append(float(raw_value))
                except (TypeError, ValueError):
                    continue

            if numeric_values:
                # Normalize large scales before averaging into probability space.
                max_abs = float(np.nanmax(np.abs(np.array(numeric_values, dtype=float))))
                if max_abs > 1.0:
                    numeric_values = [value / max_abs for value in numeric_values]
                phase_values.append(_safe_mean(numeric_values))

        if phase_values:
            # Average over all matched columns to get one phase probability.
            transition_probs[phase_idx] = float(np.mean(phase_values))

    return transition_probs


def load_netsec_dataset(
    dataset_path: str = "data/netsecdata.csv",
    dataset_url: str = (
        "https://huggingface.co/datasets/stratosphere/NetSecData/resolve/main/netsecdata.csv"
    ),
) -> Tuple[Optional[List[Dict[str, str]]], Dict[int, float]]:
    """Load CSV data and infer transition probabilities."""
    local_path = Path(dataset_path)
    try:
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[Info] Local dataset not found. Downloading from: {dataset_url}")
            urlretrieve(dataset_url, local_path.as_posix())

        rows: List[Dict[str, str]] = []
        # Use stdlib CSV parsing to avoid pandas dependency issues.
        with local_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows.append(row)

        probs = infer_transition_probabilities_from_rows(rows)
        return rows, probs
    except Exception as exc:
        print(
            f"[Warning] Could not load dataset from '{local_path.as_posix()}' or URL: {exc}"
        )
        print("[Warning] Falling back to deterministic transition probabilities.")
        return None, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
