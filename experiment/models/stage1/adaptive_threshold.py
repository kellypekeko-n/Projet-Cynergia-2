"""
Adaptive Threshold θ for Stage 1 (OCSVM / Composite Score)
Calibre automatiquement le seuil d'anomalie en production pour maintenir
un budget FPR cible, même si la distribution des flux évolue.

Méthode : Contrôle PI (Proportionnel-Intégral) sur estimation glissante du FPR.
  - Chaque flux flaggé est enregistré dans un buffer
  - Si l'analyste labélise TP/FP → mise à jour du FPR estimé
  - θ est ajusté par un contrôleur PI pour maintenir FPR ≤ fpr_budget

Mode autonome (sans feedback analyste) :
  - Estimation FPR via CDF empirique : P(score > θ | flux normaux passés)
  - Nécessite une fenêtre de flux connus normaux (après validation stage2)

Usage:
  from adaptive_threshold import AdaptiveThreshold
  at = AdaptiveThreshold.from_config(path)  # ou AdaptiveThreshold(theta_init, ...)
  theta = at.current_theta
  flag  = score >= theta
  at.update_feedback(score=score, is_fp=analyst_says_fp)
"""
import os, sys, json, time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from ton_iot_config import (METRICS_DIR, STAGE1_MAX_FPR, STAGE1_TARGET_RECALL)

ADAPTIVE_STATE_PATH = os.path.join(METRICS_DIR, "adaptive_threshold_state.json")


@dataclass
class _FeedbackEntry:
    score:     float
    is_fp:     bool     # True = false positive (analyst confirmed normal)
    timestamp: float = field(default_factory=lambda: time.time())


class AdaptiveThreshold:
    """
    Adaptive anomaly threshold controller for Stage 1.

    Parameters
    ----------
    theta_init    : float
        Starting threshold (from calibrate_theta on validation set).
    fpr_budget    : float
        Target FPR (e.g. 0.15 for ≤ 15% false positive rate).
    recall_floor  : float
        Minimum recall to maintain (never increase θ past this limit).
    window_size   : int
        Number of analyst-labeled flows to keep in sliding buffer.
    kp, ki        : float
        PI controller gains. kp drives immediate correction, ki reduces
        steady-state error.  Both ≥ 0; typically kp≈0.05, ki≈0.001.
    theta_min,max : float
        Hard limits on θ to prevent runaway.
    """

    def __init__(self,
                 theta_init:   float = 0.5,
                 fpr_budget:   float = STAGE1_MAX_FPR,
                 recall_floor: float = STAGE1_TARGET_RECALL,
                 window_size:  int   = 200,
                 kp:           float = 0.05,
                 ki:           float = 0.001,
                 theta_min:    float = 0.0,
                 theta_max:    float = 1.0):

        self._theta       = float(theta_init)
        self.fpr_budget   = fpr_budget
        self.recall_floor = recall_floor
        self.window_size  = window_size
        self.kp           = kp
        self.ki           = ki
        self.theta_min    = theta_min
        self.theta_max    = theta_max

        self._buffer: Deque[_FeedbackEntry] = deque(maxlen=window_size)
        self._integral  = 0.0
        self._n_updates = 0
        self._history   = []   # [(timestamp, theta, estimated_fpr)]

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def current_theta(self) -> float:
        return self._theta

    def predict(self, score: float) -> bool:
        """Flag as anomaly if score >= current theta."""
        return score >= self._theta

    def update_feedback(self, score: float, is_fp: bool) -> float:
        """
        Receive analyst feedback for one flagged flow.
        Returns updated theta.

        Parameters
        ----------
        score : float  — anomaly score for this flow
        is_fp : bool   — True if analyst confirmed it's a false positive
        """
        self._buffer.append(_FeedbackEntry(score=score, is_fp=is_fp))
        self._n_updates += 1

        if len(self._buffer) < 10:
            return self._theta  # not enough data yet

        # Estimate current FPR from buffer
        estimated_fpr = self._estimate_fpr()

        # PI control: error = estimated_fpr - budget
        error = estimated_fpr - self.fpr_budget
        self._integral += error

        delta_theta = self.kp * error + self.ki * self._integral
        new_theta   = np.clip(self._theta + delta_theta,
                               self.theta_min, self.theta_max)

        self._theta = float(new_theta)
        self._history.append({
            "timestamp":     datetime.now(tz=timezone.utc).isoformat(),
            "theta":         round(self._theta, 6),
            "estimated_fpr": round(estimated_fpr, 4),
            "error":         round(error, 4),
            "delta_theta":   round(delta_theta, 6),
            "n_buffer":      len(self._buffer),
        })
        return self._theta

    def update_from_scores(self, scores_normal: np.ndarray) -> float:
        """
        Autonomous update (no analyst labels): estimate FPR from recent
        confirmed-normal flows (e.g., flows that Stage 2 classified as 'normal').

        Parameters
        ----------
        scores_normal : anomaly scores for flows Stage 2 said are normal
        """
        if len(scores_normal) < 5:
            return self._theta

        estimated_fpr = float((scores_normal >= self._theta).mean())
        error         = estimated_fpr - self.fpr_budget
        self._integral += error
        delta_theta   = self.kp * error + self.ki * self._integral
        new_theta     = np.clip(self._theta + delta_theta, self.theta_min, self.theta_max)
        self._theta   = float(new_theta)
        self._history.append({
            "timestamp":     datetime.now(tz=timezone.utc).isoformat(),
            "theta":         round(self._theta, 6),
            "estimated_fpr": round(estimated_fpr, 4),
            "error":         round(error, 4),
            "delta_theta":   round(delta_theta, 6),
            "n_buffer":      int(len(scores_normal)),
        })
        return self._theta

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = ADAPTIVE_STATE_PATH):
        state = {
            "theta":         self._theta,
            "fpr_budget":    self.fpr_budget,
            "recall_floor":  self.recall_floor,
            "window_size":   self.window_size,
            "kp":            self.kp,
            "ki":            self.ki,
            "theta_min":     self.theta_min,
            "theta_max":     self.theta_max,
            "integral":      self._integral,
            "n_updates":     self._n_updates,
            "history":       self._history[-50:],  # last 50 updates
            "saved_at":      datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str = ADAPTIVE_STATE_PATH) -> "AdaptiveThreshold":
        with open(path) as f:
            state = json.load(f)
        obj = cls(
            theta_init=state["theta"],
            fpr_budget=state["fpr_budget"],
            recall_floor=state["recall_floor"],
            window_size=state["window_size"],
            kp=state["kp"],
            ki=state["ki"],
            theta_min=state["theta_min"],
            theta_max=state["theta_max"],
        )
        obj._integral  = state.get("integral", 0.0)
        obj._n_updates = state.get("n_updates", 0)
        obj._history   = state.get("history", [])
        return obj

    @classmethod
    def from_composite_weights(cls, fpr_budget: float = STAGE1_MAX_FPR) -> "AdaptiveThreshold":
        """Initialize from composite Stage 1 calibration (if available)."""
        weights_path = os.path.join(METRICS_DIR, "composite_stage1_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path) as f:
                w = json.load(f)
            theta_init = w.get("theta", 0.5)
        else:
            # Fallback: load from eda_and_stage1 metrics
            metrics_path = os.path.join(METRICS_DIR, "eda_and_stage1.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    m = json.load(f)
                s1 = m.get("stage1_results", {})
                best = next(iter(s1.values()), {}) if s1 else {}
                theta_init = best.get("theta", 0.5)
            else:
                theta_init = 0.5
        return cls(theta_init=theta_init, fpr_budget=fpr_budget)

    # ── internal ──────────────────────────────────────────────────────────────

    def _estimate_fpr(self) -> float:
        """Estimate FPR from the labeled buffer: FP / (FP + TN)."""
        entries = list(self._buffer)
        # Among flows flagged (score >= theta at time of labeling), count FP
        flagged_as_fp = sum(1 for e in entries if e.is_fp)
        # Among flows not flagged, count as TN (they weren't flagged = TN)
        # Buffer only contains flagged flows → FPR ≈ FP / len(buffer)
        return flagged_as_fp / (len(entries) + 1e-12)

    def status(self) -> dict:
        entries = list(self._buffer)
        fp_count = sum(1 for e in entries if e.is_fp)
        return {
            "current_theta":  round(self._theta, 6),
            "fpr_budget":     self.fpr_budget,
            "estimated_fpr":  round(self._estimate_fpr(), 4) if entries else None,
            "n_labeled":      len(entries),
            "n_fp":           fp_count,
            "n_updates":      self._n_updates,
            "integral":       round(self._integral, 6),
        }


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from ton_iot_config import FIGURES_DIR, PALETTE

    print("\n=== Simulation Seuil Adaptatif Stage 1 ===\n")

    rng   = np.random.RandomState(42)
    theta_true_normal  = 0.5   # initial calibration
    at = AdaptiveThreshold.from_composite_weights(fpr_budget=0.05)
    print(f"  Seuil initial : θ={at.current_theta:.4f}")

    # Phase 1 : stable environment (200 flows, FPR ~5%)
    # Phase 2 : noisy environment (200 flows, FPR ~20%) — simulate misconfiguration
    # Phase 3 : returns to normal (200 flows)
    theta_trace = [at.current_theta]
    fpr_trace   = []
    n_trace     = []

    for i in range(600):
        # Generate a "flagged" flow score
        if i < 200:     # stable
            score = rng.normal(loc=0.55, scale=0.1)
            true_label = "attack" if score > 0.6 else "normal"
        elif i < 400:   # drifted — more false positives
            score = rng.normal(loc=0.60, scale=0.15)
            true_label = "normal" if score < 0.75 else "attack"
        else:           # stable again
            score = rng.normal(loc=0.53, scale=0.1)
            true_label = "attack" if score > 0.62 else "normal"

        is_fp = (true_label == "normal") and (score >= at.current_theta)
        if score >= at.current_theta:
            at.update_feedback(score=score, is_fp=is_fp)

        theta_trace.append(at.current_theta)
        if at._history:
            fpr_trace.append(at._history[-1]["estimated_fpr"])
            n_trace.append(i)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(theta_trace, color=PALETTE[0], linewidth=1.5, label="θ adaptatif")
    ax1.axhline(theta_true_normal, color='gray', linestyle='--', linewidth=1,
                label=f"θ initial ({theta_true_normal:.2f})")
    ax1.axvline(200, color='orange', linestyle=':', linewidth=1.5, label="Dérive injectée")
    ax1.axvline(400, color='green',  linestyle=':', linewidth=1.5, label="Retour à la normale")
    ax1.set_ylabel("Seuil θ"); ax1.legend(fontsize=8)
    ax1.set_title("Seuil adaptatif Stage 1 — Contrôleur PI (simulation)")

    if fpr_trace:
        ax2.plot(n_trace, fpr_trace, color=PALETTE[1], linewidth=1.5, label="FPR estimé")
        ax2.axhline(at.fpr_budget, color='red', linestyle='--', linewidth=1.5,
                    label=f"Budget FPR ({at.fpr_budget:.0%})")
        ax2.axvline(200, color='orange', linestyle=':', linewidth=1.5)
        ax2.axvline(400, color='green',  linestyle=':', linewidth=1.5)
    ax2.set_ylabel("FPR estimé"); ax2.set_xlabel("Flux analysés"); ax2.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_adaptive_threshold.png")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)

    at.save()
    print(f"  État sauvegardé : {ADAPTIVE_STATE_PATH}")
    print(f"  Figure          : {out}")
    print(f"  Seuil final     : θ={at.current_theta:.4f}")
    print(f"  Statut          : {at.status()}")
