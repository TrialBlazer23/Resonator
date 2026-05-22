"""
Resonator Framework — ChaosCore
The unsupervised latent monitor and intervention engine.

Architecture:
  - Hooks into any HuggingFace CausalLM via forward hooks (no model modification)
  - Tracks token surprise, hidden state trajectory, and all derived metrics
  - Computes attractor state at each generation step
  - Returns logit adjustments to the generation loop via LogitsProcessor API

Design principle (Resonator): detect phase shifts and gradients, not static states.
The Chaos Core fires on the APPROACH to failure, not on failure itself.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .config import ChaosConfig
from .metrics import (
    compute_ar1,
    compute_variance_ratio,
    compute_latent_velocity,
    compute_orbit_resonance,
    compute_svd_entropy,
    compute_integrated_stress,
    attractor_state,
)


@dataclass
class StepTelemetry:
    """Snapshot of all Chaos Core metrics at a single generation step."""
    step: int
    token_id: int
    token_str: str
    surprise: float
    ar1: float
    variance_ratio: float
    latent_velocity: float
    orbit_period: Optional[int]
    orbit_distance: Optional[float]
    svd_entropy: float
    integrated_stress: float
    state: str
    intervention_active: bool
    temperature_applied: float
    logit_delta: float  # magnitude of logit adjustment applied


class ChaosCore(LogitsProcessor):
    """
    Wraps a HuggingFace CausalLM to provide unsupervised cognitive safety monitoring.

    Usage:
        core = ChaosCore(model, tokenizer)
        with core.monitor():
            output = model.generate(
                inputs,
                logits_processor=LogitsProcessorList([core]),
                max_new_tokens=200
            )
        telemetry = core.get_telemetry()

    The monitor context manager registers/unregisters the hidden state hook.
    The LogitsProcessor interface intercepts logits at each step for intervention.
    """

    def __init__(self, model, tokenizer, config: Optional[ChaosConfig] = None, verbose: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ChaosConfig()
        self.verbose = verbose

        # Rolling state
        self._surprise_history: deque = deque(maxlen=self.config.window_size)
        self._hidden_states: deque = deque(maxlen=max(self.config.orbit_max_lag + 5, self.config.svd_window + 5))
        self._integrated_stress: float = 0.0
        self._schmitt_active: bool = False  # hysteresis state
        self._step: int = 0
        self._telemetry: List[StepTelemetry] = []

        # Hook state
        self._hook = None
        self._last_hidden: Optional[np.ndarray] = None

        # Anchor token IDs (resolve once at init)
        self._anchor_ids = self._resolve_anchor_ids()

        # Baseline variance (updated during healthy generation)
        self._baseline_variance: Optional[float] = None

    # ── Context Manager ──────────────────────────────────────────────────────

    def __enter__(self):
        self._register_hook()
        self.reset()
        return self

    def __exit__(self, *args):
        self._remove_hook()

    def monitor(self):
        """Alias for use as context manager: `with core.monitor():`"""
        return self

    def reset(self):
        """Clear all state for a new generation run."""
        self._surprise_history.clear()
        self._hidden_states.clear()
        self._integrated_stress = 0.0
        self._schmitt_active = False
        self._step = 0
        self._telemetry = []
        self._last_hidden = None
        self._baseline_variance = None

    # ── Hook Registration ────────────────────────────────────────────────────

    def _register_hook(self):
        """Register forward hook on the final transformer block."""
        target = self._find_hook_target()
        if target is None:
            raise RuntimeError(
                "ChaosCore: Cannot find a hookable layer. "
                "Supported architectures: GPT-2, GPT-Neo, LLaMA, Mistral, Falcon. "
                "Pass a custom hook target via ChaosCore(..., hook_target=layer)."
            )

        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # Take the last token's hidden state, flatten to 1D
            self._last_hidden = h[0, -1, :].detach().float().cpu().numpy()

        self._hook = target.register_forward_hook(hook_fn)

    def _find_hook_target(self):
        """Auto-detect the final transformer layer for common architectures."""
        model = self.model
        # GPT-2
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h[-1]
        # GPT-Neo, GPT-J
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h[-1]
        # LLaMA, Mistral, Falcon
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers[-1]
        # OPT
        if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
            return model.model.decoder.layers[-1]
        # Bloom
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h[-1]
        return None

    def _remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    # ── LogitsProcessor Interface ────────────────────────────────────────────

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Called by HuggingFace generate() at every step.
        Computes metrics, classifies state, applies interventions.
        """
        cfg = self.config

        # ── 1. Capture hidden state (populated by forward hook) ──────────────
        if self._last_hidden is not None:
            self._hidden_states.append(self._last_hidden.copy())

        # ── 2. Compute token surprise (entropy of last generated token) ───────
        last_token_id = int(input_ids[0, -1])
        with torch.no_grad():
            probs = F.softmax(scores[0], dim=-1)
            token_prob = probs[last_token_id].item()
        surprise = -np.log(np.clip(token_prob, 1e-10, 1.0))
        self._surprise_history.append(surprise)

        # ── 3. Compute all metrics ────────────────────────────────────────────
        surprise_arr = np.array(list(self._surprise_history))
        hidden_list = list(self._hidden_states)

        ar1 = compute_ar1(surprise_arr)
        var_ratio = compute_variance_ratio(surprise_arr)
        velocity = compute_latent_velocity(hidden_list)
        orbit_period, orbit_dist = compute_orbit_resonance(
            hidden_list, cfg.orbit_min_lag, cfg.orbit_max_lag, cfg.orbit_distance_threshold
        )
        svd_entropy = compute_svd_entropy(hidden_list, cfg.svd_window)

        # ── 4. Classify attractor state ───────────────────────────────────────
        state = "HEALTHY"
        if self._step >= cfg.min_history:
            state = attractor_state(ar1, velocity, orbit_period, svd_entropy, cfg)

        # ── 5. Compute threat signal and update integrated stress ─────────────
        threat = self._compute_threat(state, ar1, velocity, orbit_dist)
        self._integrated_stress = compute_integrated_stress(
            self._integrated_stress, threat, cfg.stress_rise_rate, cfg.stress_decay_rate
        )
        F_t = self._integrated_stress

        # ── 6. Schmitt trigger (hysteresis prevents rapid switching) ──────────
        if not self._schmitt_active and F_t >= cfg.schmitt_upper:
            self._schmitt_active = True
        elif self._schmitt_active and F_t <= cfg.schmitt_lower:
            self._schmitt_active = False

        intervention_active = self._schmitt_active or state in ("POINT_ATTRACTOR", "LIMIT_CYCLE")

        # ── 7. Build modified logits ──────────────────────────────────────────
        modified_scores = scores.clone()
        total_delta = 0.0

        if intervention_active:
            modified_scores, delta = self._apply_intervention(
                modified_scores, input_ids, state, orbit_period, F_t
            )
            total_delta = delta

        # ── 8. Record telemetry ───────────────────────────────────────────────
        token_str = self.tokenizer.decode([last_token_id])
        temp_applied = self._compute_temperature(F_t) if intervention_active else cfg.base_temperature

        self._telemetry.append(StepTelemetry(
            step=self._step,
            token_id=last_token_id,
            token_str=token_str,
            surprise=round(surprise, 4),
            ar1=round(ar1, 4),
            variance_ratio=round(var_ratio, 4),
            latent_velocity=round(velocity, 4),
            orbit_period=orbit_period,
            orbit_distance=round(orbit_dist, 4) if orbit_dist is not None else None,
            svd_entropy=round(svd_entropy, 4),
            integrated_stress=round(F_t, 4),
            state=state,
            intervention_active=intervention_active,
            temperature_applied=round(temp_applied, 4),
            logit_delta=round(total_delta, 4),
        ))

        if self.verbose:
            self._print_step(self._telemetry[-1])

        self._step += 1
        return modified_scores

    # ── Threat Computation ───────────────────────────────────────────────────

    def _compute_threat(self, state: str, ar1: float, velocity: float, orbit_dist: Optional[float]) -> float:
        """Map attractor state + metrics to a scalar threat level [0, 1]."""
        if state == "POINT_ATTRACTOR":
            return 1.0
        if state == "LIMIT_CYCLE":
            # Closer orbit = higher threat
            return 1.0 - (orbit_dist / self.config.orbit_distance_threshold) if orbit_dist else 0.8
        if state == "CSD_WARNING":
            # Scale with how far AR(1) exceeds threshold
            excess = ar1 - self.config.ar1_threshold
            return min(1.0, excess / (1.0 - self.config.ar1_threshold))
        if state == "ENTROPY_COLLAPSE":
            return 0.5
        return 0.0

    # ── Intervention Engine ──────────────────────────────────────────────────

    def _compute_temperature(self, stress: float) -> float:
        """Continuous temperature modulation: smoothly expands from base to max."""
        cfg = self.config
        t_range = cfg.max_temperature - cfg.base_temperature
        return cfg.base_temperature + t_range * stress

    def _apply_intervention(
        self,
        scores: torch.FloatTensor,
        input_ids: torch.LongTensor,
        state: str,
        orbit_period: Optional[int],
        stress: float,
    ):
        """
        Surgical intervention targeting specific failure mode.
        
        Point Attractor: penalize the dominant frozen token(s).
        Limit Cycle: penalize only the orbit-period tokens (surgical, not global).
        CSD Warning: temperature modulation + anchor boost.
        All: semantic anchoring proportional to stress.
        """
        cfg = self.config
        delta = 0.0

        # Temperature scaling (apply to all scores)
        temp = self._compute_temperature(stress)
        if temp != 1.0:
            scores = scores / temp

        if state == "POINT_ATTRACTOR":
            # Penalize the currently dominant (frozen) token
            top_token = int(scores[0].argmax())
            scores[0, top_token] -= cfg.point_attractor_penalty
            delta += cfg.point_attractor_penalty

        elif state == "LIMIT_CYCLE" and orbit_period is not None:
            # Penalize only the tokens from the active orbit
            orbit_tokens = self._extract_orbit_tokens(input_ids, orbit_period)
            for tok in orbit_tokens:
                scores[0, tok] -= cfg.orbit_penalty_strength
            delta += cfg.orbit_penalty_strength * len(orbit_tokens)

        # Semantic anchoring: boost discourse markers proportional to stress
        anchor_boost = cfg.anchor_boost_strength * stress
        if anchor_boost > 0.1 and self._anchor_ids:
            for aid in self._anchor_ids:
                if aid < scores.shape[-1]:
                    scores[0, aid] += anchor_boost
            delta -= anchor_boost  # anchoring is positive, note differently

        return scores, delta

    def _extract_orbit_tokens(self, input_ids: torch.LongTensor, period: int) -> List[int]:
        """Extract the unique token IDs from the last `period` positions."""
        ids = input_ids[0, -period:].tolist()
        return list(set(ids))

    def _resolve_anchor_ids(self) -> List[int]:
        """Convert anchor word strings to token IDs."""
        ids = []
        for word in self.config.anchor_tokens:
            for variant in [word, f" {word}", f" {word.capitalize()}"]:
                enc = self.tokenizer.encode(variant, add_special_tokens=False)
                ids.extend(enc)
        return list(set(ids))

    # ── Output ───────────────────────────────────────────────────────────────

    def get_telemetry(self) -> List[StepTelemetry]:
        return list(self._telemetry)

    def get_telemetry_dict(self) -> Dict[str, List]:
        """Columnar format for easy DataFrame construction."""
        if not self._telemetry:
            return {}
        keys = list(StepTelemetry.__dataclass_fields__.keys())
        return {k: [getattr(t, k) for t in self._telemetry] for k in keys}

    def summary(self) -> Dict[str, Any]:
        """High-level summary of the generation run."""
        tel = self._telemetry
        if not tel:
            return {}
        states = [t.state for t in tel]
        return {
            "total_steps": len(tel),
            "intervention_steps": sum(1 for t in tel if t.intervention_active),
            "state_counts": {s: states.count(s) for s in set(states)},
            "peak_stress": max(t.integrated_stress for t in tel),
            "peak_ar1": max(t.ar1 for t in tel),
            "min_velocity": min(t.latent_velocity for t in tel),
            "orbit_detections": sum(1 for t in tel if t.orbit_period is not None),
            "min_svd_entropy": min(t.svd_entropy for t in tel),
        }

    # ── Debug ────────────────────────────────────────────────────────────────

    def _print_step(self, t: StepTelemetry):
        state_color = {
            "HEALTHY": "\033[92m",
            "CSD_WARNING": "\033[93m",
            "POINT_ATTRACTOR": "\033[91m",
            "LIMIT_CYCLE": "\033[91m",
            "ENTROPY_COLLAPSE": "\033[95m",
        }.get(t.state, "")
        reset = "\033[0m"
        intervention = " ⚡INTERVENE" if t.intervention_active else ""
        print(
            f"[{t.step:3d}] {repr(t.token_str):12s} | "
            f"AR1={t.ar1:.3f} V={t.latent_velocity:.3f} "
            f"SVD={t.svd_entropy:.3f} F={t.integrated_stress:.3f} | "
            f"{state_color}{t.state}{reset}{intervention}"
        )
