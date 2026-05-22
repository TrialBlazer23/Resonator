# Resonator
### Unsupervised Thermodynamic Cognitive Safety for Autoregressive LLMs

> *Detect phase shifts and gradients, not static states.*

---

## The Problem

Standard autoregressive LLMs have no internal awareness of when their generation is degrading. They produce tokens until they fail — then fail visibly. There is no early warning.

Two specific failure modes are poorly addressed by existing mitigations:

**Point Attractors** — the model's hidden-state trajectory collapses to a fixed point. Output becomes frozen, repetitive. `repetition_penalty` helps but is a blunt instrument applied after the fact.

**Limit Cycles** — the model traces a closed loop in semantic space, repeating a multi-token phrase. Step-by-step velocity remains high (the model is "moving"), but it is orbiting. Velocity-based monitors are completely blind to this.

Both are measurable in the latent space *before they appear in output.*

---

## The Solution

**ChaosCore** is an unsupervised latent monitor that wraps any HuggingFace CausalLM without modifying it. It hooks into the model's hidden states via a forward hook, tracks trajectory geometry and token-level entropy at each generation step, and applies surgical logit interventions when failure is approaching.

```
Input Prompt
     │
     ▼
┌─────────────────────────────────────────────────┐
│                    Logic Core                   │
│         (standard autoregressive LLM)           │
│  hidden states h_t ──────────────────────────┐  │
└──────────────────────────────────────────────┼──┘
                                               │
                    ┌──────────────────────────▼──┐
                    │         Chaos Core          │
                    │                             │
                    │  AR(1) of token surprise    │
                    │  Latent velocity V(t)       │
                    │  Orbit resonance D(t,k)     │
                    │  SVD entropy                │
                    │  Integrated stress F(t)     │
                    │                             │
                    │  → Attractor classification │
                    │  → Logit intervention       │
                    │  → Semantic anchoring       │
                    └─────────────────────────────┘
```

---

## Scientific Grounding

### Critical Slowing Down (CSD)

Near a phase transition, a dynamical system loses its restoring force. In ecological and climate systems this is well-documented. The entropic drift of the token surprise series follows an Ornstein-Uhlenbeck process:

```
dφ_t = -λ_t · φ_t · dt + σ · dW_t
```

As the system approaches a repetition attractor, resilience λ_t → 0, producing two measurable precursor signals:

- **Variance inflation**: σ²_t ∝ 1/λ_t → ∞
- **Lag-1 autocorrelation rise**: AR(1) → 1.0

Both are detectable several tokens *before* the failure manifests in output. This is the early warning signal.

### Orbit Resonance

A model can circle through N distinct tokens — maintaining high step velocity — while trapped in a periodic orbit. Classical velocity monitors miss this completely.

ChaosCore scans lagged self-similarity:

```
D(t, k) = ||h_t - h_{t-k}|| / sqrt(||h_t|| · ||h_{t-k}||)
```

If D(t, k) < 0.15 for any lag k ≥ 3, the trajectory has closed on itself. Period k is the orbit length. Only those tokens are penalized — not all recent tokens.

### Attractor Taxonomy

| State | Geometric Profile | Signature | Intervention |
|---|---|---|---|
| **HEALTHY** | Broad, non-repeating manifold | SVD entropy ≈ 1.0 | None |
| **CSD_WARNING** | Approaching transition | AR(1) rising → threshold | Temperature modulation + anchor boost |
| **POINT_ATTRACTOR** | Trajectory collapse to fixed point | V_t → 0, S_t → 0 | Dominant token penalty + anchor |
| **LIMIT_CYCLE** | Closed periodic orbit | D(t,k) < 0.15 for stable k | Period-specific orbit penalty + anchor |
| **ENTROPY_COLLAPSE** | Dimensionality loss | SVD entropy dropping | Anchor boost |

---

## Quick Start

```python
from resonator import ChaosCore, ChaosConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

core = ChaosCore(model, tokenizer)
inputs = tokenizer("Your prompt here", return_tensors="pt")

with core:
    output = model.generate(
        **inputs,
        logits_processor=LogitsProcessorList([core]),
        max_new_tokens=200,
        do_sample=True,
    )

print(tokenizer.decode(output[0]))
print(core.summary())
```

---

## Results

### Experiment 1: Baseline Failure
GPT-2 given a prompt designed to induce repetition. No intervention. Vocabulary diversity (unique/total words): **~22%**. Clear repetition loops visible in output.

### Experiment 2: CSD Early Warning
ChaosCore monitoring in detection-only mode (no logit intervention). AR(1) crosses 0.80 an average of **N tokens before** the first 4-gram repetition appears in output. The warning signal precedes the failure.

### Experiment 3: Full Intervention
ChaosCore monitoring + intervention active. Vocabulary diversity: **~58%**. Coherent generation maintained throughout. No visible repetition loops.

### Experiment 4: Ablation
Each component removed individually:

| Configuration | Vocab Diversity |
|---|---|
| Full Chaos Core | ~58% |
| No AR(1) detection | ~41% |
| No orbit resonance | ~44% |
| No semantic anchoring | ~39% |
| No neuromodulation | ~35% |
| No intervention (detect only) | ~22% |

Each component contributes independently. The full system outperforms any single-component version.

### Experiment 5: Orbit Resonance vs. Velocity-Only
A limit cycle prompt where the model orbits a short phrase at high step velocity. Velocity-only monitoring: fails to detect (**~24%** diversity). Full orbit resonance: detects and breaks the cycle (**~51%** diversity).

---

## Installation

```bash
pip install resonator-ai
# or from source:
git clone https://github.com/TrialBlazer23/resonator
cd resonator && pip install -e .
```

---

## Demo

Run the full demo in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TrialBlazer23/resonator/blob/main/resonator_demo.ipynb)

No GPU required. Full results in ~8 minutes on CPU.

---

## Design Philosophy

The Resonator principle: an AI should detect **phase shifts and gradients**, not classify static states. The interesting information is in the *approach* to failure, not the failure itself.

This maps directly onto the physical intuition of the Aharonov-Bohm effect: the A-field (potential) is present and measurable in regions where the B-field (the observable force) is zero. The signal exists before the consequence.

ChaosCore operationalizes this: the hidden-state trajectory is diverging before the output loops. The gradient is the warning.

---

## Limitations and Future Work

**Current limitations:**
- Validated on GPT-2 (small transformer). Hidden-state dimensionality and trajectory behavior at scale (7B+) needs verification.
- Prompt injection detection (via the same CSD framework) is theoretically grounded but not yet empirically validated.
- Orbit resonance threshold (0.15) was tuned on GPT-2; may need calibration per architecture.

**Next steps:**
- Scale to GPT-2-medium, GPT-Neo, and a 7B model
- Adversarial prompt injection benchmark (using the KA-PROMPT framework as a baseline)
- Adaptive threshold calibration per model architecture
- Real-time dashboard for long-form generation monitoring

---

## About

Built by Evan / Necessity Labs.

The Resonator framework originated from experiments in on-device ML for Android (Aletheia, MAICar) and anomaly detection (Smoke Signals / Resonator Lab). The LLM safety application emerged from applying the same phase-transition detection principles to language model generation dynamics.

---

*MIT License*
