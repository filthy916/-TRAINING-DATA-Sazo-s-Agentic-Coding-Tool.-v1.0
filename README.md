# Sazo's Agentic Coding Tool — Training Data v1.0

A complete training data suite for a **goal-driven, self-correcting agentic coding system**. This repository packages multi-turn supervised fine-tuning (SFT) examples, a novel gradient algorithm, and a number-theoretic weight initialisation scheme — all derived from first principles.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/filthy916/-TRAINING-DATA-Sazo-s-Agentic-Coding-Tool.-v1.0.git
cd -TRAINING-DATA-Sazo-s-Agentic-Coding-Tool.-v1.0

# 2. Install dependencies
pip install -r requirements.txt

# 3. Load the pre-built training data (ready to use immediately)
python - <<'EOF'
import json
with open("data/coding_agent_train.jsonl") as f:
    examples = [json.loads(line) for line in f if line.strip()]
print(f"Loaded {len(examples)} training examples")
for ex in examples:
    print(f"  {ex['id']:20s}  task={ex['task_type']:22s}  reward={ex['reward_signal']['reward']}")
EOF

# 4. (Optional) Regenerate data files from source
python training_data.py
```

---

## Repository Contents

| File | Description |
|---|---|
| `training_data.py` | Python source — defines all training examples and exports to `data/` |
| `rcg_training.py` | Residual Compression Gradient (RCG) — novel training algorithm |
| `primitive_root_transfer.py` | Primitive Root Transfer Resonance — number-theory weight initialisation |
| `data/coding_agent_train.jsonl` | JSONL export — one training example per line, ready for SFT pipelines |
| `data/coding_agent_train_pretty.json` | Pretty-printed JSON array (human-readable) |
| `requirements.txt` | Python dependencies (`numpy`, `torch`) |

---

## Training Data Format

Each example in `data/coding_agent_train.jsonl` follows this schema:

```json
{
  "id": "unique_task_id",
  "task_type": "<one of 7 types>",
  "turns": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "reward_signal": {
    "reward": 1.0
  },
  "metadata": {
    "difficulty": "easy | medium | hard | expert",
    "self_corrected": true,
    "task_tokens": 890
  }
}
```

All examples are **multi-turn** and **execution-grounded** — assistant turns include explicit tool calls, their outputs, and self-correction loops where applicable.

---

## Task Types (7 total)

| # | Task Type | ID Prefix | Description |
|---|---|---|---|
| 1 | `bug_localization` | `bug_loc_*` | Identify the exact file, function, and line(s) containing a reported bug |
| 2 | `patch_generation` | `patch_gen_*` | Generate a minimal correct diff; iterate on test failures until all tests pass |
| 3 | `test_generation` | `test_gen_*` | Write comprehensive test suites covering edge cases and regressions |
| 4 | `multi_file_refactor` | `refactor_*` | Refactor across multiple files while preserving behaviour and passing tests |
| 5 | `error_diagnosis` | `error_diag_*` | Diagnose runtime/compiler errors from stack traces and reproduce them |
| 6 | `tool_use` | `tool_use_*` | Orchestrate shell tools (bash, grep, git, pytest) to complete a coding task |
| 7 | `self_correction` | `self_correct_*` | Detect, diagnose, and revert a regression introduced by the agent's own prior patch |

---

## Reward Signals

Each example carries a structured reward signal used in **Stage 3 RLEF** (Reinforcement Learning from Execution Feedback):

- **`tests_passed` / `tests_total`** — execution-verified pass rate
- **`localization_correct`** — boolean for bug-finding tasks
- **`self_corrected`** — whether the agent detected and fixed its own mistake
- **`reward`** — scalar reward (1.0–1.8; higher for harder tasks and successful self-correction; maximum 1.8 for expert-level self-correction with correct root-cause diagnosis)

---

## Algorithms

### Residual Compression Gradient (RCG) — `rcg_training.py`

A structurally novel training procedure grounded in **algorithmic information theory**. It is not a variant of Adam, SGD, GRPO, PPO, DPO, or any published algorithm.

**Core idea:** a model that has genuinely learned should be able to *compress* its future predictions using its current weights as a codebook. RCG optimises:

```
min  K(θ) + |X : θ| − δ(θ, X_future)
```

where:
- `K(θ)` — approximated Kolmogorov complexity of the weight state
- `|X : θ|` — description length of training data given θ (the MDL fit term)
- `δ(θ, X_next)` — **compression bonus**: how much shorter the next batch is described using `θ_t` versus `θ_{t-1}`

The `δ` term makes the update signal not "did we reduce loss now?" but **"did our weight update make future data more compressible?"** — inverting the usual signal flow.

**Key properties:**
- No human-defined loss function; the compression oracle *is* the loss
- Penalises weight updates that do not improve future generalisation
- Will not converge on incompressible (pure-noise) data — by design
- Gradient is approximated via bits-back coding (VAE-style reparameterisation)

---

### Primitive Root Transfer Resonance — `primitive_root_transfer.py`

A weight-initialisation and learning-rate scheduling scheme built on **primitive root mathematics**.

`g` is a primitive root mod `p` iff `{g¹, g², …, g^(p−1)} = {1, 2, …, p−1} mod p` — the orbit covers the entire multiplicative group `(ℤ/pℤ)*` with no repetition.

**Modules provided:**

| Module | Purpose |
|---|---|
| `is_prime` | Miller-Rabin primality test (deterministic for n < 3.2×10¹⁸ per the implementation's witness set) |
| `euler_totient` | φ(n) — count of integers coprime to n |
| `prime_factors` | Distinct prime factors of n |
| `primitive_root_init()` | Equidistributed weight initialisation — eliminates neuron clustering |
| `primitive_root_lr_schedule()` | LR schedule derived from the root orbit |
| `resonance_coupling_matrix()` | Layer-coupling matrix for transfer tasks |

**Transfer resonance intuition:** standard random initialisation places neurons at Gaussian-distributed positions — clustering is likely. Primitive root initialisation maps weight values to the orbit of `g mod p` (where `p` is chosen relative to layer size), guaranteeing equidistribution within that modulus and reducing representation collapse on transfer tasks. For layers larger than `p−1`, the orbit wraps cyclically, maintaining uniform coverage across residue classes.

This module is designed to integrate with `rcg_training.py`.

---

## Usage

### Load training data in a pipeline

```python
import json

with open("data/coding_agent_train.jsonl") as f:
    examples = [json.loads(line) for line in f if line.strip()]

# Each example: id, task_type, turns, reward_signal, metadata
for ex in examples:
    messages = ex["turns"]          # list of {"role": ..., "content": ...}
    reward   = ex["reward_signal"]["reward"]
    print(ex["id"], reward)
```

### Regenerate data files from source

```python
python training_data.py
# outputs:
#   data/coding_agent_train.jsonl        (JSONL, one example per line)
#   data/coding_agent_train_pretty.json  (pretty-printed JSON array)
```

### Use RCG as the training loop

```python
# See rcg_training.py for full implementation
from rcg_training import RCGTrainer

trainer = RCGTrainer(model, compress_fn=your_compressor)
trainer.train(dataloader)
```

### Use primitive root initialisation

```python
# See primitive_root_transfer.py for full implementation
from primitive_root_transfer import primitive_root_init, resonance_coupling_matrix

model = MyModel()
primitive_root_init(model)  # equidistributed weight init
C = resonance_coupling_matrix(layer_sizes=[256, 512, 256])
```

---

## Training Pipeline Stages

| Stage | Type | Data Source |
|---|---|---|
| Stage 1 | Pre-training | Not included (use standard code corpora) |
| Stage 2 | Supervised Fine-Tuning (SFT) | `data/coding_agent_train.jsonl` |
| Stage 3 | RLEF | `reward_signal` fields in each example + live execution feedback |

---

## Dataset Statistics

```
Total examples     : 7
Task type coverage : 7/7
Self-corrected     : 3/7
Difficulty range   : hard → expert
Reward range       : 1.0 – 1.8
Avg reward         : 1.44
```

---

## Requirements

- Python 3.10+
- `numpy>=1.24` (primitive_root_transfer.py)
- `torch>=2.0` (rcg_training.py)

Install with:

```bash
pip install -r requirements.txt
```

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute with attribution.

