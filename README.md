# Sazo's Agentic Coding Tool — Training Data v1.0

A collection of structured training data for a goal-driven, self-correcting agentic coding tool. This dataset covers multi-turn task execution, bug localization, code generation, and novel training procedures grounded in algorithmic information theory.

## Repository Contents

| File | Description |
|------|-------------|
| `CODING_AGENT_TRAINING_DATA` | JSONL-ready multi-turn training pairs for the coding agent, covering 7 task types (bug localization, code generation, refactoring, test writing, etc.) with execution-grounded reward signals. Intended for Stage 2 SFT and Stage 3 RLEF training. |
| `CODING_AGENT_JSON` | JSON-formatted version of the coding agent training data for easy programmatic consumption. |
| `CODING_AGENT_TRAIN_PRETTY` | Pretty-printed, human-readable version of the coding agent training data. |
| `RCG_TRAINING` | Residual Compression Gradient (RCG) — a novel training procedure derived from first principles in algorithmic information theory. A model is considered to have truly learned if it can compress its own future predictions using its current weights as a codebook. |
| `Primitive_Root_Transfer` | Primitive Root Algorithm — Transfer Resonance: a transfer-learning method based on primitive root structures. |

## Training Data Format

Each entry in `CODING_AGENT_TRAINING_DATA` / `CODING_AGENT_JSON` follows this schema:

```json
{
  "id": "<unique task id>",
  "task_type": "<one of 7 task types>",
  "turns": [
    { "role": "system", "content": "..." },
    { "role": "user",   "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "reward_signal": "<execution feedback>",
  "metadata": {
    "difficulty": "...",
    "tokens": 0,
    "self_corrected": false,
    "tests_passed": true
  }
}
```

## Task Types Covered

1. Bug Localization
2. Code Generation
3. Refactoring
4. Test Writing
5. Documentation
6. Security Review
7. Performance Optimization

## Usage

This dataset is designed to train and fine-tune agentic coding models that can:
- Reason step-by-step before making changes
- Localize bugs precisely before patching
- Minimize diff size
- Run tests before submitting solutions
- Self-correct based on execution feedback
