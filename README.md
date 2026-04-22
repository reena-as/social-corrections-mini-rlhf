# Social Corrections Mini-RLHF

**JHU EN.601.773 Machine Social Intelligence — Spring 2026 project**
Reena Assassa, Winston Li

Learning polite task-oriented communication from natural-language corrections, treated as an RLHF-style training signal. Evaluates whether small sets of corrections can improve a language agent's social behavior across multi-turn SOTOPIA-style interactions, without harming general task competence.

Pitch in one sentence: we take socially flawed assistant replies, collect natural-language rewrites of them, and train an 8B open-source model on Tinker using both SFT and DPO on those rewrites — then see whether the trained agent stays socially appropriate over long-horizon role-plays.

---

## Four systems compared

| System | What it is | Where it lives |
|---|---|---|
| **A — Base model** | Unmodified 8B instruction-tuned open model | `inference.OpenAIModelClient` (or `HFModelClient`) |
| **B — Rule-based** | Base model + your existing deterministic politeness post-processor | `rule_based.RuleBasedRewriter` wrapping System A |
| **C — SFT** | Base model fine-tuned on correction pairs via Tinker LoRA | `training.sft_tinker` |
| **D — DPO** | Base model trained via DPO on `(bad, better)` preference pairs on Tinker | `training.dpo_tinker` |

Every system is accessed through the same `BaseModelClient.chat(...)` interface, so the evaluation scripts don't have to care which one they're talking to.

---

## Three evaluation layers

| Layer | What it measures | Script |
|---|---|---|
| **L1 — Sentence-level politeness** | Heuristic + Stanford-Politeness classifier scores on held-out correction prompts | `evaluation.sentence_level_eval` |
| **L2 — SOTOPIA multi-turn** | SOTOPIA-Eval seven-dimension scores across 5–10 scenarios × N episodes | `evaluation.sotopia_eval` |
| **L3 — MMLU** | Task-competence no-regression check | `evaluation.mmlu_eval` |

The paper reports per-dimension L2 scores with special attention to `goal` (did capability regress?) vs. `social_rules` + `relationship` (did we get the intended gain?).

---

## Quick start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data,classifier,inference,plot]"
```

Then copy `.env.example` → `.env` and fill in `TINKER_API_KEY` and `OPENAI_API_KEY`.

Tinker itself installs via the [Tinker docs](https://tinker-docs.thinkingmachines.ai/). You also need `tinker-cookbook` for renderers and the SL data helpers:

```bash
pip install tinker
pip install -e "git+https://github.com/thinking-machines-lab/tinker-cookbook@main#egg=tinker_cookbook"
```

### 2. Sanity-check the install

```bash
pytest -q
```

The offline test suite checks taxonomy, schema, JSONL build, rule-based rewriter, and heuristic scorer — no network, no Tinker, no API keys.

### 3. Build the training JSONL from the seed dataset

```bash
python scripts/build_jsonl.py
# -> data/processed/{train,val,test}_{sft,dpo}.jsonl
```

### 4. Scale the dataset (optional but recommended)

```bash
# If you have a local PoliteRewrite dump at data/raw/politerewrite.json:
python scripts/adapt_politerewrite.py --local data/raw/politerewrite.json \
    --max-examples 500 \
    --out data/processed/politerewrite_adapted.json

# Harvest in-distribution flagged turns from SOTOPIA-style role-plays
# (requires OPENAI_API_KEY):
python scripts/harvest_sotopia.py --episodes-per-scenario 3

# Merge everything and re-build the JSONL splits from the larger pool:
python scripts/merge_datasets.py
python scripts/build_jsonl.py --input data/processed/correction_pairs_all.json
```

Hand-review `data/processed/sotopia_flagged_candidates.json` before promoting the drafts into training pairs; the LLM-written corrections are a starting point, not a finished label.

### 5. Train on Tinker

```bash
# SFT first (cheaper, quicker feedback loop)
python scripts/train_sft.py \
    --train-path data/processed/train_sft.jsonl \
    --val-path   data/processed/val_sft.jsonl \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --lora-rank 32 --learning-rate 1e-4 \
    --num-epochs 3 --batch-size 8 \
    --output-name social-corrections-sft

# Then DPO
python scripts/train_dpo.py \
    --train-path data/processed/train_dpo.jsonl \
    --val-path   data/processed/val_dpo.jsonl \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --lora-rank 32 --learning-rate 1e-5 \
    --num-epochs 2 --batch-size 4 --dpo-beta 0.1 \
    --output-name social-corrections-dpo

# Optional: SFT -> DPO pipeline, warm-starting DPO from the SFT checkpoint
python scripts/train_dpo.py \
    ...
    --load-sft-checkpoint <path-from-sft-summary.json>
```

Each run drops a `summary.json` with the final Tinker `model_path`. Pass that into the eval scripts.

### 6. Evaluate

```bash
# Base model
python scripts/run_all_eval.py --system base --model gpt-4o-mini \
    --partner-model gpt-4o --judge-model gpt-4o \
    --episodes-per-scenario 5 --mmlu-n 200 \
    --out-dir data/processed/eval_base

# Rule-based (System B)
python scripts/run_all_eval.py --system rule_based --model gpt-4o-mini \
    --partner-model gpt-4o --judge-model gpt-4o \
    --episodes-per-scenario 5 --mmlu-n 200 \
    --out-dir data/processed/eval_rule_based

# SFT model
python scripts/run_all_eval.py --system tinker \
    --tinker-model-path <model_path_from_sft_summary> \
    --tinker-base-model meta-llama/Llama-3.1-8B-Instruct \
    --partner-model gpt-4o --judge-model gpt-4o \
    --episodes-per-scenario 5 --mmlu-n 200 \
    --out-dir data/processed/eval_sft

# DPO model (same pattern, different --tinker-model-path)

# Plots comparing all four systems
python scripts/plot_results.py \
    --base-dir data/processed/eval_base \
    --rule-dir data/processed/eval_rule_based \
    --sft-dir  data/processed/eval_sft \
    --dpo-dir  data/processed/eval_dpo \
    --out-dir  data/processed/plots
```

---

## How this addresses the midway feedback

The instructor flagged two things:

**"The connection between what you have been working on and the topics covered in class is somewhat weak."**
The revised pipeline explicitly instantiates three course threads at once. Correction pairs are treated as natural-language human feedback (*RLHF*, Mar 31 readings) and turned into SFT + DPO training signals. The corrections themselves are a mechanism of *social learning from language* (Apr 7 readings, esp. *Yell at Your Robot* and *How to talk so AI will learn*). Every citation we'd need for the Related Work section is already on the syllabus.

**"Go beyond simple conversation — how to engage in task-oriented communication while maintaining politeness. Recall SOTOPIA."**
The training data is harvested from multi-turn role-play in task-oriented scenarios (negotiation, delivering bad news, code review disagreement, tutoring, etc.), not from isolated single-turn politeness prompts. Evaluation is done with SOTOPIA-Eval's seven-dimension rubric across those same scenarios. The paper's central finding is whether the trained agent maintains goal-completion while improving on the social-rule and relationship dimensions — i.e., whether it solves the long-horizon consistency problem the instructor named.

---

## Failure-type taxonomy

Every training example and every judge call uses the same six canonical labels:

| Short tag | Label |
|---|---|
| `harsh` | Overly Harsh or Judgmental Language |
| `overconfident` | Overconfidence / Lack of Uncertainty |
| `no_ack` | Lack of Acknowledgment |
| `commanding` | Direct or Commanding Tone |
| `negative_framing` | Negative Framing Instead of Constructive Framing |
| `cold` | Emotional Mismatch (Too Cold or Abrupt) |

Full descriptions are in `src/social_corrections/taxonomy.py`. They map directly onto the error categories defined in the midway report.

---

## Compute budget (for Tinker's $250 student credit)

Rough expectations for Llama-3.1-8B-Instruct at LoRA rank 32 on Tinker:

- SFT on ~500 examples × 3 epochs ≈ well under $10.
- DPO on ~500 preference pairs × 2 epochs ≈ under $15 (DPO doubles forward passes per step).
- MMLU / SOTOPIA sampling use Tinker's sampling endpoint or OpenAI directly, budgeted on top.

Start by running everything on Qwen3-4B-Instruct-2507 (≈3× cheaper) to shake out bugs before moving to the 8B model.

---

## Repo layout

```
social-corrections-mini-rlhf/
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .env.example
├── LICENSE
│
├── configs/
│   ├── sft_llama_8b.yaml            # SFT hyperparameters
│   └── dpo_llama_8b.yaml            # DPO hyperparameters
│
├── data/
│   ├── correction_dataset.json      # 15 seed correction pairs (from midway)
│   ├── rule_based_results.json      # rule-based outputs on the seed set
│   ├── prompts/
│   │   ├── failure_judge.txt        # prompt for flagging bad turns during harvest
│   │   ├── correction_writer.txt    # prompt for drafting corrections
│   │   ├── sotopia_judge.txt        # SOTOPIA-Eval seven-dimension rubric
│   │   └── sotopia_scenarios.json   # 8 task-oriented role-play scenarios
│   ├── raw/                         # put Stanford-Politeness CSV, PoliteRewrite dumps here
│   └── processed/                   # all generated JSONL + eval outputs land here
│
├── scripts/
│   ├── build_jsonl.py               # data/correction_dataset.json -> processed/*.jsonl
│   ├── adapt_politerewrite.py       # PoliteRewrite -> CorrectionPairs
│   ├── harvest_sotopia.py           # role-play + judge to find bad turns
│   ├── merge_datasets.py            # dedupe + combine all sources
│   ├── train_sft.py                 # Tinker SFT
│   ├── train_dpo.py                 # Tinker DPO
│   ├── run_all_eval.py              # all three eval layers for one system
│   └── plot_results.py              # bar plots comparing systems
│
├── src/social_corrections/
│   ├── taxonomy.py
│   ├── data/
│   │   ├── schema.py                # CorrectionPair + JSONL helpers
│   │   ├── build_jsonl.py           # build SFT / DPO splits
│   │   ├── politerewrite_adapter.py
│   │   ├── sotopia_harvester.py
│   │   └── merge_datasets.py
│   ├── rule_based/rewriter.py
│   ├── training/
│   │   ├── sft_tinker.py
│   │   └── dpo_tinker.py
│   ├── inference/model_client.py
│   ├── evaluation/
│   │   ├── heuristic_scorer.py
│   │   ├── politeness_classifier.py
│   │   ├── llm_judge.py
│   │   ├── sentence_level_eval.py
│   │   ├── sotopia_eval.py
│   │   └── mmlu_eval.py
│   └── utils/io.py
│
├── notebooks/
│   └── rule_based_baseline.ipynb    # preserved from midway
│
└── tests/
    ├── conftest.py
    ├── test_taxonomy.py
    ├── test_rewriter.py
    ├── test_schema.py
    └── test_heuristic_scorer.py
```

---

## Key references

- Zhou et al., *SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents*, arXiv:2310.11667.
- Wang et al., *SOTOPIA-π: Interactive Learning of Socially Intelligent Language Agents*, arXiv:2403.08715.
- Casper et al., *Open Problems and Fundamental Limitations of RLHF*, arXiv:2307.15217.
- Tien et al., *Causal Confusion and Reward Misidentification in Preference-Based Reward Learning*, arXiv:2204.06601.
- Shi et al., *Yell At Your Robot: Improving On-the-Fly from Language Corrections*, arXiv:2403.12910.
- Sumers et al., *How to talk so AI will learn: Instructions, descriptions, and autonomy*, NeurIPS 2022.
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.
- Danescu-Niculescu-Mizil et al., *A Computational Approach to Politeness with Application to Social Factors* (Stanford Politeness Corpus).
