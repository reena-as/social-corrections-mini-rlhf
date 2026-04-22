# Social Corrections Mini-RLHF — common workflows.
#
# Most targets don't depend on a running Tinker account; those that do are
# clearly labeled with "(tinker)". The pipeline is:
#
#     make install          install the package + core dependencies
#     make build-jsonl      convert seed correction_dataset.json to JSONL splits
#     make test             run the offline test suite
#     make train-sft        (tinker) run SFT on Tinker
#     make train-dpo        (tinker) run DPO on Tinker
#     make eval-base        run all three eval layers against GPT-4o-mini base
#     make eval-rule-based  run all three eval layers against the rule-based system
#     make plots            plot final comparison bars from all eval outputs

PY ?= python
MODEL ?= meta-llama/Llama-3.1-8B-Instruct
OPENAI_MODEL ?= gpt-4o-mini
PARTNER_MODEL ?= gpt-4o
JUDGE_MODEL ?= gpt-4o
EPISODES ?= 5
MMLU_N ?= 200

.PHONY: install test build-jsonl adapt-pr harvest-sotopia merge train-sft train-dpo \
        classify-train eval-base eval-rule-based eval-sft eval-dpo plots clean

install:
	pip install -e ".[dev,data,classifier,inference,plot]"

test:
	pytest -q

# ---------- data pipeline ----------

build-jsonl:
	$(PY) scripts/build_jsonl.py

adapt-pr:
	$(PY) scripts/adapt_politerewrite.py

harvest-sotopia:
	$(PY) scripts/harvest_sotopia.py \
	    --agent-model $(OPENAI_MODEL) --partner-model $(PARTNER_MODEL) \
	    --judge-model $(JUDGE_MODEL) --writer-model $(JUDGE_MODEL)

merge:
	$(PY) scripts/merge_datasets.py

# ---------- classifier ----------

classify-train:
	$(PY) -m social_corrections.evaluation.politeness_classifier train \
	    --corpus data/raw/stanford_politeness.csv --out-path models/politeness_clf.pkl

# ---------- training (tinker) ----------

train-sft:
	$(PY) scripts/train_sft.py \
	    --train-path data/processed/train_sft.jsonl \
	    --val-path   data/processed/val_sft.jsonl \
	    --model-name $(MODEL) --lora-rank 32 --learning-rate 1e-4 \
	    --num-epochs 3 --batch-size 8 \
	    --output-name social-corrections-sft

train-dpo:
	$(PY) scripts/train_dpo.py \
	    --train-path data/processed/train_dpo.jsonl \
	    --val-path   data/processed/val_dpo.jsonl \
	    --model-name $(MODEL) --lora-rank 32 --learning-rate 1e-5 \
	    --num-epochs 2 --batch-size 4 --dpo-beta 0.1 \
	    --output-name social-corrections-dpo

# ---------- evaluation ----------

eval-base:
	$(PY) scripts/run_all_eval.py --system base --model $(OPENAI_MODEL) \
	    --partner-model $(PARTNER_MODEL) --judge-model $(JUDGE_MODEL) \
	    --episodes-per-scenario $(EPISODES) --mmlu-n $(MMLU_N) \
	    --out-dir data/processed/eval_base

eval-rule-based:
	$(PY) scripts/run_all_eval.py --system rule_based --model $(OPENAI_MODEL) \
	    --partner-model $(PARTNER_MODEL) --judge-model $(JUDGE_MODEL) \
	    --episodes-per-scenario $(EPISODES) --mmlu-n $(MMLU_N) \
	    --out-dir data/processed/eval_rule_based

# For --system tinker, set TINKER_MODEL_PATH on the command line:
#   make eval-sft TINKER_MODEL_PATH=<path-from-sft-summary.json>
eval-sft:
	$(PY) scripts/run_all_eval.py --system tinker \
	    --tinker-model-path $(TINKER_MODEL_PATH) \
	    --tinker-base-model $(MODEL) \
	    --partner-model $(PARTNER_MODEL) --judge-model $(JUDGE_MODEL) \
	    --episodes-per-scenario $(EPISODES) --mmlu-n $(MMLU_N) \
	    --out-dir data/processed/eval_sft

eval-dpo:
	$(PY) scripts/run_all_eval.py --system tinker \
	    --tinker-model-path $(TINKER_MODEL_PATH) \
	    --tinker-base-model $(MODEL) \
	    --partner-model $(PARTNER_MODEL) --judge-model $(JUDGE_MODEL) \
	    --episodes-per-scenario $(EPISODES) --mmlu-n $(MMLU_N) \
	    --out-dir data/processed/eval_dpo

plots:
	$(PY) scripts/plot_results.py \
	    --base-dir data/processed/eval_base \
	    --rule-dir data/processed/eval_rule_based \
	    --sft-dir  data/processed/eval_sft \
	    --dpo-dir  data/processed/eval_dpo \
	    --out-dir  data/processed/plots

clean:
	rm -rf data/processed __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache *.egg-info build dist
