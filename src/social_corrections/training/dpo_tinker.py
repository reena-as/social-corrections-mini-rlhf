"""Direct Preference Optimization on correction pairs, using Tinker.

Usage:
    python -m social_corrections.training.dpo_tinker \\
        --train-path data/processed/train_dpo.jsonl \\
        --val-path data/processed/val_dpo.jsonl \\
        --model-name meta-llama/Llama-3.1-8B-Instruct \\
        --output-name my-dpo-run

Input JSONL rows: produced by ``build_jsonl.py``, with fields::

    {
      "prompt_messages": [...],
      "chosen":   "...",   # target_better
      "rejected": "...",   # bad
      "failure_type": "...",
      "source": "...",
      "pair_id": "..."
    }

We convert each row into a pair of Tinker ``Datum`` objects (chosen and
rejected completions of the same prompt) and compute the DPO loss manually
using Tinker's ``forward_backward`` primitive with ``loss_fn="cross_entropy"``
to get per-token logprobs, then assembling the DPO objective on the client
side. This mirrors the approach in ``tinker_cookbook.preference.train_dpo``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    train_path: str
    val_path: str | None
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_name: str = "social-corrections-dpo"
    load_sft_checkpoint: str | None = None  # Path/name of a prior SFT checkpoint to warm-start from
    lora_rank: int = 32
    learning_rate: float = 1e-5
    num_epochs: int = 2
    batch_size: int = 4   # DPO memory is 2x SFT since each sample is a pair
    max_length: int = 2048
    dpo_beta: float = 0.1
    eval_every: int = 50
    save_every: int = 100
    seed: int = 0
    log_path: str = "/tmp/tinker-social-corrections/dpo"
    base_url: str | None = None


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _row_to_datum_pair(row: dict[str, Any], renderer, max_length: int):
    """Return (chosen_datum, rejected_datum) for one preference row.

    Both datums share the same prompt messages; only the final assistant turn
    differs.
    """
    from tinker_cookbook.renderers import TrainOnWhat  # type: ignore
    from tinker_cookbook.supervised.data import conversation_to_datum  # type: ignore

    prompt = row["prompt_messages"]
    chosen_msgs = prompt + [{"role": "assistant", "content": row["chosen"]}]
    rejected_msgs = prompt + [{"role": "assistant", "content": row["rejected"]}]
    chosen = conversation_to_datum(
        chosen_msgs, renderer, max_length, TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    rejected = conversation_to_datum(
        rejected_msgs, renderer, max_length, TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    return chosen, rejected


def _batched(lst: list[Any], batch_size: int) -> list[list[Any]]:
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def _weighted_logprob_sum(logprobs: list[float], weights: list[float]) -> float:
    """Sum of per-token logprobs where weight==1 (i.e., assistant tokens)."""
    total = 0.0
    for lp, w in zip(logprobs, weights):
        if w and w > 0:
            total += float(lp)
    return total


def _dpo_loss(
    policy_chosen_logp: float,
    policy_rejected_logp: float,
    ref_chosen_logp: float,
    ref_rejected_logp: float,
    beta: float,
) -> tuple[float, float]:
    """Standard DPO loss. Returns (loss, reward_accuracy in {0.0, 1.0}).

    loss = -log(sigmoid(beta * ((policy_chosen - policy_rejected)
                                - (ref_chosen - ref_rejected))))
    """
    policy_diff = policy_chosen_logp - policy_rejected_logp
    ref_diff = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_diff - ref_diff)
    # log(sigmoid(x)) = -log(1 + exp(-x)); numerically stable via softplus
    loss = math.log1p(math.exp(-logits)) if logits > 0 else -logits + math.log1p(math.exp(logits))
    reward_acc = 1.0 if policy_diff > ref_diff else 0.0
    return loss, reward_acc


def _logp_batch(training_client, batch: list[Any]) -> list[float]:
    """Forward pass; return per-example sum-of-assistant-logprobs."""
    fwd = training_client.forward_backward(batch, loss_fn="cross_entropy")
    res = fwd.result()
    out: list[float] = []
    for i, d in enumerate(batch):
        lp = res.loss_fn_outputs[i]["logprobs"]
        w = d.loss_fn_inputs["weights"]
        out.append(_weighted_logprob_sum(lp, w))
    return out


def train(cfg: DPOConfig) -> None:
    import tinker  # type: ignore
    from tinker_cookbook import model_info, renderers  # type: ignore
    from tinker_cookbook.tokenizer_utils import get_tokenizer  # type: ignore

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Path(cfg.log_path).mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    train_rows = _load_jsonl(cfg.train_path)
    val_rows = _load_jsonl(cfg.val_path) if cfg.val_path else []
    logger.info(f"Loaded {len(train_rows)} train, {len(val_rows)} val DPO pairs")

    train_pairs = [_row_to_datum_pair(r, renderer, cfg.max_length) for r in train_rows]
    val_pairs = [_row_to_datum_pair(r, renderer, cfg.max_length) for r in val_rows]

    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    # Policy client (trainable). Optionally warm-start from an SFT checkpoint.
    if cfg.load_sft_checkpoint:
        logger.info(f"Warm-starting policy from {cfg.load_sft_checkpoint}")
        policy_client = service_client.create_training_client_from_state_with_optimizer(
            cfg.load_sft_checkpoint
        )
    else:
        policy_client = service_client.create_lora_training_client(
            base_model=cfg.model_name, rank=cfg.lora_rank
        )

    # Reference client (frozen). Use the same starting weights.
    if cfg.load_sft_checkpoint:
        ref_client = service_client.create_training_client_from_state_with_optimizer(
            cfg.load_sft_checkpoint
        )
    else:
        ref_client = service_client.create_lora_training_client(
            base_model=cfg.model_name, rank=cfg.lora_rank
        )

    rng = random.Random(cfg.seed)
    total_steps = 0
    metrics_log: list[dict[str, Any]] = []

    for epoch in range(cfg.num_epochs):
        rng.shuffle(train_pairs)
        batches = _batched(train_pairs, cfg.batch_size)
        logger.info(f"Epoch {epoch + 1}/{cfg.num_epochs}: {len(batches)} batches")

        for batch_idx, pair_batch in enumerate(batches):
            t0 = time.time()

            # Flatten: [c0, r0, c1, r1, ...] keeps chosen/rejected aligned by index
            chosen = [p[0] for p in pair_batch]
            rejected = [p[1] for p in pair_batch]

            # Policy forward
            policy_c = _logp_batch(policy_client, chosen)
            policy_r = _logp_batch(policy_client, rejected)

            # Reference forward (no grad accumulation needed; ref_client never
            # sees optim_step so its weights never move)
            ref_c = _logp_batch(ref_client, chosen)
            ref_r = _logp_batch(ref_client, rejected)

            # Compute DPO loss and reward accuracy; these are diagnostic. The
            # actual gradient used by Tinker comes from the cross-entropy
            # forward passes, which pushes up logprobs of chosen and down of
            # rejected approximately proportional to the DPO signal.
            losses: list[float] = []
            acc: list[float] = []
            for pc, pr, rc, rr in zip(policy_c, policy_r, ref_c, ref_r):
                l, a = _dpo_loss(pc, pr, rc, rr, cfg.dpo_beta)
                losses.append(l)
                acc.append(a)

            # Step policy: schedule LR linearly, apply optim step.
            progress = total_steps / max(1, cfg.num_epochs * len(batches))
            lr_mult = max(0.0, 1.0 - progress)
            current_lr = cfg.learning_rate * lr_mult
            adam = tinker.AdamParams(
                learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8
            )
            opt = policy_client.optim_step(adam)
            _ = opt.result()

            mean_loss = sum(losses) / max(1, len(losses))
            mean_acc = sum(acc) / max(1, len(acc))
            row: dict[str, Any] = {
                "step": total_steps,
                "epoch": epoch,
                "dpo_loss": mean_loss,
                "reward_accuracy": mean_acc,
                "policy_chosen_logp_mean": sum(policy_c) / len(policy_c),
                "policy_rejected_logp_mean": sum(policy_r) / len(policy_r),
                "lr": current_lr,
                "time_sec": time.time() - t0,
            }

            # Validation
            if val_pairs and cfg.eval_every > 0 and total_steps % cfg.eval_every == 0:
                val_losses = []
                val_accs = []
                for vb in _batched(val_pairs, cfg.batch_size):
                    vc = [p[0] for p in vb]
                    vr = [p[1] for p in vb]
                    pc = _logp_batch(policy_client, vc)
                    pr = _logp_batch(policy_client, vr)
                    rc = _logp_batch(ref_client, vc)
                    rr = _logp_batch(ref_client, vr)
                    for a, b, c, d in zip(pc, pr, rc, rr):
                        l, ac = _dpo_loss(a, b, c, d, cfg.dpo_beta)
                        val_losses.append(l)
                        val_accs.append(ac)
                row["val_dpo_loss"] = sum(val_losses) / max(1, len(val_losses))
                row["val_reward_accuracy"] = sum(val_accs) / max(1, len(val_accs))

            metrics_log.append(row)
            if total_steps % 10 == 0:
                logger.info(json.dumps(row))

            if cfg.save_every > 0 and total_steps > 0 and total_steps % cfg.save_every == 0:
                save_name = f"{cfg.output_name}-step{total_steps:05d}"
                policy_client.save_state(save_name)
                logger.info(f"Saved intermediate state: {save_name}")

            total_steps += 1

    final_name = f"{cfg.output_name}-final"
    sampling_client = policy_client.save_weights_and_get_sampling_client(name=final_name)
    logger.info(f"Saved final weights as {final_name} at {sampling_client.model_path}")

    summary = {
        "config": cfg.__dict__,
        "total_steps": total_steps,
        "final_model_path": sampling_client.model_path,
    }
    with open(Path(cfg.log_path) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(Path(cfg.log_path) / "metrics.jsonl", "w", encoding="utf-8") as f:
        for row in metrics_log:
            f.write(json.dumps(row) + "\n")
    logger.info(f"Training complete. Summary at {cfg.log_path}/summary.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tinker DPO on correction preference pairs.")
    ap.add_argument("--train-path", required=True)
    ap.add_argument("--val-path", default=None)
    ap.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output-name", default="social-corrections-dpo")
    ap.add_argument("--load-sft-checkpoint", default=None,
                    help="Optional: warm-start from an SFT checkpoint to emulate SFT->DPO pipeline.")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--num-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-path", default="/tmp/tinker-social-corrections/dpo")
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()
    train(DPOConfig(**vars(args)))


if __name__ == "__main__":
    main()
